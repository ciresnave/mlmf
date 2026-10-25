// SPDX-License-Identifier: MIT OR Apache-2.0
//! A stored (uncompressed) zip writer, with the CRC-32 every entry needs.
//!
//! `torch.export.load` reads the archive through PyTorch's stream reader,
//! which is built on miniz and checks each entry's CRC. Entries are stored,
//! never deflated, so a tensor's bytes sit in the file exactly as given and
//! can be borrowed from a mapping. Plain zip (no zip64): an entry or the whole
//! file past 4 GiB is refused rather than written wrongly.

use crate::Pt2Error;

const LIMIT: u64 = 0xFFFF_FFFF;

/// CRC-32 (IEEE), reflected, polynomial 0xEDB88320.
const fn table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 == 1 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

const TABLE: [u32; 256] = table();

/// CRC-32 of `data`.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    !data
        .iter()
        .fold(!0u32, |c, &b| TABLE[usize::from((c as u8) ^ b)] ^ (c >> 8))
}

fn put16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn size32(what: &str, n: usize) -> Result<u32, Pt2Error> {
    u32::try_from(n)
        .ok()
        .filter(|&v| u64::from(v) <= LIMIT)
        .ok_or_else(|| Pt2Error::TooLarge {
            what: what.to_string(),
        })
}

/// One local file header and its data; returns the header's offset.
fn local_entry(out: &mut Vec<u8>, name: &str, data: &[u8]) -> Result<u32, Pt2Error> {
    let offset = size32("archive", out.len())?;
    put32(out, 0x0403_4b50);
    put16(out, 20); // version needed
    put16(out, 0); // flags
    put16(out, 0); // method: stored
    put16(out, 0); // time
    put16(out, 0x21); // date: 1980-01-01
    put32(out, crc32(data));
    put32(out, size32(name, data.len())?);
    put32(out, size32(name, data.len())?);
    put16(
        out,
        u16::try_from(name.len()).map_err(|_| Pt2Error::TooLarge {
            what: name.to_string(),
        })?,
    );
    put16(out, 0); // extra
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(data);
    Ok(offset)
}

fn central_entry(out: &mut Vec<u8>, name: &str, data: &[u8], offset: u32) -> Result<(), Pt2Error> {
    put32(out, 0x0201_4b50);
    put16(out, 20); // made by
    put16(out, 20); // needed
    put16(out, 0);
    put16(out, 0);
    put16(out, 0);
    put16(out, 0x21);
    put32(out, crc32(data));
    put32(out, size32(name, data.len())?);
    put32(out, size32(name, data.len())?);
    put16(out, name.len() as u16);
    put16(out, 0); // extra
    put16(out, 0); // comment
    put16(out, 0); // disk
    put16(out, 0); // internal attrs
    put32(out, 0); // external attrs
    put32(out, offset);
    out.extend_from_slice(name.as_bytes());
    Ok(())
}

/// A zip holding `entries` in order, stored, with correct CRCs.
///
/// # Errors
///
/// [`Pt2Error::TooLarge`] if a size or the archive exceeds what plain zip
/// can hold.
pub fn write_stored_zip(entries: &[(String, Vec<u8>)]) -> Result<Vec<u8>, Pt2Error> {
    let mut out = Vec::new();
    let mut offsets = Vec::with_capacity(entries.len());
    for (name, data) in entries {
        offsets.push(local_entry(&mut out, name, data)?);
    }
    let cd_start = size32("archive", out.len())?;
    for ((name, data), offset) in entries.iter().zip(&offsets) {
        central_entry(&mut out, name, data, *offset)?;
    }
    let cd_len = size32("archive", out.len())? - cd_start;
    let count = u16::try_from(entries.len()).map_err(|_| Pt2Error::TooLarge {
        what: "entry count".to_string(),
    })?;
    put32(&mut out, 0x0605_4b50);
    put16(&mut out, 0);
    put16(&mut out, 0);
    put16(&mut out, count);
    put16(&mut out, count);
    put32(&mut out, cd_len);
    put32(&mut out, cd_start);
    put16(&mut out, 0);
    Ok(out)
}
