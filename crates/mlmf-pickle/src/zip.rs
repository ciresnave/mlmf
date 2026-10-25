// SPDX-License-Identifier: MIT OR Apache-2.0
//! The zip container of a `torch.save` checkpoint: the central directory,
//! and where each entry's bytes sit in the file.
//!
//! Only what a checkpoint needs: the end-of-central-directory record (with
//! its zip64 form), central-directory entries, and the local header's length
//! fields to find an entry's data. Nothing is inflated: `torch.save` stores
//! its entries, and a stored entry is a slice of the file. Every offset and
//! length comes from the file, so each is range-checked before it indexes
//! anything.

use crate::error::PickleError;

const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;
/// The EOCD record's fixed part.
const EOCD_LEN: usize = 22;
/// The largest comment a (non-zip64) EOCD can be followed by.
const MAX_COMMENT: usize = 65_535;
/// A central-directory entry with an empty name, extra and comment.
const CENTRAL_FIXED: usize = 46;
/// Refuse an archive claiming more entries than this.
const MAX_ENTRIES: u64 = 1_000_000;

/// One archive member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipEntry {
    /// The entry's path inside the archive.
    pub name: String,
    /// Compression method: 0 is stored.
    pub method: u16,
    /// Size of the data as stored.
    pub compressed_size: u64,
    /// Size of the data once inflated; equal to `compressed_size` if stored.
    pub size: u64,
    /// Absolute offset of the entry's data in the file.
    pub data_offset: u64,
}

fn bad(detail: &str) -> PickleError {
    PickleError::BadZip {
        detail: detail.to_string(),
    }
}

fn slice(bytes: &[u8], at: u64, len: u64) -> Result<&[u8], PickleError> {
    let end = at.checked_add(len).ok_or_else(|| bad("offset overflow"))?;
    let (s, e) = (usize::try_from(at), usize::try_from(end));
    match (s, e) {
        (Ok(s), Ok(e)) if e <= bytes.len() => Ok(&bytes[s..e]),
        _ => Err(bad("a structure runs past the end of the file")),
    }
}

fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn le64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

/// The central directory's entry count and byte range, zip64-aware.
fn directory_location(bytes: &[u8]) -> Result<(u64, u64), PickleError> {
    let lo = bytes.len().saturating_sub(EOCD_LEN + MAX_COMMENT);
    let eocd = (lo..=bytes.len().saturating_sub(EOCD_LEN))
        .rev()
        .find(|&i| bytes.len() >= i + EOCD_LEN && le32(bytes, i) == EOCD_SIG)
        .ok_or_else(|| bad("no end-of-central-directory record"))?;
    let (count, cd_off) = (
        u64::from(le16(bytes, eocd + 10)),
        u64::from(le32(bytes, eocd + 16)),
    );
    if count != 0xFFFF && cd_off != 0xFFFF_FFFF {
        return Ok((count, cd_off));
    }
    zip64_location(bytes, eocd)
}

fn zip64_location(bytes: &[u8], eocd: usize) -> Result<(u64, u64), PickleError> {
    let loc = eocd
        .checked_sub(20)
        .filter(|&i| le32(bytes, i) == EOCD64_LOCATOR_SIG)
        .ok_or_else(|| bad("zip64 marker without a zip64 locator"))?;
    let rec = slice(bytes, le64(bytes, loc + 8), 56)?;
    if le32(rec, 0) != EOCD64_SIG {
        return Err(bad("zip64 end record has the wrong signature"));
    }
    Ok((le64(rec, 32), le64(rec, 48)))
}

/// The body of the zip64 extra field (id 1), if the extra data holds one.
fn zip64_body(extra: &[u8]) -> Result<Option<&[u8]>, PickleError> {
    let mut at = 0usize;
    while at + 4 <= extra.len() {
        let (id, len) = (le16(extra, at), usize::from(le16(extra, at + 2)));
        let body = extra
            .get(at + 4..at + 4 + len)
            .ok_or_else(|| bad("extra field runs past its entry"))?;
        if id == 1 {
            return Ok(Some(body));
        }
        at += 4 + len;
    }
    Ok(None)
}

/// Read the zip64 extra field's values for the fields that overflowed.
fn zip64_extra(extra: &[u8], need: [bool; 3]) -> Result<[Option<u64>; 3], PickleError> {
    let Some(body) = zip64_body(extra)? else {
        return Ok([None; 3]);
    };
    let mut out = [None; 3];
    let mut at = 0usize;
    for (slot, wanted) in out.iter_mut().zip(need) {
        if wanted {
            let v = body
                .get(at..at + 8)
                .ok_or_else(|| bad("short zip64 extra"))?;
            *slot = Some(le64(v, 0));
            at += 8;
        }
    }
    Ok(out)
}

/// One central-directory entry at `at`; returns it and the next entry's offset.
fn central_entry(bytes: &[u8], at: u64) -> Result<(ZipEntry, u64, u64), PickleError> {
    let fixed = slice(bytes, at, CENTRAL_FIXED as u64)?;
    if le32(fixed, 0) != CENTRAL_SIG {
        return Err(bad("central directory entry has the wrong signature"));
    }
    let (n, x, c) = (
        usize::from(le16(fixed, 28)),
        usize::from(le16(fixed, 30)),
        usize::from(le16(fixed, 32)),
    );
    let var = slice(bytes, at + CENTRAL_FIXED as u64, (n + x + c) as u64)?;
    let name = String::from_utf8(var[..n].to_vec()).map_err(|_| bad("entry name is not UTF-8"))?;
    let (mut csize, mut size, mut local) = (
        u64::from(le32(fixed, 20)),
        u64::from(le32(fixed, 24)),
        u64::from(le32(fixed, 42)),
    );
    let need = [
        size == 0xFFFF_FFFF,
        csize == 0xFFFF_FFFF,
        local == 0xFFFF_FFFF,
    ];
    let [s, c2, l] = zip64_extra(&var[n..n + x], need)?;
    size = s.unwrap_or(size);
    csize = c2.unwrap_or(csize);
    local = l.unwrap_or(local);
    let entry = ZipEntry {
        name,
        method: le16(fixed, 10),
        compressed_size: csize,
        size,
        data_offset: 0,
    };
    Ok((entry, local, at + (CENTRAL_FIXED + n + x + c) as u64))
}

/// The absolute offset of the data of the entry whose local header is at `at`.
fn data_start(bytes: &[u8], at: u64) -> Result<u64, PickleError> {
    let h = slice(bytes, at, 30)?;
    if le32(h, 0) != LOCAL_SIG {
        return Err(bad("local header has the wrong signature"));
    }
    Ok(at + 30 + u64::from(le16(h, 26)) + u64::from(le16(h, 28)))
}

/// Read the archive's central directory.
///
/// # Errors
///
/// [`PickleError::BadZip`] for anything that is not a well-formed archive.
pub fn read_zip_directory(bytes: &[u8]) -> Result<Vec<ZipEntry>, PickleError> {
    let (count, mut at) = directory_location(bytes)?;
    if count > MAX_ENTRIES {
        return Err(PickleError::LimitExceeded {
            limit: "zip entry count",
        });
    }
    let mut out = Vec::new();
    for _ in 0..count {
        let (mut entry, local, next) = central_entry(bytes, at)?;
        entry.data_offset = data_start(bytes, local)?;
        slice(bytes, entry.data_offset, entry.compressed_size)?;
        out.push(entry);
        at = next;
    }
    Ok(out)
}
