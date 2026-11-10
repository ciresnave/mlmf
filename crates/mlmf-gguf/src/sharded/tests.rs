use super::*;
use crate::encode::{GgufTensorToWrite, GgufVersion, encode_gguf};
use crate::shard_set::ShardSetError;
use crate::split::shard_file_names;
use mlmf_core::{MetaValue, MetadataSource};
use mlmf_ggml::GgmlType;

/// One shard file: `tensors` are `(name, 8 bytes)` pairs, each one F32 x 2.
fn shard(no: u16, count: u16, total: i32, name: &str, tensors: &[(&str, [u8; 8])]) -> Vec<u8> {
    let arch = MetaValue::String("llama".into());
    let title = MetaValue::String(name.into());
    let (n, c, t) = (
        MetaValue::U16(no),
        MetaValue::U16(count),
        MetaValue::I32(total),
    );
    let meta: Vec<(&str, &MetaValue)> = vec![
        ("general.architecture", &arch),
        ("general.name", &title),
        ("split.no", &n),
        ("split.count", &c),
        ("split.tensors.count", &t),
    ];
    let write: Vec<GgufTensorToWrite<'_>> = tensors
        .iter()
        .map(|(tname, data)| GgufTensorToWrite {
            name: tname,
            ggml_type: GgmlType::F32,
            dims: &[2],
            data,
        })
        .collect();
    encode_gguf(&meta, &write, GgufVersion::V3).expect("encodes")
}

/// A pattern distinct per tensor, so a tensor read from the wrong shard
/// cannot equal the right one.
fn pat(seed: u8) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (i, b) in out.iter_mut().enumerate() {
        *b = seed.wrapping_mul(17).wrapping_add(i as u8);
    }
    out
}

struct Model {
    names: Vec<String>,
    files: Vec<Vec<u8>>,
}

/// Three shards holding a/b, c, d/e.
fn model() -> Model {
    let names = shard_file_names("m", 3).expect("names");
    let files = vec![
        shard(0, 3, 5, "first", &[("a", pat(1)), ("b", pat(2))]),
        shard(1, 3, 5, "second", &[("c", pat(3))]),
        shard(2, 3, 5, "third", &[("d", pat(4)), ("e", pat(5))]),
    ];
    Model { names, files }
}

fn inputs<'a>(m: &'a Model, order: &[usize]) -> Vec<ShardInput<'a>> {
    order
        .iter()
        .map(|&i| ShardInput::new(&m.names[i], &m.files[i]))
        .collect()
}

#[test]
fn every_tensor_reads_back_the_bytes_of_its_own_shard() {
    let m = model();
    let g = ShardedGguf::parse(&inputs(&m, &[0, 1, 2])).expect("parses");
    assert_eq!(g.shard_count(), 3);
    let want = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)];
    for (name, seed) in want {
        let d = g.tensor(name).unwrap_or_else(|| panic!("{name} declared"));
        let bytes = g.tensor_bytes(d).expect("reads");
        assert_eq!(bytes.as_ref(), pat(seed), "tensor {name}");
    }
}

#[test]
fn input_order_does_not_change_what_is_read() {
    let m = model();
    let g = ShardedGguf::parse(&inputs(&m, &[2, 0, 1])).expect("parses");
    assert_eq!(g.shard_file_name(0), Some(m.names[0].as_str()));
    assert_eq!(g.shard_file_name(2), Some(m.names[2].as_str()));
    let d = g.tensor("e").expect("e");
    assert_eq!(g.tensor_bytes(d).expect("reads").as_ref(), pat(5));
}

#[test]
fn virtual_ranges_are_disjoint_and_each_lies_inside_one_shard() {
    let m = model();
    let g = ShardedGguf::parse(&inputs(&m, &[0, 1, 2])).expect("parses");
    let mut ranges: Vec<_> = g.tensors().iter().map(|d| d.bytes.clone()).collect();
    ranges.sort_by_key(|r| r.start);
    assert!(
        ranges.windows(2).all(|w| w[0].end <= w[1].start),
        "{ranges:?}"
    );
    for d in g.tensors() {
        assert!(g.locate(d.bytes.start, d.bytes.end).is_some(), "{}", d.name);
    }
}

#[test]
fn metadata_is_the_first_shards_and_nothing_is_merged() {
    let m = model();
    let g = ShardedGguf::parse(&inputs(&m, &[2, 1, 0])).expect("parses");
    let name = g.metadata().get("general.name");
    assert_eq!(name, Some(&MetaValue::String("first".into())));
}

#[test]
fn a_missing_shard_is_refused_by_name() {
    let m = model();
    let err = ShardedGguf::parse(&inputs(&m, &[0, 2]))
        .err()
        .expect("refused");
    assert_eq!(
        err,
        ShardedError::Set(ShardSetError::MissingShard { no: 1 })
    );
}

#[test]
fn a_file_that_is_not_a_split_is_refused() {
    let m = model();
    let arch = MetaValue::String("llama".into());
    let plain = encode_gguf(&[("general.architecture", &arch)], &[], GgufVersion::V3).expect("w");
    let mut input = inputs(&m, &[0, 1, 2]);
    input[1] = ShardInput::new(&m.names[1], &plain);
    assert_eq!(
        ShardedGguf::parse(&input).err().expect("refused"),
        ShardedError::NotSplit { shard: 1 }
    );
}

#[test]
fn a_file_name_without_the_numbering_is_refused() {
    let m = model();
    let mut input = inputs(&m, &[0, 1, 2]);
    input[0] = ShardInput::new("model.gguf", &m.files[0]);
    assert_eq!(
        ShardedGguf::parse(&input).err().expect("refused"),
        ShardedError::BadFileName { shard: 0 }
    );
}

#[test]
fn a_name_declared_in_two_shards_keeps_the_first_and_reports_the_second() {
    let names = shard_file_names("m", 2).expect("names");
    let files = [
        shard(0, 2, 2, "x", &[("w", pat(7))]),
        shard(1, 2, 2, "x", &[("w", pat(8))]),
    ];
    let input: Vec<_> = (0..2)
        .map(|i| ShardInput::new(&names[i], &files[i]))
        .collect();
    let g = ShardedGguf::parse(&input).expect("parses");
    assert_eq!(g.tensors().len(), 1);
    let d = g.tensor("w").expect("w");
    assert_eq!(g.tensor_bytes(d).expect("reads").as_ref(), pat(7));
    let declined =
        g.report().entries().iter().any(
            |e| matches!(&e.kind, UnrecognizedKind::TensorDeclined { name, .. } if name == "w"),
        );
    assert!(declined, "the repeat must be reported: {:?}", g.report());
}

#[test]
fn locate_resolves_boundaries_and_refuses_a_crossing() {
    let m = model();
    let g = ShardedGguf::parse(&inputs(&m, &[0, 1, 2])).expect("parses");
    let first_len = m.files[0].len() as u64;
    let last_of_first = g.locate(first_len - 1, first_len).expect("inside shard 0");
    assert_eq!((last_of_first.shard, last_of_first.end), (0, first_len));
    let start_of_second = g.locate(first_len, first_len + 1).expect("inside shard 1");
    assert_eq!((start_of_second.shard, start_of_second.start), (1, 0));
    assert_eq!(g.locate(first_len - 1, first_len + 1), None, "crosses");
    assert_eq!(g.locate(5, 4), None, "inverted");
    assert_eq!(g.locate(u64::MAX - 1, u64::MAX), None, "past the end");
}

#[test]
fn a_descriptor_that_is_not_this_containers_is_an_error_not_a_guess() {
    let m = model();
    let g = ShardedGguf::parse(&inputs(&m, &[0, 1, 2])).expect("parses");
    let mut foreign = g.tensor("a").expect("a").clone();
    foreign.bytes = (1 << 40)..((1 << 40) + 8);
    assert!(g.tensor_bytes(&foreign).is_err());
    // Control: the same descriptor, untouched, reads.
    assert!(g.tensor_bytes(g.tensor("a").expect("a")).is_ok());
}

#[test]
fn a_shard_cut_short_is_refused_not_read_from_the_next_shard() {
    let m = model();
    // The writer pads each tensor to the alignment, so trimming the file's
    // tail may only trim padding. Cut to 4 bytes before tensor b's declared
    // end (shard 0 sits at virtual base 0, so its range is shard-local).
    let whole = ShardedGguf::parse(&inputs(&m, &[0, 1, 2])).expect("parses");
    let b_end = whole.tensor("b").expect("b").bytes.end as usize;
    let cut = &m.files[0][..b_end - 4];
    let mut input = inputs(&m, &[0, 1, 2]);
    input[0] = ShardInput::new(&m.names[0], cut);
    let err = ShardedGguf::parse(&input).err().expect("refused");
    // Exactly this variant: one file keeps such a descriptor and fails at
    // read time, so only the shard check stops its range landing in the
    // next shard's bytes.
    assert_eq!(
        err,
        ShardedError::TensorPastShardEnd {
            shard: 0,
            name: "b".into(),
            end: b_end as u64,
            len: (b_end - 4) as u64,
        }
    );
}
