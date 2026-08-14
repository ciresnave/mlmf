//! The seam between the format axis, the source axis, and everything above.
//!
//! Format crates implement [`TensorContainer`] and [`MetadataSource`].
//! Source crates implement [`ByteSource`]. Both depend on `mlmf-core`
//! alone, which is what lets any source compose with any format.

use crate::{MetaValue, Result, TensorDescriptor};

/// A contiguous region of bytes obtained from somewhere.
///
/// Implemented by source crates over a memory map, an owned buffer, or a
/// downloaded file. `mlmf-core` performs no I/O and never implements this.
pub trait ByteSource {
    /// The bytes.
    fn as_bytes(&self) -> &[u8];
}

/// Something that declares named tensors and can hand out their bytes.
///
/// Bytes are **borrowed**. There is no tensor type, no device and no
/// backend trait: a consumer builds whatever it wants from the slice.
/// Alignment is not guaranteed — see [`crate::align`].
pub trait TensorContainer {
    /// Every tensor this container declares.
    fn tensors(&self) -> &[TensorDescriptor];

    /// The bytes for one tensor.
    ///
    /// # Errors
    ///
    /// If the descriptor's range lies outside the container's data.
    fn tensor_bytes(&self, descriptor: &TensorDescriptor) -> Result<&[u8]>;
}

/// Something that declares typed metadata under string keys.
///
/// GGUF's in-file metadata and HuggingFace's JSON sidecars both project
/// into this, which is what makes the layer above format-agnostic:
/// config accessors and chat-template extraction are written once.
pub trait MetadataSource {
    /// The value declared under `key`, if any.
    ///
    /// Absent means **not declared**. It never means a default (spec §5).
    fn get(&self, key: &str) -> Option<&MetaValue>;

    /// Every key declared, in unspecified order.
    fn keys(&self) -> Vec<&str>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Encoding, MetaValue, Shape, TensorDescriptor};
    use std::collections::HashMap;

    struct Fake {
        blob: Vec<u8>,
        tensors: Vec<TensorDescriptor>,
        meta: HashMap<String, MetaValue>,
    }

    impl ByteSource for Fake {
        fn as_bytes(&self) -> &[u8] {
            &self.blob
        }
    }

    impl TensorContainer for Fake {
        fn tensors(&self) -> &[TensorDescriptor] {
            &self.tensors
        }
        fn tensor_bytes(&self, d: &TensorDescriptor) -> crate::Result<&[u8]> {
            let start = usize::try_from(d.bytes.start).expect("fits");
            let end = usize::try_from(d.bytes.end).expect("fits");
            Ok(&self.blob[start..end])
        }
    }

    impl MetadataSource for Fake {
        fn get(&self, key: &str) -> Option<&MetaValue> {
            self.meta.get(key)
        }
        fn keys(&self) -> Vec<&str> {
            self.meta.keys().map(String::as_str).collect()
        }
    }

    fn fake() -> Fake {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        Fake {
            blob: vec![0u8; 32],
            tensors: vec![TensorDescriptor {
                name: "blk.0.attn_q.weight".into(),
                shape: Shape::new([2usize, 4]),
                encoding: Encoding::Dense(DType::F32),
                bytes: 0..32,
            }],
            meta,
        }
    }

    #[test]
    fn a_container_hands_out_borrowed_bytes_not_owned_buffers() {
        let f = fake();
        let d = &f.tensors()[0];
        let bytes = f.tensor_bytes(d).expect("in range");
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes.as_ptr(), f.as_bytes().as_ptr());
    }

    #[test]
    fn one_type_can_satisfy_both_seams() {
        // This is what makes the layer above format-agnostic: GGUF
        // metadata and HF JSON both project into MetadataSource, so
        // accessors are written once and serve both.
        let f = fake();
        assert_eq!(
            f.get("general.architecture").and_then(MetaValue::as_str),
            Some(&"llama".to_string())
        );
        assert!(f.get("absent").is_none());
        assert_eq!(f.keys().len(), 1);
    }

    #[test]
    fn traits_are_object_safe() {
        let f = fake();
        let m: &dyn MetadataSource = &f;
        assert!(m.get("general.architecture").is_some());
    }
}
