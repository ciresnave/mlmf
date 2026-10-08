//! Tensor name mapping for loading HuggingFace models into custom formats
//!
//! This module provides intelligent mapping between HuggingFace naming conventions
//! and target framework naming patterns. It automatically detects model architecture
//! and handles tensor name transformations for various model families.

use crate::error::{Error, Result};

use std::collections::HashMap;

/// The part of a `N.component` layer section after the layer number, for
/// both `model.layers.N.component` and `blk.N.component` conventions.
///
/// Used only to name a component in an error message when it was NOT
/// recognized, so a missing dot (itself a malformed name) falls back to
/// the whole section rather than panicking on an error path.
fn component_suffix(layer_section: &str) -> &str {
    layer_section
        .find('.')
        .map_or(layer_section, |i| &layer_section[i + 1..])
}

/// Detected model architecture from tensor names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// LLaMA family (LLaMA 2/3, TinyLlama, Qwen, Mistral)
    LLaMA,
    /// GPT-2 style models
    GPT2,
    /// GPT-NeoX style models  
    GPTNeoX,
    /// Unknown or unsupported architecture
    Unknown,
}

impl Architecture {
    /// Get human-readable name for the architecture
    pub fn name(self) -> &'static str {
        match self {
            Architecture::LLaMA => "LLaMA",
            Architecture::GPT2 => "GPT-2",
            Architecture::GPTNeoX => "GPT-NeoX",
            Architecture::Unknown => "Unknown",
        }
    }

    /// Get expected tensor pattern examples for this architecture
    pub fn example_patterns(self) -> Vec<&'static str> {
        match self {
            Architecture::LLaMA => vec![
                "model.embed_tokens.weight",
                "model.layers.N.self_attn.q_proj.weight",
                "model.norm.weight",
            ],
            Architecture::GPT2 => vec![
                "transformer.wte.weight",
                "transformer.h.N.attn.c_attn.weight",
                "transformer.ln_f.weight",
            ],
            Architecture::GPTNeoX => vec![
                "gpt_neox.embed_in.weight",
                "gpt_neox.layers.N.attention.query_key_value.weight",
                "gpt_neox.final_layer_norm.weight",
            ],
            Architecture::Unknown => vec!["No patterns available"],
        }
    }

    /// Whether this architecture uses Rotary Position Embeddings (RoPE).
    ///
    /// RoPE computes position information on the fly and stores no position-embedding
    /// table, so memory estimation should not include a position-embedding parameter
    /// term for these architectures.
    pub fn uses_rope(self) -> bool {
        match self {
            Architecture::LLaMA => true, // LLaMA, Mistral, Qwen, SmolLM all use RoPE
            Architecture::GPTNeoX => true, // GPT-NeoX uses RoPE
            Architecture::GPT2 => false, // GPT-2 uses learned absolute embeddings
            Architecture::Unknown => false, // Conservative: assume learned embeddings
        }
    }
}

/// Maps tensor names from HuggingFace format to target format
#[derive(Debug, Clone)]
pub struct TensorNameMapper {
    architecture: Architecture,
    name_map: HashMap<String, String>,
}

impl TensorNameMapper {
    /// Auto-detect architecture and create mapper from tensor names
    ///
    /// # Arguments
    /// * `tensor_names` - List of tensor names from the model
    ///
    /// # Returns
    /// A configured mapper that can transform names between formats
    ///
    /// # Examples
    /// ```rust
    /// use mlmf::name_mapping::{TensorNameMapper, Architecture};
    ///
    /// let names = vec![
    ///     "model.embed_tokens.weight".to_string(),
    ///     "model.layers.0.self_attn.q_proj.weight".to_string(),
    /// ];
    /// let mapper = TensorNameMapper::from_tensor_names(&names)?;
    /// assert_eq!(mapper.architecture(), Architecture::LLaMA);
    /// # Ok::<(), mlmf::Error>(())
    /// ```
    pub fn from_tensor_names(tensor_names: &[String]) -> Result<Self> {
        let architecture = Self::detect_architecture(tensor_names);
        if architecture == Architecture::Unknown {
            return Err(Error::UnknownArchitecture);
        }

        let name_map = Self::build_name_map(architecture, tensor_names)?;

        Ok(Self {
            architecture,
            name_map,
        })
    }

    /// Create mapper with explicit architecture
    ///
    /// Use this when you know the architecture in advance, or want to force
    /// a specific mapping even if detection would suggest otherwise.
    pub fn with_architecture(arch: Architecture, tensor_names: &[String]) -> Result<Self> {
        if arch == Architecture::Unknown {
            return Err(Error::UnknownArchitecture);
        }

        let name_map = Self::build_name_map(arch, tensor_names)?;

        Ok(Self {
            architecture: arch,
            name_map,
        })
    }

    /// Detect model architecture from tensor names
    ///
    /// Uses pattern matching to identify the model family based on
    /// characteristic tensor naming patterns.
    fn detect_architecture(tensor_names: &[String]) -> Architecture {
        // Check for LLaMA pattern: "model.layers.N.self_attn..."
        // Also matches Mistral, Qwen, TinyLlama variants
        if tensor_names.iter().any(|n| {
            (n.contains("model.layers.") && n.contains(".self_attn."))
                || (n.contains("model.embed_tokens.weight"))
                || (n.contains("model.norm.weight"))
        }) {
            return Architecture::LLaMA;
        }

        // Check for GGUF LLaMA pattern: "blk.N.attn_q..."
        if tensor_names.iter().any(|n| {
            n.contains("blk.")
                && (n.contains(".attn_q") || n.contains(".attn_k") || n.contains(".attn_v"))
        }) {
            return Architecture::LLaMA;
        }

        // Check for GPT-2 pattern: "transformer.h.N.attn..."
        if tensor_names.iter().any(|n| {
            (n.contains("transformer.h.") && n.contains(".attn."))
                || n.contains("transformer.wte.weight")
                || n.contains("transformer.ln_f.weight")
        }) {
            return Architecture::GPT2;
        }

        // Check for GPT-NeoX pattern: "gpt_neox.layers.N.attention..."
        if tensor_names.iter().any(|n| {
            (n.contains("gpt_neox.layers.") && n.contains(".attention."))
                || n.contains("gpt_neox.embed_in.weight")
                || n.contains("gpt_neox.final_layer_norm.weight")
        }) {
            return Architecture::GPTNeoX;
        }

        Architecture::Unknown
    }

    /// Build complete name mapping from HF → target format
    fn build_name_map(
        architecture: Architecture,
        tensor_names: &[String],
    ) -> Result<HashMap<String, String>> {
        let mut map = HashMap::new();

        match architecture {
            Architecture::LLaMA => Self::build_llama_map(&mut map, tensor_names)?,
            Architecture::GPT2 => Self::build_gpt2_map(&mut map, tensor_names)?,
            Architecture::GPTNeoX => Self::build_neox_map(&mut map, tensor_names)?,
            Architecture::Unknown => {
                return Err(Error::UnknownArchitecture);
            }
        }

        Ok(map)
    }

    /// Build mapping for LLaMA-style models
    ///
    /// Supports both SafeTensors format (model.layers.N.self_attn.q_proj.weight)
    /// and GGUF format (blk.N.attn_q.weight)
    ///
    /// ⚠️ **A component this map does not recognize is a hard error naming
    /// EVERY unrecognized component, not the first.** The previous shape
    /// used `?` per tensor, so the FIRST unknown component aborted the
    /// whole loop and discarded whatever had already been mapped — and the
    /// only caller of this function (`SmartTensorNameMapper::from_tensor_names`)
    /// wraps that `Err` in `.ok()`, so the entire file silently lost name
    /// translation for **every** tensor, not just the unrecognized one.
    ///
    /// Measured against a real file (`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`,
    /// `general.architecture = qwen3`, 398 tensors): its `attn_q_norm.weight`
    /// / `attn_k_norm.weight` (QK-Norm) were not in
    /// [`Self::parse_llama_gguf_layer`]'s known list, and under the old
    /// shape this silently discarded translation for all 398 tensors, not
    /// just the two unrecognized components — the first one encountered
    /// poisoned the whole result. #96 added both as recognized components
    /// (structural renames, not guessed values -- see
    /// [`Self::parse_llama_gguf_layer`] and
    /// [`Self::parse_llama_safetensors_layer`]), so this particular pair no
    /// longer reaches the refusal below. The refusal itself stays: it is
    /// what the NEXT unanticipated component needs, and a partial map was
    /// considered and rejected for exactly the reason QK-Norm demonstrated
    /// here -- an unrecognized attention component has no safe default
    /// translation, so a caller building a model from a partial map could
    /// run it as ordinary attention while silently omitting a step the
    /// architecture actually needs: wrong numerics with no error, which is
    /// worse than a loud, named refusal.
    ///
    /// ⚠️ **This is a breaking behaviour change**, and deliberately so: a
    /// file with an unrecognized component used to silently fall back to
    /// unmapped raw-name lookups (see `LoadedModel::get_tensor`) with no
    /// error at all; it now refuses. Checked before making this change:
    /// zero corpus files and zero existing tests exercise that fallback
    /// for a tensor-bearing file (the reference corpus's 9 tensor-bearing
    /// files all declare `llama` with the original nine components; the
    /// other 19 declare non-llama architectures but carry zero tensors).
    fn build_llama_map(map: &mut HashMap<String, String>, tensor_names: &[String]) -> Result<()> {
        // Components, not raw tensor names: the same unrecognized component
        // recurs once per layer, and a caller needs to know WHICH
        // components are missing, not a wall of near-duplicate tensor
        // names differing only by layer number.
        let mut unrecognized: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();

        for name in tensor_names {
            let target_name = if name == "model.embed_tokens.weight" || name == "token_embd.weight"
            {
                "wte.weight".to_string()
            } else if name == "model.norm.weight" || name == "output_norm.weight" {
                "ln_f.weight".to_string()
            } else if name == "lm_head.weight" || name == "output.weight" {
                "lm_head.weight".to_string()
            } else if let Some(layer_section) = name.strip_prefix("model.layers.") {
                // SafeTensors format: model.layers.N.component
                match Self::parse_llama_safetensors_layer(layer_section) {
                    Ok(t) => t,
                    Err(_) => {
                        unrecognized.insert(component_suffix(layer_section).to_string());
                        continue;
                    }
                }
            } else if let Some(layer_section) = name.strip_prefix("blk.") {
                // GGUF format: blk.N.component
                match Self::parse_llama_gguf_layer(layer_section) {
                    Ok(t) => t,
                    Err(_) => {
                        unrecognized.insert(component_suffix(layer_section).to_string());
                        continue;
                    }
                }
            } else {
                continue; // Skip unknown tensors
            };

            map.insert(name.clone(), target_name);
        }

        if !unrecognized.is_empty() {
            return Err(Error::tensor_name_mapping(format!(
                "{} unrecognized layer component(s), refusing rather than silently \
                 dropping translation for the whole file or guessing a partial map: {}",
                unrecognized.len(),
                unrecognized.into_iter().collect::<Vec<_>>().join(", ")
            )));
        }

        Ok(())
    }

    /// Parse SafeTensors LLaMA layer tensor name
    fn parse_llama_safetensors_layer(layer_section: &str) -> Result<String> {
        if let Some(dot_pos) = layer_section.find('.') {
            let layer_num = &layer_section[..dot_pos];
            let rest = &layer_section[dot_pos + 1..];

            let target_component = match rest {
                s if s == "input_layernorm.weight" => format!("h.{}.ln_1.weight", layer_num),
                s if s == "post_attention_layernorm.weight" => {
                    format!("h.{}.ln_2.weight", layer_num)
                }
                s if s.starts_with("self_attn.q_proj.") => {
                    let suffix = &s["self_attn.q_proj.".len()..];
                    format!("h.{}.attn.q_proj.{}", layer_num, suffix)
                }
                s if s.starts_with("self_attn.k_proj.") => {
                    let suffix = &s["self_attn.k_proj.".len()..];
                    format!("h.{}.attn.k_proj.{}", layer_num, suffix)
                }
                s if s.starts_with("self_attn.v_proj.") => {
                    let suffix = &s["self_attn.v_proj.".len()..];
                    format!("h.{}.attn.v_proj.{}", layer_num, suffix)
                }
                s if s.starts_with("self_attn.o_proj.") => {
                    let suffix = &s["self_attn.o_proj.".len()..];
                    format!("h.{}.attn.out_proj.{}", layer_num, suffix)
                }
                // qwen3's QK-Norm (#96), HF naming -- same component the
                // GGUF arm above now recognizes under its `blk.N.` spelling.
                s if s.starts_with("self_attn.q_norm.") => {
                    let suffix = &s["self_attn.q_norm.".len()..];
                    format!("h.{}.attn.q_norm.{}", layer_num, suffix)
                }
                s if s.starts_with("self_attn.k_norm.") => {
                    let suffix = &s["self_attn.k_norm.".len()..];
                    format!("h.{}.attn.k_norm.{}", layer_num, suffix)
                }
                s if s.starts_with("mlp.gate_proj.") => {
                    let suffix = &s["mlp.gate_proj.".len()..];
                    format!("h.{}.mlp.gate_proj.{}", layer_num, suffix)
                }
                s if s.starts_with("mlp.up_proj.") => {
                    let suffix = &s["mlp.up_proj.".len()..];
                    format!("h.{}.mlp.up_proj.{}", layer_num, suffix)
                }
                s if s.starts_with("mlp.down_proj.") => {
                    let suffix = &s["mlp.down_proj.".len()..];
                    format!("h.{}.mlp.c_proj.{}", layer_num, suffix)
                }
                _ => {
                    return Err(Error::tensor_name_mapping(format!(
                        "model.layers.{}",
                        layer_section
                    )));
                }
            };

            Ok(target_component)
        } else {
            Err(Error::tensor_name_mapping(format!(
                "model.layers.{}",
                layer_section
            )))
        }
    }

    /// Parse GGUF LLaMA layer tensor name
    fn parse_llama_gguf_layer(layer_section: &str) -> Result<String> {
        if let Some(dot_pos) = layer_section.find('.') {
            let layer_num = &layer_section[..dot_pos];
            let rest = &layer_section[dot_pos + 1..];

            let target_component = match rest {
                "attn_norm.weight" => format!("h.{}.ln_1.weight", layer_num),
                "ffn_norm.weight" => format!("h.{}.ln_2.weight", layer_num),
                "attn_q.weight" => format!("h.{}.attn.q_proj.weight", layer_num),
                "attn_k.weight" => format!("h.{}.attn.k_proj.weight", layer_num),
                "attn_v.weight" => format!("h.{}.attn.v_proj.weight", layer_num),
                "attn_output.weight" => format!("h.{}.attn.out_proj.weight", layer_num),
                "ffn_gate.weight" => format!("h.{}.mlp.gate_proj.weight", layer_num),
                "ffn_up.weight" => format!("h.{}.mlp.up_proj.weight", layer_num),
                "ffn_down.weight" => format!("h.{}.mlp.c_proj.weight", layer_num),
                // qwen3's QK-Norm (#96): a real, documented GGUF component
                // this map previously had no entry for at all -- not a
                // guessed value, the same structural rename every other
                // component here gets.
                "attn_q_norm.weight" => format!("h.{}.attn.q_norm.weight", layer_num),
                "attn_k_norm.weight" => format!("h.{}.attn.k_norm.weight", layer_num),
                _ => return Err(Error::tensor_name_mapping(format!("blk.{}", layer_section))),
            };

            Ok(target_component)
        } else {
            Err(Error::tensor_name_mapping(format!("blk.{}", layer_section)))
        }
    }

    /// Build mapping for GPT-2 style models (mostly identity mapping)
    fn build_gpt2_map(map: &mut HashMap<String, String>, tensor_names: &[String]) -> Result<()> {
        for name in tensor_names {
            // GPT-2 format is already close to our target format
            let target_name = if let Some(stripped) = name.strip_prefix("transformer.") {
                stripped.to_string()
            } else {
                // Keep as-is for other tensors
                name.clone()
            };

            map.insert(name.clone(), target_name);
        }
        Ok(())
    }

    /// Build mapping for GPT-NeoX style models
    fn build_neox_map(_map: &mut HashMap<String, String>, _tensor_names: &[String]) -> Result<()> {
        // TODO: Implement GPT-NeoX mapping when needed
        Err(Error::other(
            "GPT-NeoX architecture mapping not yet implemented",
        ))
    }

    /// Map a HuggingFace tensor name to target format
    ///
    /// # Examples
    /// ```rust
    /// use mlmf::name_mapping::TensorNameMapper;
    ///
    /// let names = vec!["model.embed_tokens.weight".to_string()];
    /// let mapper = TensorNameMapper::from_tensor_names(&names)?;
    ///
    /// assert_eq!(
    ///     mapper.map_name("model.embed_tokens.weight"),
    ///     Some("wte.weight")
    /// );
    /// # Ok::<(), mlmf::Error>(())
    /// ```
    pub fn map_name(&self, hf_name: &str) -> Option<&str> {
        self.name_map.get(hf_name).map(|s| s.as_str())
    }

    /// Get the detected architecture
    pub fn architecture(&self) -> Architecture {
        self.architecture
    }

    /// Get all mapped names (for debugging)
    pub fn all_mappings(&self) -> &HashMap<String, String> {
        &self.name_map
    }

    /// Create a reverse mapping (target → HF names)
    pub fn reverse_map(&self) -> HashMap<String, String> {
        self.name_map
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect()
    }

    /// Get the number of mapped tensors
    pub fn len(&self) -> usize {
        self.name_map.len()
    }

    /// Check if the mapper is empty
    pub fn is_empty(&self) -> bool {
        self.name_map.is_empty()
    }

    /// Iterate over all name mappings
    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.name_map.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_safetensors_detection() {
        let names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.norm.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&names).unwrap();
        assert_eq!(mapper.architecture(), Architecture::LLaMA);
    }

    #[test]
    fn test_llama_gguf_detection() {
        let names = vec![
            "token_embd.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
            "output_norm.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&names).unwrap();
        assert_eq!(mapper.architecture(), Architecture::LLaMA);
    }

    #[test]
    fn test_llama_safetensors_mapping() {
        let names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.input_layernorm.weight".to_string(),
            "model.norm.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&names).unwrap();

        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            Some("wte.weight")
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("h.0.attn.q_proj.weight")
        );
        assert_eq!(
            mapper.map_name("model.layers.0.input_layernorm.weight"),
            Some("h.0.ln_1.weight")
        );
        assert_eq!(mapper.map_name("model.norm.weight"), Some("ln_f.weight"));
    }

    #[test]
    fn test_llama_gguf_mapping() {
        let names = vec![
            "token_embd.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_norm.weight".to_string(),
            "output_norm.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&names).unwrap();

        assert_eq!(mapper.map_name("token_embd.weight"), Some("wte.weight"));
        assert_eq!(
            mapper.map_name("blk.0.attn_q.weight"),
            Some("h.0.attn.q_proj.weight")
        );
        assert_eq!(
            mapper.map_name("blk.0.attn_norm.weight"),
            Some("h.0.ln_1.weight")
        );
        assert_eq!(mapper.map_name("output_norm.weight"), Some("ln_f.weight"));
    }

    /// ⚠️ THE DEFECT THIS FIX REPLACES: an unrecognized component used to
    /// silently discard translation for the WHOLE file (`.ok()` in
    /// `SmartTensorNameMapper::from_tensor_names`), not just itself. It now
    /// refuses instead, by name.
    ///
    /// ⚠️ #96 ADDED `attn_q_norm`/`attn_k_norm` AS RECOGNIZED COMPONENTS, so
    /// this test (previously built on those two) now uses a genuinely
    /// synthetic, never-to-be-recognized component name instead — the
    /// mechanism under test is "an unrecognized component is a named
    /// refusal", not qwen3 specifically. qwen3's QK-Norm now has its own
    /// positive-case test below
    /// (`qwen3_qk_norm_components_are_recognized_not_refused`).
    #[test]
    fn an_unrecognized_gguf_component_is_a_named_refusal() {
        let names = vec![
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_q_mystery.weight".to_string(),
            "blk.0.ffn_also_unrecognized.weight".to_string(),
        ];

        let err = TensorNameMapper::from_tensor_names(&names).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("attn_q_mystery.weight") && msg.contains("ffn_also_unrecognized.weight"),
            "both unrecognized components must be named, not just the first: {msg}"
        );
    }

    /// ⚠️ THE CONTROL for the test above: naming BOTH is not naming
    /// everything regardless of what was asked. A recognized-components-only
    /// file must still succeed exactly as before this fix. Includes qwen3's
    /// QK-Norm components (#96): they are recognized now, so a file using
    /// them belongs in this control, not in the refusal test above.
    #[test]
    fn a_file_using_only_recognized_components_still_succeeds() {
        let names = vec![
            "token_embd.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_q_norm.weight".to_string(),
            "blk.0.attn_k.weight".to_string(),
            "blk.0.attn_k_norm.weight".to_string(),
            "blk.0.attn_v.weight".to_string(),
            "blk.0.attn_output.weight".to_string(),
            "blk.0.attn_norm.weight".to_string(),
            "blk.0.ffn_gate.weight".to_string(),
            "blk.0.ffn_up.weight".to_string(),
            "blk.0.ffn_down.weight".to_string(),
            "blk.0.ffn_norm.weight".to_string(),
            "output_norm.weight".to_string(),
        ];
        assert!(TensorNameMapper::from_tensor_names(&names).is_ok());
    }

    /// ⚠️ ACROSS MULTIPLE LAYERS, THE SAME COMPONENT IS NAMED ONCE, NOT
    /// ONCE PER LAYER. A caller needs to know WHICH components are
    /// missing, not read a wall of near-duplicate tensor names differing
    /// only by layer number — a real file has one unrecognized-component
    /// message per layer count, which for a 30-layer model would otherwise
    /// bury the actual components inside dozens of repeats.
    #[test]
    fn the_same_unrecognized_component_across_layers_is_named_once() {
        // `.attn_q` as a SUBSTRING (not an exact component match) is enough
        // for architecture detection to classify this as LLaMA -- the same
        // reason the original #97 fixture used `attn_q_norm` before #96
        // made it a recognized component. The point under test is the
        // de-duplication, not architecture detection, so a synthetic name
        // that is detectable but never recognized keeps that separate.
        let names = vec![
            "blk.0.attn_q_mystery.weight".to_string(),
            "blk.1.attn_q_mystery.weight".to_string(),
            "blk.2.attn_q_mystery.weight".to_string(),
        ];
        let err = TensorNameMapper::from_tensor_names(&names).unwrap_err();
        let msg = err.to_string();
        assert_eq!(
            msg.matches("attn_q_mystery.weight").count(),
            1,
            "one component, named once, regardless of how many layers repeat it: {msg}"
        );
    }

    // ================================================================
    // #96 -- qwen3's QK-Norm (`attn_q_norm`/`attn_k_norm` in GGUF,
    // `self_attn.q_norm`/`self_attn.k_norm` in HF/SafeTensors) are real,
    // documented components this map previously had no entry for at all,
    // which poisoned name translation for the WHOLE file via the refusal
    // mechanism above. They are now recognized, structural renames --
    // not guessed values.
    // ================================================================

    /// ⚠️ THE POSITIVE CASE #96 ADDS. Before the fix, both GGUF names here
    /// hit `parse_llama_gguf_layer`'s catch-all arm and the whole file was
    /// refused (this is the exact fixture the old
    /// `an_unrecognized_gguf_component_is_a_named_refusal` test used to
    /// assert failure on). Now both translate, alongside the ordinary
    /// components in the same layer, which is the CONTROL that adding the
    /// two new arms did not disturb the nine pre-existing ones.
    #[test]
    fn qwen3_qk_norm_components_are_recognized_not_refused() {
        let names = vec![
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_q_norm.weight".to_string(),
            "blk.0.attn_k.weight".to_string(),
            "blk.0.attn_k_norm.weight".to_string(),
        ];
        let mapper = TensorNameMapper::from_tensor_names(&names)
            .expect("attn_q_norm/attn_k_norm are now recognized GGUF components");

        assert_eq!(
            mapper.map_name("blk.0.attn_q_norm.weight"),
            Some("h.0.attn.q_norm.weight")
        );
        assert_eq!(
            mapper.map_name("blk.0.attn_k_norm.weight"),
            Some("h.0.attn.k_norm.weight")
        );
        // CONTROL: the ordinary components in the same layer are unaffected.
        assert_eq!(
            mapper.map_name("blk.0.attn_q.weight"),
            Some("h.0.attn.q_proj.weight")
        );
    }

    /// The same component pair, HF/SafeTensors spelling.
    #[test]
    fn qwen3_qk_norm_components_are_recognized_in_safetensors_naming_too() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.self_attn.q_norm.weight".to_string(),
            "model.layers.0.self_attn.k_norm.weight".to_string(),
        ];
        let mapper = TensorNameMapper::from_tensor_names(&names)
            .expect("self_attn.q_norm/k_norm are now recognized SafeTensors components");

        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            Some("h.0.attn.q_norm.weight")
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.k_norm.weight"),
            Some("h.0.attn.k_norm.weight")
        );
    }

    #[test]
    fn test_gpt2_detection() {
        let names = vec![
            "transformer.wte.weight".to_string(),
            "transformer.h.0.attn.c_attn.weight".to_string(),
            "transformer.ln_f.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&names).unwrap();
        assert_eq!(mapper.architecture(), Architecture::GPT2);
    }

    #[test]
    fn test_gpt2_mapping() {
        let names = vec![
            "transformer.wte.weight".to_string(),
            "transformer.h.0.attn.c_attn.weight".to_string(),
        ];

        let mapper = TensorNameMapper::from_tensor_names(&names).unwrap();

        assert_eq!(
            mapper.map_name("transformer.wte.weight"),
            Some("wte.weight")
        );
        assert_eq!(
            mapper.map_name("transformer.h.0.attn.c_attn.weight"),
            Some("h.0.attn.c_attn.weight")
        );
    }

    #[test]
    fn test_uses_rope() {
        // Architectures that compute position on-the-fly (no stored table)
        assert!(Architecture::LLaMA.uses_rope(), "LLaMA should use RoPE");
        assert!(
            Architecture::GPTNeoX.uses_rope(),
            "GPT-NeoX should use RoPE"
        );
        // Architectures with learned absolute position embedding tables
        assert!(!Architecture::GPT2.uses_rope(), "GPT-2 should not use RoPE");
        assert!(
            !Architecture::Unknown.uses_rope(),
            "Unknown should default to no RoPE (conservative)"
        );
    }

    #[test]
    fn test_unknown_architecture() {
        let names = vec!["some.random.tensor.weight".to_string()];

        let result = TensorNameMapper::from_tensor_names(&names);
        assert!(matches!(result, Err(Error::UnknownArchitecture)));
    }

    #[test]
    fn test_architecture_names() {
        assert_eq!(Architecture::LLaMA.name(), "LLaMA");
        assert_eq!(Architecture::GPT2.name(), "GPT-2");
        assert_eq!(Architecture::GPTNeoX.name(), "GPT-NeoX");
        assert_eq!(Architecture::Unknown.name(), "Unknown");
    }
}
