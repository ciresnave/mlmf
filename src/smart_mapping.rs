//! Smart tensor name mapping with ML-powered suggestions
//!
//! This module provides intelligent tensor name mapping using pluggable inference systems.
//! When static pattern matching fails, it can leverage ML models to make educated guesses
//! about tensor name correspondences.

use crate::{
    error::{Error, Result},
    name_mapping::{Architecture, TensorNameMapper},
};
use std::collections::HashMap;

/// A plugin trait for intelligent tensor name mapping
///
/// This allows any inference system (Lightbulb, Cognition, etc.) to be plugged in
/// to help resolve ambiguous tensor name mappings.
pub trait NameMappingOracle: Send + Sync {
    /// Given model file names and expected program names, suggest mappings
    ///
    /// # Arguments
    /// * `model_names` - Tensor names found in the model file
    /// * `program_names` - Tensor names expected by the program
    /// * `context` - Additional context (architecture, format, etc.)
    ///
    /// # Returns
    /// A mapping from program_name -> model_name for suggested correspondences
    fn suggest_mappings(
        &self,
        model_names: &[String],
        program_names: &[String],
        context: &MappingContext,
    ) -> Result<HashMap<String, String>>;

    /// Get a confidence score for a specific mapping suggestion
    ///
    /// Returns a value between 0.0 and 1.0, where 1.0 is highest confidence
    fn confidence_score(
        &self,
        model_name: &str,
        program_name: &str,
        context: &MappingContext,
    ) -> f32;

    /// Name of this oracle (for logging and debugging)
    fn name(&self) -> &str;
}

/// Context information for smart tensor mapping
#[derive(Debug, Clone)]
pub struct MappingContext {
    /// Detected or known architecture
    pub architecture: Option<Architecture>,

    /// Model format (SafeTensors, GGUF, etc.)
    pub format: Option<String>,

    /// Model size estimate (helpful for architecture inference)
    pub estimated_params: Option<u64>,

    /// Additional metadata from model file
    pub metadata: HashMap<String, String>,
}

impl Default for MappingContext {
    fn default() -> Self {
        Self {
            architecture: None,
            format: None,
            estimated_params: None,
            metadata: HashMap::new(),
        }
    }
}

impl MappingContext {
    /// Create new mapping context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set architecture
    pub fn with_architecture(mut self, arch: Architecture) -> Self {
        self.architecture = Some(arch);
        self
    }

    /// Set format
    pub fn with_format<S: Into<String>>(mut self, format: S) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Enhanced tensor name mapper with smart fallbacks
pub struct SmartTensorNameMapper {
    /// Static pattern-based mapper (first choice)
    static_mapper: Option<TensorNameMapper>,

    /// ML-powered oracle (fallback when patterns fail)
    oracle: Option<Box<dyn NameMappingOracle>>,

    /// Context for smart mapping
    context: MappingContext,

    /// Cached mappings from oracle
    cached_mappings: HashMap<String, String>,
}

impl SmartTensorNameMapper {
    /// Create a new smart mapper
    pub fn new() -> Self {
        Self {
            static_mapper: None,
            oracle: None,
            context: MappingContext::default(),
            cached_mappings: HashMap::new(),
        }
    }

    /// Try to create from tensor names using static patterns
    pub fn from_tensor_names(tensor_names: &[String]) -> Result<Self> {
        let static_mapper = TensorNameMapper::from_tensor_names(tensor_names).ok();

        let mut context = MappingContext::new();
        if let Some(mapper) = &static_mapper {
            context.architecture = Some(mapper.architecture().clone());
        }

        Ok(Self {
            static_mapper,
            oracle: None,
            context,
            cached_mappings: HashMap::new(),
        })
    }

    /// Set the ML oracle for smart mapping
    pub fn with_oracle(mut self, oracle: Box<dyn NameMappingOracle>) -> Self {
        self.oracle = Some(oracle);
        self
    }

    /// Set context information
    pub fn with_context(mut self, context: MappingContext) -> Self {
        self.context = context;
        self
    }

    /// Map a tensor name using static patterns or smart fallback
    pub fn map_name(&mut self, model_name: &str) -> Option<String> {
        // First, try static pattern matching
        if let Some(mapper) = &self.static_mapper {
            if let Some(mapped) = mapper.map_name(model_name) {
                return Some(mapped.to_string());
            }
        }

        // Check cache
        if let Some(cached) = self.cached_mappings.get(model_name) {
            return Some(cached.clone());
        }

        // If we have an oracle, this would normally ask it for suggestions
        // For now, return None (could be enhanced to query oracle in batch)
        None
    }

    /// Suggest mappings for a batch of program names using the oracle
    pub fn suggest_batch_mappings(
        &mut self,
        model_names: &[String],
        program_names: &[String],
    ) -> Result<HashMap<String, String>> {
        if let Some(oracle) = &self.oracle {
            let suggestions = oracle.suggest_mappings(model_names, program_names, &self.context)?;

            // Cache the suggestions
            self.cached_mappings.extend(suggestions.clone());

            Ok(suggestions)
        } else {
            // Without an oracle, return empty mapping
            Ok(HashMap::new())
        }
    }

    /// Get the detected architecture
    pub fn architecture(&self) -> Option<&Architecture> {
        self.context.architecture.as_ref()
    }

    /// Get reverse mapping (mapped name -> original name)
    pub fn reverse_map(&self) -> HashMap<String, String> {
        let mut reverse = HashMap::new();

        // Add static mappings
        if let Some(mapper) = &self.static_mapper {
            reverse.extend(mapper.reverse_map());
        }

        // Add cached oracle mappings (reversed)
        for (original, mapped) in &self.cached_mappings {
            reverse.insert(mapped.clone(), original.clone());
        }

        reverse
    }

    /// Get all current mappings (original name -> mapped name)
    pub fn all_mappings(&self) -> HashMap<String, String> {
        let mut all = HashMap::new();

        // Add static mappings
        if let Some(mapper) = &self.static_mapper {
            for (k, v) in mapper.all_mappings() {
                all.insert(k.clone(), v.clone());
            }
        }

        // Add cached oracle mappings
        all.extend(self.cached_mappings.clone());

        all
    }

    /// Get the number of available mappings
    pub fn len(&self) -> usize {
        self.all_mappings().len()
    }

    /// Check if the mapper has no mappings
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all mappings (original name -> mapped name)
    pub fn iter(&self) -> impl Iterator<Item = (String, String)> {
        self.all_mappings().into_iter()
    }
}

/// A simple chat-based oracle that formats the mapping problem as a prompt
///
/// This can be used with any chat interface (OpenAI, local models, etc.)
pub struct ChatBasedOracle<F>
where
    F: Fn(&str) -> Result<String> + Send + Sync,
{
    /// Chat function: takes prompt, returns response
    chat_fn: F,
    /// Name for this oracle
    name: String,
}

impl<F> ChatBasedOracle<F>
where
    F: Fn(&str) -> Result<String> + Send + Sync,
{
    /// Create new chat-based oracle
    pub fn new<S: Into<String>>(name: S, chat_fn: F) -> Self {
        Self {
            name: name.into(),
            chat_fn,
        }
    }

    /// Format the mapping problem as a chat prompt
    fn create_prompt(
        &self,
        model_names: &[String],
        program_names: &[String],
        context: &MappingContext,
    ) -> String {
        let mut prompt = String::new();
        prompt
            .push_str("I need help mapping tensor names between a model file and my program.\n\n");

        if let Some(arch) = &context.architecture {
            prompt.push_str(&format!("Architecture: {}\n", arch.name()));
        }
        if let Some(format) = &context.format {
            prompt.push_str(&format!("Format: {}\n", format));
        }

        prompt.push_str("\nModel file contains these tensor names:\n");
        for name in model_names {
            prompt.push_str(&format!("- {}\n", name));
        }

        prompt.push_str("\nMy program expects these tensor names:\n");
        for name in program_names {
            prompt.push_str(&format!("- {}\n", name));
        }

        prompt.push_str(
            "\nPlease suggest which model tensor name corresponds to each program name.\n\
            Return your answer as JSON in this format:\n\
            {\n  \"program_name1\": \"model_name1\",\n  \"program_name2\": \"model_name2\"\n}\n\n\
            Only include mappings you're confident about.",
        );

        prompt
    }

    /// Parse JSON response into mappings
    fn parse_response(&self, response: &str) -> Result<HashMap<String, String>> {
        // Find JSON in the response (it might have extra text)
        let start = response
            .find('{')
            .ok_or_else(|| Error::model_loading("No JSON found in oracle response"))?;
        let end = response
            .rfind('}')
            .ok_or_else(|| Error::model_loading("Incomplete JSON in oracle response"))?
            + 1;

        let json_str = &response[start..end];

        serde_json::from_str(json_str).map_err(|e| {
            Error::model_loading(&format!("Failed to parse oracle response as JSON: {}", e))
        })
    }
}

impl<F> NameMappingOracle for ChatBasedOracle<F>
where
    F: Fn(&str) -> Result<String> + Send + Sync,
{
    fn suggest_mappings(
        &self,
        model_names: &[String],
        program_names: &[String],
        context: &MappingContext,
    ) -> Result<HashMap<String, String>> {
        let prompt = self.create_prompt(model_names, program_names, context);
        let response = (self.chat_fn)(&prompt)?;
        self.parse_response(&response)
    }

    fn confidence_score(
        &self,
        _model_name: &str,
        _program_name: &str,
        _context: &MappingContext,
    ) -> f32 {
        // Chat-based oracles don't provide granular confidence scores
        // Could be enhanced to ask for confidence in a follow-up prompt
        0.8
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_mapper_creation() {
        let mapper = SmartTensorNameMapper::new();
        assert!(mapper.static_mapper.is_none());
        assert!(mapper.oracle.is_none());
    }

    #[test]
    fn test_mapping_context() {
        let context = MappingContext::new()
            .with_architecture(Architecture::LLaMA)
            .with_format("SafeTensors")
            .with_metadata("size", "7B");

        assert_eq!(context.architecture, Some(Architecture::LLaMA));
        assert_eq!(context.format, Some("SafeTensors".to_string()));
        assert_eq!(context.metadata.get("size"), Some(&"7B".to_string()));
    }

    #[test]
    fn test_chat_oracle_prompt() {
        let oracle = ChatBasedOracle::new("test", |_| Ok("{}".to_string()));

        let model_names = vec!["blk.0.attn_q.weight".to_string()];
        let program_names = vec!["layer.0.attention.query.weight".to_string()];
        let context = MappingContext::new().with_architecture(Architecture::LLaMA);

        let prompt = oracle.create_prompt(&model_names, &program_names, &context);

        assert!(prompt.contains("Architecture: LLaMA"));
        assert!(prompt.contains("blk.0.attn_q.weight"));
        assert!(prompt.contains("layer.0.attention.query.weight"));
        assert!(prompt.contains("JSON"));
    }
}
