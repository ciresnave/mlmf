//! Model Card Generation Example
//!
//! This example demonstrates how to generate comprehensive model cards for documentation,
//! model management, and responsible AI practices.

use mlmf::{
    config::ModelConfig,
    model_card::{EvaluationInfo, ModelCardGenerator, TrainingInfo},
    name_mapping::Architecture,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 MLMF Model Card Generation Example");
    println!("=====================================\n");

    // Create example model configurations
    let models = create_example_models();

    for (model_name, config) in models {
        println!("📋 Generating model card for: {}", model_name);
        println!("   Architecture: {}", config.architecture);
        println!(
            "   Parameters: {:.1}M",
            estimate_parameters(&config) as f64 / 1_000_000.0
        );

        // Create model card generator
        let generator = ModelCardGenerator::new()
            .with_technical_details(true)
            .with_memory_estimation(true)
            .with_metadata("generator_version", "1.0.0")
            .with_metadata("created_for", "MLMF Example");

        // Generate model card
        let dummy_path = Path::new("./models").join(&model_name);
        let model_card =
            generator.generate_from_config(&config, &dummy_path, model_name.clone())?;

        // Add some example training and evaluation data
        let mut enhanced_card = model_card;
        enhanced_card.training_info = Some(create_example_training_info(&config.architecture));
        enhanced_card.evaluation_info = Some(create_example_evaluation_info(&config.architecture));

        // Save as JSON
        let json_path = format!(
            "{}_model_card.json",
            model_name.replace(" ", "_").to_lowercase()
        );
        generator.generate_and_save_json(
            &config,
            &dummy_path,
            Path::new(&json_path),
            model_name.clone(),
        )?;
        println!("   ✅ JSON saved to: {}", json_path);

        // Save as README.md
        let readme_path = format!("{}_README.md", model_name.replace(" ", "_").to_lowercase());
        generator.generate_and_save_readme(
            &config,
            &dummy_path,
            Path::new(&readme_path),
            model_name.clone(),
        )?;
        println!("   ✅ README saved to: {}", readme_path);

        // Display key information
        println!("   📊 Technical Specs:");
        println!(
            "      - Vocab Size: {}",
            enhanced_card.technical_specs.vocab_size
        );
        println!(
            "      - Hidden Size: {}",
            enhanced_card.technical_specs.hidden_size
        );
        println!(
            "      - Layers: {}",
            enhanced_card.technical_specs.num_layers
        );
        println!(
            "      - Attention Heads: {}",
            enhanced_card.technical_specs.num_attention_heads
        );
        println!(
            "      - Memory (Inference): {:.1} GB",
            enhanced_card
                .technical_specs
                .memory_requirements
                .inference_mb as f64
                / 1024.0
        );

        println!("   📋 Usage Guidelines:");
        for use_case in &enhanced_card.usage_info.intended_uses {
            println!("      + {}", use_case);
        }

        println!("   ⚠️  Key Limitations:");
        for (i, limitation) in enhanced_card
            .usage_info
            .limitations
            .iter()
            .take(3)
            .enumerate()
        {
            println!("      {}. {}", i + 1, limitation);
        }

        if let Some(eval) = &enhanced_card.evaluation_info {
            println!("   🏆 Evaluation Results:");
            for (benchmark, score) in &eval.benchmarks {
                println!("      - {}: {:.2}", benchmark, score);
            }
        }

        println!(); // Add spacing
    }

    // Demonstrate format-specific model card generation
    demonstrate_format_specific_cards()?;

    // Show model card customization
    demonstrate_card_customization()?;

    println!("✅ Model card generation complete!");
    println!("\n💡 Next Steps:");
    println!("   1. Review generated JSON and README files");
    println!("   2. Customize training_info and evaluation_info for real models");
    println!("   3. Add model-specific metadata using .with_metadata()");
    println!("   4. Integrate model cards into your model management workflow");
    println!("   5. Use cards for responsible AI documentation");

    Ok(())
}

fn create_example_models() -> Vec<(String, ModelConfig)> {
    vec![
        (
            "LLaMA-7B-Chat".to_string(),
            ModelConfig {
                vocab_size: 32000,
                hidden_size: 4096,
                num_attention_heads: 32,
                num_hidden_layers: 32,
                intermediate_size: 11008,
                max_position_embeddings: 4096,
                dropout: 0.0,
                layer_norm_eps: 1e-6,
                attention_dropout: 0.0,
                activation_function: "silu".to_string(),
                rope_theta: 10000.0,
                tie_word_embeddings: false,
                architecture: Architecture::LLaMA,
                raw_config: serde_json::json!({}),
            },
        ),
        (
            "GPT2-Medium".to_string(),
            ModelConfig {
                vocab_size: 50257,
                hidden_size: 1024,
                num_attention_heads: 16,
                num_hidden_layers: 24,
                intermediate_size: 4096,
                max_position_embeddings: 1024,
                dropout: 0.1,
                layer_norm_eps: 1e-5,
                attention_dropout: 0.1,
                activation_function: "gelu".to_string(),
                rope_theta: 10000.0,
                tie_word_embeddings: true,
                architecture: Architecture::GPT2,
                raw_config: serde_json::json!({}),
            },
        ),
        (
            "GPTNeoX-Small".to_string(),
            ModelConfig {
                vocab_size: 50432,
                hidden_size: 768,
                num_attention_heads: 12,
                num_hidden_layers: 12,
                intermediate_size: 3072,
                max_position_embeddings: 2048,
                dropout: 0.1,
                layer_norm_eps: 1e-5,
                attention_dropout: 0.1,
                activation_function: "gelu".to_string(),
                rope_theta: 10000.0,
                tie_word_embeddings: false,
                architecture: Architecture::GPTNeoX,
                raw_config: serde_json::json!({}),
            },
        ),
    ]
}

fn create_example_training_info(architecture: &Architecture) -> TrainingInfo {
    match architecture {
        Architecture::LLaMA => TrainingInfo {
            datasets: vec![
                "Common Crawl".to_string(),
                "Wikipedia".to_string(),
                "BookCorpus".to_string(),
                "OpenWebText".to_string(),
            ],
            procedure: Some("Causal language modeling with next-token prediction".to_string()),
            framework: Some("PyTorch".to_string()),
            hardware: Some("A100 GPUs".to_string()),
            duration: Some("Several weeks".to_string()),
            tokens_or_steps: Some("1.4T tokens".to_string()),
            learning_rate: Some("Peak 3e-4 with cosine decay".to_string()),
            batch_size: Some("4M tokens per batch".to_string()),
        },
        Architecture::GPT2 => TrainingInfo {
            datasets: vec![
                "WebText".to_string(),
                "BookCorpus".to_string(),
                "Common Crawl (filtered)".to_string(),
            ],
            procedure: Some("Autoregressive language modeling".to_string()),
            framework: Some("PyTorch".to_string()),
            hardware: Some("V100 GPUs".to_string()),
            duration: Some("2-3 weeks".to_string()),
            tokens_or_steps: Some("40B tokens".to_string()),
            learning_rate: Some("2.5e-4 with linear warmup".to_string()),
            batch_size: Some("512 sequences".to_string()),
        },
        Architecture::GPTNeoX => TrainingInfo {
            datasets: vec!["The Pile".to_string(), "Common Crawl".to_string()],
            procedure: Some("Autoregressive language modeling with parallel training".to_string()),
            framework: Some("DeepSpeed".to_string()),
            hardware: Some("A100 GPUs".to_string()),
            duration: Some("Several weeks".to_string()),
            tokens_or_steps: Some("400B tokens".to_string()),
            learning_rate: Some("1.6e-4 with warmup".to_string()),
            batch_size: Some("2M tokens per batch".to_string()),
        },
        _ => TrainingInfo {
            datasets: vec!["Unknown dataset".to_string()],
            procedure: None,
            framework: None,
            hardware: None,
            duration: None,
            tokens_or_steps: None,
            learning_rate: None,
            batch_size: None,
        },
    }
}

fn create_example_evaluation_info(architecture: &Architecture) -> EvaluationInfo {
    let mut benchmarks = HashMap::new();

    match architecture {
        Architecture::LLaMA => {
            benchmarks.insert("HellaSwag".to_string(), 76.1);
            benchmarks.insert("MMLU".to_string(), 35.1);
            benchmarks.insert("TruthfulQA".to_string(), 33.1);
            benchmarks.insert("Winogrande".to_string(), 70.1);
            benchmarks.insert("GSM8K".to_string(), 11.0);
        }
        Architecture::GPT2 => {
            benchmarks.insert("LAMBADA".to_string(), 63.2);
            benchmarks.insert("HellaSwag".to_string(), 52.9);
            benchmarks.insert("Winograd".to_string(), 70.2);
            benchmarks.insert("PIQA".to_string(), 70.2);
        }
        Architecture::GPTNeoX => {
            benchmarks.insert("LAMBADA".to_string(), 67.1);
            benchmarks.insert("HellaSwag".to_string(), 54.2);
            benchmarks.insert("PIQA".to_string(), 71.0);
            benchmarks.insert("Winogrande".to_string(), 65.7);
        }
        _ => {
            benchmarks.insert("Unknown".to_string(), 0.0);
        }
    }

    EvaluationInfo {
        benchmarks,
        methodology: Some("Standard evaluation protocol with few-shot prompting".to_string()),
        test_datasets: vec!["Held-out test sets".to_string()],
        limitations: Some("Results may vary with different prompting strategies".to_string()),
    }
}

fn demonstrate_format_specific_cards() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Format-Specific Model Card Generation");
    println!("=========================================\n");

    // Simulate different model formats
    let formats = vec![
        ("SafeTensors Model", "model.safetensors"),
        ("GGUF Model", "model.gguf"),
        ("PyTorch Model", "model.pt"),
        ("ONNX Model", "model.onnx"),
    ];

    let config = ModelConfig {
        vocab_size: 50257,
        hidden_size: 768,
        num_attention_heads: 12,
        num_hidden_layers: 12,
        intermediate_size: 3072,
        max_position_embeddings: 1024,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
        attention_dropout: 0.1,
        activation_function: "gelu".to_string(),
        rope_theta: 10000.0,
        tie_word_embeddings: true,
        architecture: Architecture::GPT2,
        raw_config: serde_json::json!({}),
    };

    for (format_name, filename) in formats {
        println!("📄 Generating card for: {}", format_name);

        let generator = ModelCardGenerator::new().with_metadata(
            "model_format",
            filename.split('.').last().unwrap_or("unknown"),
        );

        let model_path = Path::new(filename);
        let card = generator.generate_from_config(&config, &model_path, format_name.to_string())?;

        println!("   Format: {}", card.technical_specs.model_format);
        println!("   Tags: {:?}", card.card_metadata.tags);
        println!();
    }

    Ok(())
}

fn demonstrate_card_customization() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Model Card Customization");
    println!("============================\n");

    let config = ModelConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        num_attention_heads: 16,
        num_hidden_layers: 16,
        intermediate_size: 5504,
        max_position_embeddings: 2048,
        dropout: 0.0,
        layer_norm_eps: 1e-6,
        attention_dropout: 0.0,
        activation_function: "silu".to_string(),
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        architecture: Architecture::LLaMA,
        raw_config: serde_json::json!({}),
    };

    // Minimal card (no technical details, no memory estimation)
    println!("📋 Minimal Model Card:");
    let minimal_generator = ModelCardGenerator::new()
        .with_technical_details(false)
        .with_memory_estimation(false);

    let minimal_card = minimal_generator.generate_from_config(
        &config,
        Path::new("minimal_model"),
        "Minimal Model".to_string(),
    )?;

    println!(
        "   Technical details included: {}",
        minimal_generator.include_technical_details
    );
    println!(
        "   Memory estimation: {}",
        minimal_generator.estimate_memory
    );
    println!(
        "   Memory requirements: {:.1} MB",
        minimal_card
            .technical_specs
            .memory_requirements
            .parameters_mb
    );

    // Comprehensive card with custom metadata
    println!("\n📋 Comprehensive Model Card:");
    let comprehensive_generator = ModelCardGenerator::new()
        .with_technical_details(true)
        .with_memory_estimation(true)
        .with_metadata("organization", "Example Corp")
        .with_metadata("research_paper", "https://arxiv.org/abs/2023.example")
        .with_metadata("fine_tuned_from", "base-model-v1")
        .with_metadata("use_case", "conversational_ai");

    let comprehensive_card = comprehensive_generator.generate_from_config(
        &config,
        Path::new("comprehensive_model"),
        "Comprehensive Model".to_string(),
    )?;

    println!(
        "   Technical details included: {}",
        comprehensive_generator.include_technical_details
    );
    println!(
        "   Memory estimation: {}",
        comprehensive_generator.estimate_memory
    );
    println!(
        "   Additional metadata: {} items",
        comprehensive_generator.additional_metadata.len()
    );
    println!(
        "   Memory requirements: {:.1} GB",
        comprehensive_card
            .technical_specs
            .memory_requirements
            .inference_mb as f64
            / 1024.0
    );

    Ok(())
}

fn estimate_parameters(config: &ModelConfig) -> u64 {
    let embedding_params = config.vocab_size * config.hidden_size;
    let attention_params = config.num_hidden_layers * config.hidden_size * config.hidden_size * 4;
    let ffn_params = config.num_hidden_layers * config.hidden_size * config.intermediate_size * 2;
    let norm_params = config.num_hidden_layers * config.hidden_size * 2;

    (embedding_params + attention_params + ffn_params + norm_params) as u64
}
