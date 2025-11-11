//! Progress reporting utilities for model loading operations
//!
//! This module provides progress callback functionality for long-running model loading
//! operations, allowing users to track the status of loading, validation, and processing.

/// Progress callback function type
pub type ProgressFn = Box<dyn Fn(ProgressEvent) + Send + Sync>;

/// Events reported during model loading operations
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Starting to load configuration file
    LoadingConfig {
        /// Path to the config file
        path: String,
    },

    /// Scanning for model files in directory
    ScanningFiles {
        /// Number of files found so far
        count: usize,
    },

    /// Detecting model architecture from tensor names
    DetectingArchitecture,

    /// Loading tensor data from files
    LoadingTensors {
        /// Current file being processed
        current: usize,
        /// Total number of files
        total: usize,
        /// Name of current file
        file_name: Option<String>,
    },

    /// Mapping tensor names between formats
    MappingNames {
        /// Number of tensor names mapped
        count: usize,
    },

    /// Building the model from loaded tensors
    BuildingModel,

    /// Validating model configuration and tensors
    ValidatingModel,

    /// Loading a specific file
    LoadingFile {
        /// Path to the file being loaded
        file: std::path::PathBuf,
        /// Format of the file being loaded  
        format: String,
    },

    /// Loading tensors from file(s)
    LoadingTensorsFromFiles {
        /// Number of tensors to load
        count: usize,
        /// Format being loaded
        format: String,
    },

    /// Saving a specific file  
    SavingFile {
        /// Path to the file being saved
        file: std::path::PathBuf,
        /// Format being saved
        format: String,
    },

    /// Saving tensors to file(s)
    SavingTensors {
        /// Number of tensors to save
        count: usize,
        /// Format being saved
        format: String,
    },

    /// Loading operation completed successfully
    Complete {
        /// Number of tensors loaded/saved
        tensor_count: usize,
        /// Format that was processed
        format: String,
    },

    /// Custom status message
    Status {
        /// Message to display
        message: String,
    },

    /// Saving a checkpoint
    SavingCheckpoint,

    /// Checkpoint saved successfully
    CheckpointSaved,

    /// Parsing metadata from memory-mapped file
    ParsingMetadata,

    /// Prefetching tensors into cache
    PrefetchingTensors {
        /// Number of tensors to prefetch
        count: usize,
    },

    /// Loading a checkpoint
    LoadingCheckpoint {
        /// Path to checkpoint
        path: String,
    },

    /// Checkpoint loaded successfully
    CheckpointLoaded,
}

impl ProgressEvent {
    /// Get a human-readable description of this event
    pub fn description(&self) -> String {
        match self {
            ProgressEvent::LoadingConfig { path } => {
                format!("Loading config from {}", path)
            }
            ProgressEvent::ScanningFiles { count } => {
                if *count == 0 {
                    "Scanning for model files...".to_string()
                } else {
                    format!("Found {} model file(s)", count)
                }
            }
            ProgressEvent::DetectingArchitecture => "Detecting model architecture...".to_string(),
            ProgressEvent::LoadingTensors {
                current,
                total,
                file_name,
            } => {
                if let Some(name) = file_name {
                    format!("Loading tensors [{}/{}]: {}", current, total, name)
                } else {
                    format!("Loading tensors [{}/{}]", current, total)
                }
            }
            ProgressEvent::MappingNames { count } => {
                format!("Mapped {} tensor names", count)
            }
            ProgressEvent::BuildingModel => "Building model from tensors...".to_string(),
            ProgressEvent::ValidatingModel => "Validating model configuration...".to_string(),
            ProgressEvent::LoadingFile { file, format } => {
                format!("Loading {} file: {}", format, file.display())
            }
            ProgressEvent::LoadingTensorsFromFiles { count, format } => {
                format!("Loading {} tensors from {} format", count, format)
            }
            ProgressEvent::SavingFile { file, format } => {
                format!("Saving {} file: {}", format, file.display())
            }
            ProgressEvent::SavingTensors { count, format } => {
                format!("Saving {} tensors to {} format", count, format)
            }
            ProgressEvent::Complete {
                tensor_count,
                format,
            } => {
                format!(
                    "✓ {} format: {} tensors processed successfully",
                    format, tensor_count
                )
            }
            ProgressEvent::Status { message } => message.clone(),
            ProgressEvent::SavingCheckpoint => "Saving checkpoint...".to_string(),
            ProgressEvent::CheckpointSaved => "Checkpoint saved successfully".to_string(),
            ProgressEvent::LoadingCheckpoint { path } => {
                format!("Loading checkpoint from {}", path)
            }
            ProgressEvent::CheckpointLoaded => "Checkpoint loaded successfully".to_string(),
            ProgressEvent::ParsingMetadata => {
                "Parsing tensor metadata from memory-mapped file...".to_string()
            }
            ProgressEvent::PrefetchingTensors { count } => {
                format!("Prefetching {} tensors into cache...", count)
            }
        }
    }

    /// Check if this is a completion event
    pub fn is_complete(&self) -> bool {
        matches!(self, ProgressEvent::Complete { .. })
    }

    /// Check if this is an error-related event
    pub fn is_error(&self) -> bool {
        // Currently no error events, but could be extended
        false
    }
}

/// Default progress reporter that prints to stdout
///
/// # Examples
/// ```rust
/// use mlmf::progress::default_progress;
///
/// let progress_fn = default_progress();
/// // Use with LoadOptions
/// ```
pub fn default_progress() -> ProgressFn {
    Box::new(|event: ProgressEvent| {
        let description = event.description();
        if event.is_complete() {
            println!("{}", description);
        } else {
            println!("📦 {}", description);
        }
    })
}

/// Silent progress reporter (no-op)
///
/// Use this when you don't want any progress output.
///
/// # Examples
/// ```rust
/// use mlmf::progress::silent_progress;
///
/// let progress_fn = silent_progress();
/// // Use with LoadOptions for silent loading
/// ```
pub fn silent_progress() -> ProgressFn {
    Box::new(|_event: ProgressEvent| {
        // Do nothing
    })
}

/// Progress reporter that logs with timestamps
///
/// # Examples
/// ```rust
/// use mlmf::progress::timestamped_progress;
///
/// let progress_fn = timestamped_progress();
/// // Outputs: [2023-10-01 12:34:56] Loading config from ...
/// ```
pub fn timestamped_progress() -> ProgressFn {
    Box::new(|event: ProgressEvent| {
        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S");
        let description = event.description();

        if event.is_complete() {
            println!("[{}] {}", timestamp, description);
        } else {
            println!("[{}] 📦 {}", timestamp, description);
        }
    })
}

/// Progress reporter with custom prefix
///
/// # Arguments
/// * `prefix` - Custom prefix to add before each message
///
/// # Examples
/// ```rust
/// use mlmf::progress::prefixed_progress;
///
/// let progress_fn = prefixed_progress("MODEL_LOADER".to_string());
/// // Outputs: [MODEL_LOADER] Loading config from ...
/// ```
pub fn prefixed_progress(prefix: String) -> ProgressFn {
    Box::new(move |event: ProgressEvent| {
        let description = event.description();
        if event.is_complete() {
            println!("[{}] {}", prefix, description);
        } else {
            println!("[{}] 📦 {}", prefix, description);
        }
    })
}

#[cfg(feature = "progress")]
/// Progress reporter with a visual progress bar
///
/// Uses the `indicatif` crate to show a progress bar for tensor loading operations.
///
/// # Examples
/// ```rust
/// use mlmf::progress::progress_bar;
///
/// let progress_fn = progress_bar();
/// // Shows: Loading tensors [██████████████████████████████] 100%
/// ```
pub fn progress_bar() -> ProgressFn {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::sync::{Arc, Mutex};

    let pb = Arc::new(Mutex::new(None::<ProgressBar>));

    Box::new(move |event: ProgressEvent| {
        let mut pb_guard = pb.lock().unwrap();

        match event {
            ProgressEvent::LoadingTensors { current, total, .. } => {
                if pb_guard.is_none() {
                    let new_pb = ProgressBar::new(total as u64);
                    new_pb.set_style(
                        ProgressStyle::default_bar()
                            .template("📦 Loading tensors [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                            .unwrap()
                            .progress_chars("█▉▊▋▌▍▎▏ "),
                    );
                    *pb_guard = Some(new_pb);
                }

                if let Some(ref pb) = *pb_guard {
                    pb.set_position(current as u64);
                    if let Some(file_name) = event.description().split(": ").nth(1) {
                        pb.set_message(file_name.to_string());
                    }
                }
            }
            ProgressEvent::Complete { .. } => {
                if let Some(ref pb) = *pb_guard {
                    pb.finish_with_message("✓ Complete");
                }
                *pb_guard = None;
                println!("{}", event.description());
            }
            _ => {
                // For non-tensor events, print normally
                println!("📦 {}", event.description());
            }
        }
    })
}

#[cfg(not(feature = "progress"))]
/// Progress reporter with a visual progress bar (fallback when indicatif not available)
pub fn progress_bar() -> ProgressFn {
    // Fallback to default progress when indicatif feature is not enabled
    default_progress()
}

/// Create a custom progress reporter from a closure
///
/// # Arguments
/// * `f` - Closure that handles progress events
///
/// # Examples
/// ```rust
/// use mlmf::progress::{custom_progress, ProgressEvent};
///
/// let progress_fn = custom_progress(|event: ProgressEvent| {
///     match event {
///         ProgressEvent::Complete { tensor_count, format } => {
///             println!("Loaded {} tensors in {} format!", tensor_count, format);
///         }
///         _ => {
///             // Handle other events
///         }
///     }
/// });
/// ```
pub fn custom_progress<F>(f: F) -> ProgressFn
where
    F: Fn(ProgressEvent) + Send + Sync + 'static,
{
    Box::new(f)
}

/// Utility to time an operation and report completion
pub struct ProgressTimer {
    start_time: std::time::Instant,
    progress_fn: Option<ProgressFn>,
}

impl ProgressTimer {
    /// Create a new progress timer with optional progress reporting
    pub fn new(progress_fn: Option<ProgressFn>) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            progress_fn,
        }
    }

    /// Report a progress event
    pub fn report(&self, event: ProgressEvent) {
        if let Some(ref progress_fn) = self.progress_fn {
            progress_fn(event);
        }
    }

    /// Report completion and return elapsed time
    pub fn complete(&self) -> f64 {
        let _elapsed_secs = self.start_time.elapsed().as_secs_f64();
        self.report(ProgressEvent::Complete {
            tensor_count: 0, // We don't track tensor count in ProgressTimer
            format: "Generic".to_string(),
        });
        _elapsed_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_progress_event_descriptions() {
        let event = ProgressEvent::LoadingConfig {
            path: "/path/to/config.json".to_string(),
        };
        assert_eq!(
            event.description(),
            "Loading config from /path/to/config.json"
        );

        let event = ProgressEvent::ScanningFiles { count: 3 };
        assert_eq!(event.description(), "Found 3 model file(s)");

        let event = ProgressEvent::Complete {
            tensor_count: 1234,
            format: "SafeTensors".to_string(),
        };
        assert_eq!(
            event.description(),
            "✓ SafeTensors format: 1234 tensors processed successfully"
        );
        assert!(event.is_complete());
    }

    #[test]
    fn test_custom_progress() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let progress_fn = custom_progress(move |event: ProgressEvent| {
            events_clone.lock().unwrap().push(event);
        });

        progress_fn(ProgressEvent::DetectingArchitecture);
        progress_fn(ProgressEvent::Complete {
            tensor_count: 100,
            format: "SafeTensors".to_string(),
        });

        let captured_events = events.lock().unwrap();
        assert_eq!(captured_events.len(), 2);
        assert!(matches!(
            captured_events[0],
            ProgressEvent::DetectingArchitecture
        ));
        assert!(matches!(captured_events[1], ProgressEvent::Complete { .. }));
    }

    #[test]
    fn test_progress_timer() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let progress_fn = custom_progress(move |event: ProgressEvent| {
            events_clone.lock().unwrap().push(event);
        });

        let timer = ProgressTimer::new(Some(progress_fn));
        timer.report(ProgressEvent::DetectingArchitecture);
        let elapsed = timer.complete();

        assert!(elapsed >= 0.0);

        let captured_events = events.lock().unwrap();
        assert_eq!(captured_events.len(), 2);
        assert!(matches!(captured_events[1], ProgressEvent::Complete { .. }));
    }
}
