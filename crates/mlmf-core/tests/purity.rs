//! C3: mlmf-core performs no I/O. Enforced at the source level, not by convention.

use std::fs;
use std::path::Path;

const FORBIDDEN: &[&str] = &[
    "std::fs",
    "std::net",
    "memmap2",
    "reqwest",
    "ureq",
    "tokio",
    "hf_hub",
];

fn collect_rs(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).expect("src directory must exist") {
        let path = entry.expect("readable entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn core_performs_no_io() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs(&src, &mut files);
    assert!(!files.is_empty(), "found no source files to check");

    let mut violations = Vec::new();
    for file in &files {
        let text = fs::read_to_string(file).expect("source file is readable");
        for needle in FORBIDDEN {
            if text.contains(needle) {
                violations.push(format!("{}: {needle}", file.display()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "mlmf-core must perform no I/O (C3); found:\n  {}",
        violations.join("\n  ")
    );
}
