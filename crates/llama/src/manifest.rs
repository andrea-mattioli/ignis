//! A llama.cpp model directory (ADR 0052): the GGUF weights, the vision
//! encoder's mmproj, and the six frontend resources the `.ninfer` container
//! would carry (`tokenizer.json`, `chat_template.jinja`, ... from the
//! model's Hugging Face repository), named by `ignis-llama.json`:
//!
//! ```json
//! { "model_id": "qwen3-vl-4b", "gguf": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
//!   "mmproj": "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
//!   "files": { "Qwen3VL-4B-Instruct-Q4_K_M.gguf": 2497281120 } }
//! ```
//!
//! `files` is the sidecar's role in miniature: every file it names must be
//! there at that size, or the load is refused rather than served from a
//! partial copy.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The manifest's file name inside the model directory.
pub const MANIFEST: &str = "ignis-llama.json";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Manifest {
    pub model_id: String,
    pub gguf: String,
    #[serde(default)]
    pub mmproj: Option<String>,
    /// Expected byte sizes, by file name.
    #[serde(default)]
    pub files: std::collections::BTreeMap<String, u64>,
}

/// A model directory whose manifest was read and whose files were checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDir {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl ModelDir {
    /// Whether `path` is a llama.cpp model directory at all.
    pub fn is_one(path: &Path) -> bool {
        path.is_dir() && path.join(MANIFEST).is_file()
    }

    /// Read and check the directory.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join(MANIFEST)).map_err(|e| format!("{MANIFEST}: {e}"))?;
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| format!("{MANIFEST}: {e}"))?;
        let mut required = vec![manifest.gguf.clone()];
        required.extend(manifest.mmproj.clone());
        for name in &required {
            if !dir.join(name).is_file() {
                return Err(format!("{name} is named by {MANIFEST} but is not in {}", dir.display()));
            }
        }
        for (name, &bytes) in &manifest.files {
            let got = std::fs::metadata(dir.join(name)).map_err(|e| format!("{name}: {e}"))?.len();
            if got != bytes {
                return Err(format!("{name} is {got} bytes, {MANIFEST} says {bytes}: a partial or different copy"));
            }
        }
        Ok(Self { dir: dir.to_path_buf(), manifest })
    }

    pub fn gguf(&self) -> PathBuf {
        self.dir.join(&self.manifest.gguf)
    }

    pub fn mmproj(&self) -> Option<PathBuf> {
        self.manifest.mmproj.as_ref().map(|m| self.dir.join(m))
    }

    /// A frontend resource by its base name (`tokenizer.json`, ...).
    pub fn resource(&self, name: &str) -> Result<Vec<u8>, String> {
        std::fs::read(self.dir.join(name)).map_err(|e| format!("{name} in {}: {e}", self.dir.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &[u8])], manifest: &str) -> tempdir::Dir {
        let d = tempdir::Dir::new();
        for (name, bytes) in files {
            std::fs::write(d.path().join(name), bytes).unwrap();
        }
        std::fs::write(d.path().join(MANIFEST), manifest).unwrap();
        d
    }

    mod tempdir {
        pub struct Dir(std::path::PathBuf);
        impl Dir {
            pub fn new() -> Self {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let p = std::env::temp_dir().join(format!("ignis-llama-{}-{n}", std::process::id()));
                std::fs::create_dir_all(&p).unwrap();
                Self(p)
            }
            pub fn path(&self) -> &std::path::Path {
                &self.0
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn a_complete_directory_opens() {
        let d = dir_with(&[("m.gguf", b"1234"), ("p.gguf", b"12")],
                         r#"{"model_id":"qwen3-vl-4b","gguf":"m.gguf","mmproj":"p.gguf","files":{"m.gguf":4}}"#);
        assert!(ModelDir::is_one(d.path()));
        let m = ModelDir::open(d.path()).unwrap();
        assert_eq!(m.manifest.model_id, "qwen3-vl-4b");
        assert_eq!(m.mmproj().unwrap(), d.path().join("p.gguf"));
    }

    #[test]
    fn a_missing_or_short_file_is_a_refusal() {
        let d = dir_with(&[("m.gguf", b"12")], r#"{"model_id":"x","gguf":"m.gguf","mmproj":"p.gguf"}"#);
        assert!(ModelDir::open(d.path()).unwrap_err().contains("p.gguf"));
        let d = dir_with(&[("m.gguf", b"12")], r#"{"model_id":"x","gguf":"m.gguf","files":{"m.gguf":4}}"#);
        assert!(ModelDir::open(d.path()).unwrap_err().contains("partial"));
    }
}
