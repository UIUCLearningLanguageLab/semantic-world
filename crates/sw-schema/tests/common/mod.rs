//! Shared helpers for the schema crate's integration tests.
#![allow(dead_code)] // each test file uses a subset

use std::path::{Path, PathBuf};

/// The repository root.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh copy of `data/` in a temporary directory, so a test can break one file.
pub struct DataCopy {
    pub dir: PathBuf,
}

impl DataCopy {
    pub fn new(label: &str) -> DataCopy {
        let dir = std::env::temp_dir().join(format!(
            "sw-schema-test-{}-{}-{label}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        copy_dir(&repo_root().join("data"), &dir);
        DataCopy { dir }
    }

    pub fn path(&self, relative: &str) -> PathBuf {
        self.dir.join(relative)
    }

    /// Replace text in one file. Panics if the text is not there, so a stale fixture fails
    /// loudly.
    pub fn replace(&self, relative: &str, from: &str, to: &str) {
        let path = self.path(relative);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(from), "{relative} does not contain {from:?}");
        std::fs::write(&path, text.replacen(from, to, 1)).unwrap();
    }

    pub fn write(&self, relative: &str, text: &str) {
        let path = self.path(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
}

impl Drop for DataCopy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}
