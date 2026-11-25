//! The safetensors index: the tensor names, types and shapes of a weight file, without reading
//! the data. The data of one tensor is read on demand, so the 2.45 GB checkpoint never sits in
//! memory as a whole.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::{Result, TtsError};

/// The type of a tensor in a safetensors file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDtype {
    F32,
    F16,
    BF16,
    I64,
    Other,
}

impl SafeDtype {
    fn from_str(s: &str) -> Self {
        match s {
            "F32" => SafeDtype::F32,
            "F16" => SafeDtype::F16,
            "BF16" => SafeDtype::BF16,
            "I64" => SafeDtype::I64,
            _ => SafeDtype::Other,
        }
    }

    /// The bytes of one value.
    pub fn size(self) -> usize {
        match self {
            SafeDtype::F32 => 4,
            SafeDtype::F16 | SafeDtype::BF16 => 2,
            SafeDtype::I64 => 8,
            SafeDtype::Other => 1,
        }
    }
}

/// One tensor of a safetensors file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: SafeDtype,
    /// In PyTorch order: the slowest dimension first.
    pub shape: Vec<usize>,
    /// The byte offset of the data inside the file.
    pub offset: u64,
    pub bytes: u64,
}

impl TensorEntry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The tensors of a safetensors file, by name.
#[derive(Debug, Clone)]
pub struct TensorIndex {
    pub path: PathBuf,
    /// The first byte of the data block.
    data_start: u64,
    tensors: BTreeMap<String, TensorEntry>,
}

impl TensorIndex {
    /// Reads the header only. The header holds one JSON object with `dtype`, `shape` and
    /// `data_offsets` for every tensor, after a `u64` length.
    pub fn open(path: &Path) -> Result<TensorIndex> {
        let mut f = File::open(path).map_err(|e| TtsError::ModelLoad(format!("could not open {}: {e}", path.display())))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| TtsError::ModelLoad(format!("{}: {e}", path.display())))?;
        let header_len = u64::from_le_bytes(len);
        // A header of 100 MB means this is not a safetensors file.
        if header_len == 0 || header_len > 100 * 1024 * 1024 {
            return Err(TtsError::ModelLoad(format!("{} is not a safetensors file (header {header_len} B)", path.display())));
        }
        let mut header = vec![0u8; header_len as usize];
        f.read_exact(&mut header).map_err(|e| TtsError::ModelLoad(format!("{}: {e}", path.display())))?;
        let json: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|e| TtsError::ModelLoad(format!("{} has a broken safetensors header: {e}", path.display())))?;
        let data_start = 8 + header_len;
        let mut tensors = BTreeMap::new();
        let obj = json
            .as_object()
            .ok_or_else(|| TtsError::ModelLoad(format!("{} has a safetensors header that is not an object", path.display())))?;
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = SafeDtype::from_str(v.get("dtype").and_then(|x| x.as_str()).unwrap_or(""));
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|x| x.as_array())
                .ok_or_else(|| TtsError::ModelLoad(format!("{name} has no shape")))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let offs = v
                .get("data_offsets")
                .and_then(|x| x.as_array())
                .ok_or_else(|| TtsError::ModelLoad(format!("{name} has no data_offsets")))?;
            let start = offs.first().and_then(|x| x.as_u64()).unwrap_or(0);
            let end = offs.get(1).and_then(|x| x.as_u64()).unwrap_or(0);
            tensors.insert(
                name.clone(),
                TensorEntry { name: name.clone(), dtype, shape, offset: data_start + start, bytes: end.saturating_sub(start) },
            );
        }
        Ok(TensorIndex { path: path.to_path_buf(), data_start, tensors })
    }

    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// The tensor named `name`, or a `ModelLoad` error that names it.
    pub fn entry(&self, name: &str) -> Result<&TensorEntry> {
        self.get(name)
            .ok_or_else(|| TtsError::ModelLoad(format!("{} has no tensor {name}", self.path.display())))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// The data of one tensor as f32. An F32 tensor is read as is, an F16 or BF16 tensor is
    /// widened, and another type is an error.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?.clone();
        let mut f = File::open(&self.path).map_err(|e| TtsError::ModelLoad(format!("could not open {}: {e}", self.path.display())))?;
        f.seek(SeekFrom::Start(e.offset)).map_err(|e| TtsError::ModelLoad(format!("{}: {e}", self.path.display())))?;
        let mut raw = vec![0u8; e.bytes as usize];
        f.read_exact(&mut raw).map_err(|e| TtsError::ModelLoad(format!("{}: {e}", self.path.display())))?;
        let want = e.numel();
        let got: Vec<f32> = match e.dtype {
            SafeDtype::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            SafeDtype::F16 => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            SafeDtype::BF16 => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            other => return Err(TtsError::ModelLoad(format!("{name} is {other:?}, not a float tensor"))),
        };
        if got.len() != want {
            return Err(TtsError::ModelLoad(format!("{name} holds {} values, not {want}", got.len())));
        }
        Ok(got)
    }
}
