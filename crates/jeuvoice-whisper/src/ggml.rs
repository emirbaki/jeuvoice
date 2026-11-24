//! Reads a whisper.cpp ggml model file (`ggml-large-v3-turbo.bin`) in plain Rust.
//!
//! The layout comes from `whisper_model_load` in whisper.cpp (whisper-rs-sys 0.15.0,
//! `whisper.cpp/src/whisper.cpp`, line 1485). All numbers are little-endian:
//!
//! 1. `u32` magic `0x67676d6c` ("ggml").
//! 2. 11 `i32` hparams, in the order of the `Hparams` fields.
//! 3. Mel filters: `i32 n_mel`, `i32 n_fft`, then `n_mel * n_fft` `f32`.
//! 4. Vocab: `i32 n`, then for each token a `u32` length and the bytes.
//! 5. Tensor records until the end of the file: `i32 n_dims`, `i32 name_len`, `i32 ttype`,
//!    `n_dims` `i32` sizes (ggml order: the fastest dimension first), the name, then the
//!    data with no padding. `ttype` 0 is f32 and 1 is f16. Other types are quantized.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::f16;

use crate::WhisperError;

const MAGIC: u32 = 0x6767_6d6c;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hparams {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    pub ftype: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
}

impl GgmlType {
    fn size(self) -> u64 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgmlTensor {
    pub name: String,
    pub dtype: GgmlType,
    /// PyTorch order: the slowest dimension first.
    pub shape: Vec<usize>,
    /// Byte offset of the data in the file.
    pub data_offset: u64,
    pub data_len: u64,
}

impl GgmlTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug)]
pub struct GgmlModel {
    pub hparams: Hparams,
    /// `(n_mel, n_fft, data)`: the mel filter bank, `n_mel` rows of `n_fft` weights.
    pub mel_filters: (usize, usize, Vec<f32>),
    /// The bytes of each token. Whisper tokens are byte-level BPE, so not always UTF-8.
    pub vocab: Vec<Vec<u8>>,
    pub tensors: Vec<GgmlTensor>,
    path: PathBuf,
}

fn bad(why: impl Into<String>) -> WhisperError {
    WhisperError::ModelLoad(why.into())
}

/// A reader that turns a short read into a plain `ModelLoad` error.
struct R<T> {
    inner: T,
    what: &'static str,
}

impl<T: Read + Seek> R<T> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, WhisperError> {
        let mut b = vec![0u8; n];
        self.inner.read_exact(&mut b).map_err(|e| bad(format!("the model file ends early in the {}: {e}", self.what)))?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, WhisperError> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().expect("4 bytes")))
    }
    fn i32(&mut self) -> Result<i32, WhisperError> {
        Ok(i32::from_le_bytes(self.bytes(4)?.try_into().expect("4 bytes")))
    }
    /// A size field: negative or huge values are a broken file, not a panic.
    fn len(&mut self, limit: usize) -> Result<usize, WhisperError> {
        let v = self.i32()?;
        usize::try_from(v).ok().filter(|&n| n <= limit).ok_or_else(|| bad(format!("bad size {v} in the {}", self.what)))
    }
}

impl GgmlModel {
    /// Read the header, the mel filters, the vocab and the tensor list. Tensor data stays
    /// in the file until `read_f32` / `read_f16`.
    pub fn open(path: &Path) -> Result<GgmlModel, WhisperError> {
        let file = File::open(path).map_err(|e| bad(format!("could not open {}: {e}", path.display())))?;
        let file_len = file.metadata().map_err(|e| bad(e.to_string()))?.len();
        let mut r = R { inner: BufReader::new(file), what: "header" };
        if r.u32()? != MAGIC {
            return Err(bad("this is not a ggml model file (bad magic)"));
        }
        let mut h = [0i32; 11];
        for v in &mut h {
            *v = r.i32()?;
        }
        let hparams = Hparams {
            n_vocab: h[0],
            n_audio_ctx: h[1],
            n_audio_state: h[2],
            n_audio_head: h[3],
            n_audio_layer: h[4],
            n_text_ctx: h[5],
            n_text_state: h[6],
            n_text_head: h[7],
            n_text_layer: h[8],
            n_mels: h[9],
            ftype: h[10],
        };

        r.what = "mel filters";
        let n_mel = r.len(1 << 12)?;
        let n_fft = r.len(1 << 16)?;
        let raw = r.bytes(n_mel * n_fft * 4)?;
        let filters = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();

        r.what = "vocab";
        let n_vocab = r.len(1 << 20)?;
        let mut vocab = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            let n = r.u32()? as usize;
            if n > 1 << 16 {
                return Err(bad(format!("bad token length {n} in the vocab")));
            }
            vocab.push(r.bytes(n)?);
        }

        r.what = "tensor list";
        let mut tensors = Vec::new();
        let mut pos = r.inner.stream_position().map_err(|e| bad(e.to_string()))?;
        while pos < file_len {
            let n_dims = r.len(4)?;
            let name_len = r.len(1 << 10)?;
            let ttype = r.i32()?;
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(r.len(1 << 30)?);
            }
            let name = String::from_utf8(r.bytes(name_len)?).map_err(|_| bad("a tensor name is not UTF-8"))?;
            let dtype = match ttype {
                0 => GgmlType::F32,
                1 => GgmlType::F16,
                t => return Err(bad(format!("tensor {name} has type {t}. Only f32 and f16 models are supported, not quantized ones"))),
            };
            ne.reverse();
            // Checked: a corrupt record must give an error, not a wrapped size.
            let data_len = ne
                .iter()
                .try_fold(dtype.size(), |n, &d| n.checked_mul(d as u64))
                .filter(|&n| usize::try_from(n).is_ok())
                .ok_or_else(|| bad(format!("tensor {name} is too large: {ne:?}")))?;
            let data_offset = r.inner.stream_position().map_err(|e| bad(e.to_string()))?;
            if data_offset + data_len > file_len {
                return Err(bad(format!("the model file ends early in tensor {name}")));
            }
            r.inner.seek(SeekFrom::Start(data_offset + data_len)).map_err(|e| bad(e.to_string()))?;
            tensors.push(GgmlTensor { name, dtype, shape: ne, data_offset, data_len });
            pos = data_offset + data_len;
        }

        Ok(GgmlModel { hparams, mel_filters: (n_mel, n_fft, filters), vocab, tensors, path: path.to_path_buf() })
    }

    /// Checks that the GPU encoder can run this model: the large-v3-turbo encoder shapes
    /// (128 mel bands with 201 filter bins, 1500 positions, heads of 64, at least one layer).
    pub fn check_gpu_encoder(&self) -> Result<(), WhisperError> {
        let h = &self.hparams;
        let fail = |what: String| Err(bad(format!("the GPU path supports Whisper large-v3 models only: {what}")));
        if h.n_mels != 128 || self.mel_filters.0 != 128 || self.mel_filters.1 != 201 {
            return fail(format!("{} mel bands with {}x{} filters, not 128 with 128x201", h.n_mels, self.mel_filters.0, self.mel_filters.1));
        }
        if h.n_audio_ctx != 1500 {
            return fail(format!("{} audio positions, not 1500", h.n_audio_ctx));
        }
        if h.n_audio_head <= 0 || h.n_audio_state != 64 * h.n_audio_head {
            return fail(format!("width {} with {} heads, not heads of 64", h.n_audio_state, h.n_audio_head));
        }
        if h.n_audio_layer <= 0 {
            return fail(format!("{} encoder layers", h.n_audio_layer));
        }
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Option<&GgmlTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    fn raw(&self, t: &GgmlTensor) -> Result<Vec<u8>, WhisperError> {
        let mut f = File::open(&self.path).map_err(|e| bad(e.to_string()))?;
        f.seek(SeekFrom::Start(t.data_offset)).map_err(|e| bad(e.to_string()))?;
        let mut b = vec![0u8; t.data_len as usize];
        f.read_exact(&mut b).map_err(|e| bad(format!("could not read tensor {}: {e}", t.name)))?;
        Ok(b)
    }

    /// The tensor data as f32 (an f16 tensor is converted).
    pub fn read_f32(&self, t: &GgmlTensor) -> Result<Vec<f32>, WhisperError> {
        let b = self.raw(t)?;
        Ok(match t.dtype {
            GgmlType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect(),
            GgmlType::F16 => b.chunks_exact(2).map(|c| f16::from_le_bytes(c.try_into().expect("2 bytes")).to_f32()).collect(),
        })
    }

    /// The tensor data as f16 (an f32 tensor is rounded).
    pub fn read_f16(&self, t: &GgmlTensor) -> Result<Vec<f16>, WhisperError> {
        let b = self.raw(t)?;
        Ok(match t.dtype {
            GgmlType::F16 => b.chunks_exact(2).map(|c| f16::from_le_bytes(c.try_into().expect("2 bytes"))).collect(),
            GgmlType::F32 => b.chunks_exact(4).map(|c| f16::from_f32(f32::from_le_bytes(c.try_into().expect("4 bytes")))).collect(),
        })
    }
}
