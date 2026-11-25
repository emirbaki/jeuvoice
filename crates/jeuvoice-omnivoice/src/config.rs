//! The config of the OmniVoice main model and the paths of its files.

use std::path::{Path, PathBuf};

use crate::{Result, TtsError};

/// The special token ids of the OmniVoice layout. The `config.json` holds only the end and
/// pad tokens, so the others are constants of the model, as in `ommivoice.py:1198-1273`.
pub const TOK_DENOISE: u32 = 151669;
pub const TOK_LANG_START: u32 = 151670;
pub const TOK_LANG_END: u32 = 151671;
pub const TOK_INSTRUCT_START: u32 = 151672;
pub const TOK_INSTRUCT_END: u32 = 151673;
pub const TOK_TEXT_START: u32 = 151674;
pub const TOK_TEXT_END: u32 = 151675;

/// The Qwen3 backbone and the audio head of the OmniVoice checkpoint.
///
/// The values come from `config.json`: the top level holds the audio fields, and `llm_config`
/// the Qwen3 fields.
#[derive(Debug, Clone, PartialEq)]
pub struct OmniVoiceConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    /// The codebooks of one audio frame: 8.
    pub num_audio_codebook: usize,
    /// The codes of one codebook plus the mask: 1025.
    pub audio_vocab_size: usize,
    /// The mask id inside a codebook: 1024.
    pub audio_mask_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl OmniVoiceConfig {
    /// Reads `config.json`. A missing key or a wrong type is a `ModelLoad` error with the key
    /// name.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| TtsError::ModelLoad(format!("could not read {}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|e| TtsError::ModelLoad(format!("config.json is not JSON: {e}")))?;
        let llm = root.get("llm_config").ok_or_else(|| TtsError::ModelLoad("config.json has no llm_config".into()))?;
        let int = |v: &serde_json::Value, key: &str| -> Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| TtsError::ModelLoad(format!("config.json has no integer {key}")))
        };
        let num = |v: &serde_json::Value, key: &str| -> Result<f32> {
            v.get(key)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| TtsError::ModelLoad(format!("config.json has no number {key}")))
        };
        let rope = llm
            .get("rope_parameters")
            .ok_or_else(|| TtsError::ModelLoad("config.json has no llm_config.rope_parameters".into()))?;
        let rope_type = rope.get("rope_type").and_then(|x| x.as_str()).unwrap_or("");
        if rope_type != "default" {
            return Err(TtsError::ModelLoad(format!("only the default RoPE is supported, not {rope_type:?}")));
        }
        Ok(Self {
            num_hidden_layers: int(llm, "num_hidden_layers")?,
            hidden_size: int(llm, "hidden_size")?,
            num_attention_heads: int(llm, "num_attention_heads")?,
            num_key_value_heads: int(llm, "num_key_value_heads")?,
            head_dim: int(llm, "head_dim")?,
            intermediate_size: int(llm, "intermediate_size")?,
            rms_norm_eps: num(llm, "rms_norm_eps")?,
            rope_theta: num(rope, "rope_theta")?,
            vocab_size: int(llm, "vocab_size")?,
            num_audio_codebook: int(&root, "num_audio_codebook")?,
            audio_vocab_size: int(&root, "audio_vocab_size")?,
            audio_mask_id: int(&root, "audio_mask_id")? as u32,
            eos_token_id: int(&root, "eos_token_id")? as u32,
            pad_token_id: int(&root, "pad_token_id")? as u32,
        })
    }

    /// The rows of the audio head: one block of `audio_vocab_size` logits per codebook.
    pub fn audio_head_rows(&self) -> usize {
        self.num_audio_codebook * self.audio_vocab_size
    }

    /// The width of the query projection: `num_attention_heads * head_dim` (2048 here, twice
    /// the hidden size: Qwen3 keeps 16 heads of 128 in a 1024 wide model).
    pub fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// The width of the key and value projections: `num_key_value_heads * head_dim` (1024).
    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Rejects a checkpoint that is not the large Qwen3 backbone of OmniVoice. The arenas and
    /// the kernels assume these shapes, and a smaller Qwen3 model would load and then give
    /// wrong audio in silence.
    pub fn check(&self) -> Result<()> {
        let want = [
            ("num_hidden_layers", self.num_hidden_layers, 28),
            ("hidden_size", self.hidden_size, 1024),
            ("num_attention_heads", self.num_attention_heads, 16),
            ("num_key_value_heads", self.num_key_value_heads, 8),
            ("head_dim", self.head_dim, 128),
            ("intermediate_size", self.intermediate_size, 3072),
            ("vocab_size", self.vocab_size, 151676),
            ("num_audio_codebook", self.num_audio_codebook, 8),
            ("audio_vocab_size", self.audio_vocab_size, 1025),
            ("audio_mask_id", self.audio_mask_id as usize, 1024),
        ];
        let wrong: Vec<String> = want.iter().filter(|(_, got, want)| got != want).map(|(name, got, want)| format!("{name} {got}, not {want}")).collect();
        if !wrong.is_empty() {
            return Err(TtsError::ModelLoad(format!(
                "the GPU path needs the OmniVoice Qwen3-0.6B checkpoint (28 layers, width 1024, 8 codebooks of 1025): {}",
                wrong.join(", ")
            )));
        }
        Ok(())
    }
}

/// The snapshot folder of the OmniVoice checkpoint in the Hugging Face cache: the first
/// `models--k2-fsa--OmniVoice/snapshots/<rev>/` folder that holds a `model.safetensors`.
///
/// The environment variable `JEUVOICE_OMNIVOICE_MODEL` wins, which the tests and the tools use
/// to point at another copy.
pub fn model_dir(cache_root: &Path, env: Option<&str>) -> Result<PathBuf> {
    if let Some(dir) = env.map(str::trim).filter(|s| !s.is_empty()) {
        let dir = PathBuf::from(dir);
        return if dir.join("model.safetensors").is_file() {
            Ok(dir)
        } else {
            Err(TtsError::ModelLoad(format!("{} has no model.safetensors", dir.display())))
        };
    }
    let snapshots = cache_root.join("models--k2-fsa--OmniVoice").join("snapshots");
    let mut found: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .map_err(|e| TtsError::ModelLoad(format!("could not read {}: {e}. Download the model with the engine once.", snapshots.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("model.safetensors").is_file())
        .collect();
    found.sort();
    found
        .pop()
        .ok_or_else(|| TtsError::ModelLoad(format!("no model.safetensors under {}", snapshots.display())))
}

/// The Hugging Face cache folder of this user.
pub fn default_cache_root() -> PathBuf {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
    PathBuf::from(home).join(".cache").join("huggingface").join("hub")
}
