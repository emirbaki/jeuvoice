//! The encoder weights on the GPU: every `encoder.*` tensor of the ggml model, as f16. The
//! ggml file stores 2-D weights as f16 and biases and norms as f32. The biases and norms
//! are rounded to f16 here, because the ops take one dtype.
//!
//! The weights go into the same arena as the activations: each op takes one arena for all
//! its tensors. `plan` reserves them in a shared `ArenaPlan`, `upload` fills them.

use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

use crate::WhisperError;
use crate::ggml::GgmlModel;

/// The tensors of one encoder block.
pub struct BlockWeights {
    pub attn_ln_w: Tensor,
    pub attn_ln_b: Tensor,
    pub q_w: Tensor,
    pub q_b: Tensor,
    /// The key projection has no bias in Whisper.
    pub k_w: Tensor,
    pub v_w: Tensor,
    pub v_b: Tensor,
    pub o_w: Tensor,
    pub o_b: Tensor,
    pub mlp_ln_w: Tensor,
    pub mlp_ln_b: Tensor,
    pub mlp1_w: Tensor,
    pub mlp1_b: Tensor,
    pub mlp2_w: Tensor,
    pub mlp2_b: Tensor,
}

pub struct EncoderWeights {
    /// The ggml name of each planned tensor, for `upload`.
    list: Vec<(String, Tensor)>,
    bytes: usize,
    pub n_mels: usize,
    /// Positions after conv2 (1500).
    pub n_ctx: usize,
    /// The model width (1280).
    pub d: usize,
    pub n_head: usize,
    /// The MLP width (5120).
    pub d_mlp: usize,
    /// `[1280, 128, 3]`
    pub conv1_w: Tensor,
    pub conv1_b: Tensor,
    /// `[1280, 1280, 3]`
    pub conv2_w: Tensor,
    pub conv2_b: Tensor,
    /// `[1500, 1280]`
    pub pos: Tensor,
    pub blocks: Vec<BlockWeights>,
    pub ln_post_w: Tensor,
    pub ln_post_b: Tensor,
}

impl EncoderWeights {
    /// Reserves every encoder tensor of `model` in `plan`, with the shape the ops need.
    pub fn plan(plan: &mut ArenaPlan, model: &GgmlModel) -> Result<EncoderWeights, WhisperError> {
        let h = &model.hparams;
        let (n_mels, n_ctx, d, n_head, n_layer) =
            (h.n_mels as usize, h.n_audio_ctx as usize, h.n_audio_state as usize, h.n_audio_head as usize, h.n_audio_layer as usize);

        let start = plan.total_bytes();
        let mut list: Vec<(String, Tensor)> = Vec::new();
        let mut add = |name: String, shape: &[usize]| -> Result<Tensor, WhisperError> {
            let src = model.tensor(&name).ok_or_else(|| WhisperError::ModelLoad(format!("the model has no tensor {name}")))?;
            let want: usize = shape.iter().product();
            if src.numel() != want {
                return Err(WhisperError::ModelLoad(format!("tensor {name} has shape {:?}, expected {shape:?}", src.shape)));
            }
            let t = plan.add(DType::F16, shape);
            list.push((name, t.clone()));
            Ok(t)
        };
        let conv1_w = add("encoder.conv1.weight".into(), &[d, n_mels, 3])?;
        let conv1_b = add("encoder.conv1.bias".into(), &[d])?;
        let conv2_w = add("encoder.conv2.weight".into(), &[d, d, 3])?;
        let conv2_b = add("encoder.conv2.bias".into(), &[d])?;
        let pos = add("encoder.positional_embedding".into(), &[n_ctx, d])?;
        let d_mlp = model
            .tensor("encoder.blocks.0.mlp.0.weight")
            .map(|t| t.shape[0])
            .ok_or_else(|| WhisperError::ModelLoad("the model has no encoder MLP".into()))?;
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("encoder.blocks.{i}.{s}");
            blocks.push(BlockWeights {
                attn_ln_w: add(p("attn_ln.weight"), &[d])?,
                attn_ln_b: add(p("attn_ln.bias"), &[d])?,
                q_w: add(p("attn.query.weight"), &[d, d])?,
                q_b: add(p("attn.query.bias"), &[d])?,
                k_w: add(p("attn.key.weight"), &[d, d])?,
                v_w: add(p("attn.value.weight"), &[d, d])?,
                v_b: add(p("attn.value.bias"), &[d])?,
                o_w: add(p("attn.out.weight"), &[d, d])?,
                o_b: add(p("attn.out.bias"), &[d])?,
                mlp_ln_w: add(p("mlp_ln.weight"), &[d])?,
                mlp_ln_b: add(p("mlp_ln.bias"), &[d])?,
                mlp1_w: add(p("mlp.0.weight"), &[d_mlp, d])?,
                mlp1_b: add(p("mlp.0.bias"), &[d_mlp])?,
                mlp2_w: add(p("mlp.2.weight"), &[d, d_mlp])?,
                mlp2_b: add(p("mlp.2.bias"), &[d])?,
            });
        }
        let ln_post_w = add("encoder.ln_post.weight".into(), &[d])?;
        let ln_post_b = add("encoder.ln_post.bias".into(), &[d])?;

        let bytes = plan.total_bytes() - start;
        Ok(EncoderWeights {
            list,
            bytes,
            n_mels,
            n_ctx,
            d,
            n_head,
            d_mlp,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            pos,
            blocks,
            ln_post_w,
            ln_post_b,
        })
    }

    /// Reads each planned tensor from `model` (rounded to f16) and uploads it into `arena`.
    pub fn upload(&self, gpu: &Gpu, arena: &Arena, model: &GgmlModel) -> Result<(), WhisperError> {
        for (name, t) in &self.list {
            let src = model.tensor(name).expect("plan checked every name");
            arena.upload(gpu, t, &model.read_f16(src)?)?;
        }
        Ok(())
    }

    /// Bytes of video memory the weights use.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}
