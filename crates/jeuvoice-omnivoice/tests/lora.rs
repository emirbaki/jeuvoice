//! The LoRA merge and the adapter names (plan Phase 2, Task 6). No GPU needed.

use jeuvoice_omnivoice::lora::{base_name, merge_into, other_half, replaced_name, scale_of};

/// A 2x2 merge that a reader can check by hand.
#[test]
fn merging_the_lora_into_a_2x2_matrix() {
    // W = I, A = [1, 2] (r = 1, in = 2), B = [3, 4] (out = 2, r = 1), scale 2.
    let mut w = vec![1.0, 0.0, 0.0, 1.0];
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    merge_into(&mut w, &a, &b, 2, 2, 1, 2.0);
    // B·A = [[3, 6], [4, 8]], times 2 is [[6, 12], [8, 16]].
    assert_eq!(w, vec![7.0, 12.0, 8.0, 17.0]);
}

#[test]
fn a_rank_two_merge_with_a_bias_like_second_row() {
    // out = 2, in = 3, r = 2: B·A with distinct values, so a wrong index order shows.
    let mut w = vec![0.0; 6];
    let a = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]; // A = [[1,0,1],[0,1,1]]
    let b = vec![1.0, 10.0, 5.0, 5.0]; // B = [[1,10],[5,5]]
    merge_into(&mut w, &a, &b, 2, 3, 2, 1.0);
    // Row 0: 1·[1,0,1] + 10·[0,1,1] = [1, 10, 11]. Row 1: 5·[1,0,1] + 5·[0,1,1] = [5, 5, 10].
    assert_eq!(w, vec![1.0, 10.0, 11.0, 5.0, 5.0, 10.0]);
}

#[test]
fn a_zero_b_block_leaves_the_weight_alone() {
    let mut w = vec![0.5, -0.5, 1.0, 2.0];
    let before = w.clone();
    merge_into(&mut w, &[1.0, 1.0], &[0.0, 0.0], 2, 2, 1, 2.0);
    assert_eq!(w, before);
}

#[test]
fn the_scale_is_alpha_over_r() {
    assert_eq!(scale_of(64.0, 32), 2.0);
    assert_eq!(scale_of(16.0, 16), 1.0);
    let r = std::panic::catch_unwind(|| scale_of(8.0, 0));
    assert!(r.is_err(), "r = 0 must panic with a message");
}

#[test]
fn adapter_names_map_to_the_base_names() {
    let key = "base_model.model.llm.layers.3.mlp.down_proj.lora_A.weight";
    assert_eq!(base_name(key).unwrap(), "llm.layers.3.mlp.down_proj.weight");
    assert_eq!(other_half(key).unwrap(), "base_model.model.llm.layers.3.mlp.down_proj.lora_B.weight");
    let b = "base_model.model.llm.layers.0.self_attn.q_proj.lora_B.weight";
    assert_eq!(base_name(b).unwrap(), "llm.layers.0.self_attn.q_proj.weight");
    assert_eq!(other_half(b).unwrap(), "base_model.model.llm.layers.0.self_attn.q_proj.lora_A.weight");

    // A tensors that are not LoRA halves.
    assert!(base_name("base_model.model.audio_embeddings.weight").is_none());
    assert!(base_name("llm.layers.0.self_attn.q_proj.weight").is_none());
    assert!(other_half("base_model.model.audio_heads.weight").is_none());
}

#[test]
fn the_two_modules_to_save_tables_replace_the_base_ones() {
    assert_eq!(replaced_name("base_model.model.audio_embeddings.weight").unwrap(), "audio_embeddings.weight");
    assert_eq!(replaced_name("base_model.model.audio_heads.weight").unwrap(), "audio_heads.weight");
    assert!(replaced_name("base_model.model.llm.layers.0.mlp.up_proj.lora_A.weight").is_none());
    assert!(replaced_name("base_model.model.llm.embed_tokens.weight").is_none());
}
