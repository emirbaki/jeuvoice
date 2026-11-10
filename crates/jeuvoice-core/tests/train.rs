//! Training data model and the tr2 recipe.

use std::collections::HashMap;

use jeuvoice_core::train::{Clip, Recipe, Source, minutes_by_source};

fn clip(id: &str, source: &str, secs: f32) -> Clip {
    Clip { id: id.into(), audio: format!("{id}.wav"), text: "merhaba".into(), duration_s: secs, source: source.into(), excluded: None }
}

#[test]
fn recipe_json_round_trip() {
    let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/recipes/tr2.json")).unwrap();
    let recipe: Recipe = serde_json::from_str(&json).unwrap();
    assert_eq!(serde_json::from_str::<Recipe>(&serde_json::to_string(&recipe).unwrap()).unwrap(), recipe);
}

#[test]
fn tr2_recipe_has_five_sources_with_tr2_weights() {
    let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/recipes/tr2.json")).unwrap();
    let recipe: Recipe = serde_json::from_str(&json).unwrap();
    let weights: HashMap<&str, u32> = recipe.sources.iter().map(|s| (s.name.as_str(), s.weight)).collect();
    assert_eq!(
        weights,
        HashMap::from([("alania_a8_tr2", 1), ("yodas_tr2", 1), ("antalia", 1), ("my_voice_a8", 2), ("my_voice_a8_clean", 2)])
    );
    assert_eq!(recipe.base_model, "k2-fsa/OmniVoice");
    assert_eq!(recipe.settings.steps, 6000);
    assert_eq!(recipe.settings.lora_r, 32);
}

#[test]
fn minutes_by_source_applies_weights_and_skips_excluded() {
    let sources = vec![
        Source { name: "a".into(), kind: jeuvoice_core::train::SourceKind::LocalFolder { dir: "a".into() }, weight: 1, license: None, note: None },
        Source { name: "b".into(), kind: jeuvoice_core::train::SourceKind::LocalFolder { dir: "b".into() }, weight: 3, license: None, note: None },
    ];
    let mut clips = vec![clip("a1", "a", 60.0), clip("b1", "b", 60.0), clip("b2", "b", 60.0)];
    clips.push(Clip { excluded: Some("noisy".into()), ..clip("b3", "b", 60.0) });
    let minutes = minutes_by_source(&clips, &sources);
    assert_eq!(minutes, vec![("a".to_string(), 1.0, 1.0), ("b".to_string(), 2.0, 6.0)]);
}
