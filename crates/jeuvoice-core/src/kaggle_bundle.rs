//! Kaggle bundle and notebook builder (pure, plus the bundle zip itself).
//!
//! The app uploads one private dataset per run (`<name>.zip`) and pushes the
//! notebook text from [`notebook`]. On Kaggle the notebook unpacks `audio.pack`,
//! extracts audio tokens itself, trains, and leaves adapters plus `train.log`
//! in `/kaggle/working/result`. Audio tokens are made on Kaggle, not on the
//! user's PC. Layout mirrors `tts_lab/scripts/make_omni_kaggle.py`; the token
//! extraction step is the new part (tr2 packed ready-made tokens).

use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::train::{StartFrom, TrainSettings};

/// Commit of `k2-fsa/OmniVoice` the installs pin (PyPI has no LoRA training).
const OMNIVOICE_COMMIT: &str = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2";

/// The tr2 run settings, shipped with the app.
pub fn train_template() -> Value {
    serde_json::from_str(include_str!("../../../assets/kaggle/train_template.json")).expect("the shipped train template parses")
}

/// Fill the tr2 template (`assets/kaggle/train_template.json`) with the run
/// settings. Every other key stays exactly as tr2 ran it.
pub fn train_config(settings: &TrainSettings, template: &Value) -> Value {
    let mut out = template.clone();
    if let Some(map) = out.as_object_mut() {
        map.insert("steps".into(), settings.steps.into());
        map.insert("lora_r".into(), settings.lora_r.into());
        map.insert("lora_alpha".into(), settings.lora_alpha.into());
        map.insert("batch_tokens".into(), settings.batch_tokens.into());
        map.insert("save_steps".into(), settings.save_steps.into());
        map.insert("learning_rate".into(), settings.learning_rate.into());
        let start = match &settings.start_from {
            StartFrom::Base => "k2-fsa/OmniVoice".to_string(),
            StartFrom::Adapter { repo_or_dir } => repo_or_dir.clone(),
        };
        map.insert("init_from_checkpoint".into(), start.into());
    }
    out
}

/// The trainer's data config: token manifests the extraction step writes.
pub fn data_config(manifest_dir: &str) -> Value {
    serde_json::json!({
        "train": [{"manifest_path": [format!("{manifest_dir}/train/data.lst")]}],
        "dev": [{"manifest_path": [format!("{manifest_dir}/dev/data.lst")]}],
    })
}

/// Rewrite raw manifest rows for the bundle: `audio_path` becomes
/// `audio/<id>.<ext>`, relative to the run folder the notebook unpacks to.
pub fn bundle_manifests(jsonl: &str) -> Result<Vec<Value>, String> {
    jsonl
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut row: Value =
                serde_json::from_str(line).map_err(|e| format!("The manifest has a broken row: {e}"))?;
            let id = row.get("id").and_then(|v| v.as_str()).unwrap_or("clip").to_string();
            let ext = row
                .get("audio_path")
                .and_then(|v| v.as_str())
                .and_then(|p| Path::new(p).extension())
                .and_then(|e| e.to_str())
                .unwrap_or("wav");
            row["audio_path"] = format!("audio/{id}.{ext}").into();
            Ok(row)
        })
        .collect()
}

/// The notebook text (`.ipynb` JSON). `bundle_marker` is the `bundle_<name>.txt`
/// file the notebook finds its dataset with. Cells:
/// 1. `nvidia-smi` and the pinned installs (torchao goes first: peft breaks on it).
/// 2. Find the bundle, copy it to `/kaggle/working/run`, unpack `audio.pack`.
/// 3. Extract audio tokens with `omnivoice.scripts.extract_audio_tokens`.
/// 4. Train with `accelerate launch -m omnivoice.cli.train`, pruning older
///    checkpoints to their adapters (a full one is about 3 GB).
/// 5. Copy adapters plus `train.log` to `/kaggle/working/result`, delete the rest.
/// 6. Print `JEUVOICE_RUN_DONE`.
pub fn notebook(name: &str, bundle_marker: &str) -> String {
    let setup = format!(
        "!nvidia-smi\n!pip uninstall -y -q torchao\n!pip install -q 'omnivoice[lora] @ git+https://github.com/k2-fsa/OmniVoice@{OMNIVOICE_COMMIT}' transformers==5.19.0 peft==0.21.2 accelerate==1.15.0"
    );
    let find = format!(
        "import glob, json, os, shutil, subprocess, tarfile, time, torch\n\
         src = os.path.dirname(glob.glob('/kaggle/input/**/{bundle_marker}', recursive=True)[0])\n\
         run = '/kaggle/working/run'\n\
         shutil.copytree(src, run, dirs_exist_ok=True)\n\
         os.chdir(run)\n\
         with tarfile.open('audio.pack') as t:\n\
         {ind}t.extractall('.', filter='data')\n\
         os.remove('audio.pack')\n\
         for split in ('train', 'dev'):\n\
         {ind}rows = [json.loads(l) for l in open(f'manifests/{{split}}.jsonl', encoding='utf-8')]\n\
         {ind}missing = [r['audio_path'] for r in rows if not os.path.exists(r['audio_path'])]\n\
         {ind}assert not missing, f'{{len(missing)}} clips are missing, for example {{missing[0]}}'\n\
         {ind}print(split, len(rows), 'clips')\n",
        ind = "    "
    );
    let tokens = format!(
        "for split_jsonl, split_tokens in (('manifests/train.jsonl', 'tokens/train'), ('manifests/dev.jsonl', 'tokens/dev')):\n\
         {ind}cmd = ['python', '-m', 'omnivoice.scripts.extract_audio_tokens',\n\
         {ind}       '--input_jsonl', split_jsonl,\n\
         {ind}       '--tar_output_pattern', f'{{split_tokens}}/audios/shard-%06d.tar',\n\
         {ind}       '--jsonl_output_pattern', f'{{split_tokens}}/txts/shard-%06d.jsonl']\n\
         {ind}subprocess.run(cmd, check=True)\n\
         {ind}n = len(open(f'{{split_tokens}}/data.lst', encoding='utf-8').read().splitlines())\n\
         {ind}print(split_tokens, n, 'shards')\n",
        ind = "    "
    );
    let train = format!(
        "out = '/kaggle/working/{name}'\n\
         os.makedirs(out, exist_ok=True)\n\
         ngpu = torch.cuda.device_count()\n\
         os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n\
         cmd = ['accelerate', 'launch', '--mixed_precision', 'fp16', '--num_processes', str(ngpu)] + \
         (['--multi_gpu'] if ngpu > 1 else []) + ['-m', 'omnivoice.cli.train',\n\
         \x20      '--train_config', 'config/train.json', '--data_config', 'config/data.json', '--output_dir', out]\n\
         # A full checkpoint is about 3 GB and Kaggle gives 20 GB: older checkpoints keep only the adapter\n\
         def prune():\n\
         \x20   for ck in sorted(glob.glob(f'{{out}}/checkpoint-*'), key=lambda c: int(c.rsplit('-', 1)[1]))[:-1]:\n\
         \x20       for f in glob.glob(f'{{ck}}/*'):\n\
         \x20           if os.path.basename(f) not in ('adapter_config.json', 'adapter_model.safetensors'):\n\
         \x20               shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)\n\
         proc = subprocess.Popen(cmd)\n\
         while proc.poll() is None:\n\
         \x20   time.sleep(60)\n\
         \x20   prune()\n\
         prune()\n\
         assert proc.returncode == 0, f'training failed with code {{proc.returncode}}'\n"
    );
    let result = format!(
        "out = '/kaggle/working/{name}'\n\
         res = '/kaggle/working/result'\n\
         for ck in sorted(glob.glob(f'{{out}}/checkpoint-*')):\n\
         \x20   dst = f'{{res}}/{{os.path.basename(ck)}}'\n\
         \x20   os.makedirs(dst, exist_ok=True)\n\
         \x20   for f in ('adapter_config.json', 'adapter_model.safetensors'):\n\
         \x20       shutil.copy(f'{{ck}}/{{f}}', dst)\n\
         shutil.copy(f'{{out}}/train.log', res)\n\
         shutil.rmtree(out)\n\
         shutil.rmtree('/kaggle/working/run')\n\
         print(os.listdir(res))\n\
         print('JEUVOICE_RUN_DONE')\n"
    );
    let cells: Vec<(&str, String)> = vec![
        ("markdown", format!("# {name}: OmniVoice Turkish LoRA\n\nSettings: GPU **T4 x2**, Internet **on**. Then **Save Version → Save & Run All**.")),
        ("code", setup),
        ("code", find),
        ("code", tokens.to_string()),
        ("code", train),
        ("code", result),
    ];
    let json_cells: Vec<Value> = cells
        .into_iter()
        .map(|(kind, source)| {
            let mut cell = serde_json::json!({"cell_type": kind, "metadata": {}, "source": lines(&source)});
            if kind == "code" {
                cell["outputs"] = Value::Array(Vec::new());
                cell["execution_count"] = Value::Null;
            }
            cell
        })
        .collect();
    serde_json::json!({
        "cells": json_cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    })
    .to_string()
}

/// Notebook lines: `["line\n", ...]`, like the stock exporter writes.
fn lines(source: &str) -> Vec<String> {
    source.split_inclusive('\n').map(str::to_string).collect()
}

/// Build `<dir>/<name>.zip`: the marker, both configs, both manifests (with
/// relative audio paths) and `audio.pack` — a tar of the WAVs under another
/// suffix, because Kaggle unpacks every `.tar` in an upload.
///
/// `clips_dir` holds the cleaned WAVs; `manifests` holds `train.jsonl` and
/// `dev.jsonl` from `/prep/manifest`. Each row takes its audio from
/// `clips_dir/<id>.<ext>` first, then from its own absolute `audio_path`.
pub fn build_bundle(
    dir: &Path,
    name: &str,
    clips_dir: &Path,
    manifests: &Path,
    cfg: &Value,
    data: &Value,
) -> Result<PathBuf, String> {
    let mut splits: Vec<(&str, Vec<(PathBuf, Value)>, String)> = Vec::new();
    for split in ["train", "dev"] {
        let text = std::fs::read_to_string(manifests.join(format!("{split}.jsonl")))
            .map_err(|e| format!("The {split} manifest could not be read: {e}"))?;
        let rewritten = bundle_manifests(&text)?;
        let originals: Vec<Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()
            .map_err(|e| format!("The {split} manifest has a broken row: {e}"))?;
        let mut out = String::new();
        let mut pairs = Vec::with_capacity(rewritten.len());
        for (orig, new) in originals.into_iter().zip(rewritten) {
            out.push_str(&serde_json::to_string(&new).map_err(|e| format!("The {split} manifest could not be packed: {e}"))?);
            out.push('\n');
            let from = orig.get("audio_path").and_then(|v| v.as_str()).map(PathBuf::from).unwrap_or_default();
            pairs.push((from, new));
        }
        splits.push((split, pairs, out));
    }
    let mut tar_buf = Vec::new();
    {
        let mut tar = tar::Builder::new(&mut tar_buf);
        for (_, pairs, _) in &splits {
            for (from, row) in pairs {
                let id = row.get("id").and_then(|v| v.as_str()).unwrap_or("clip");
                let entry = row.get("audio_path").and_then(|v| v.as_str()).unwrap_or("audio/clip.wav").to_string();
                let ext = entry.rsplit('.').next().unwrap_or("wav");
                let bytes = [clips_dir.join(format!("{id}.{ext}")), from.clone()]
                    .into_iter()
                    .find_map(|p| std::fs::read(p).ok())
                    .ok_or_else(|| format!("The clip {id} has no audio file."))?;
                let mut header = tar::Header::new_gnu();
                header.set_size(bytes.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                tar.append_data(&mut header, &entry, &bytes[..])
                    .map_err(|e| format!("The clip {id} could not be packed: {e}"))?;
            }
        }
        tar.into_inner().map_err(|e| format!("The audio pack could not be written: {e}"))?;
    }
    let mut zip_buf = Vec::new();
    {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut zip_buf));
        let opts = zip::write::SimpleFileOptions::default();
        zip.start_file(format!("bundle_{name}.txt"), opts).map_err(plain)?;
        zip.write_all(name.as_bytes()).map_err(plain)?;
        zip.start_file("config/train.json", opts).map_err(plain)?;
        zip.write_all(cfg.to_string().as_bytes()).map_err(plain)?;
        zip.start_file("config/data.json", opts).map_err(plain)?;
        zip.write_all(data.to_string().as_bytes()).map_err(plain)?;
        for (split, _, text) in &splits {
            zip.start_file(format!("manifests/{split}.jsonl"), opts).map_err(plain)?;
            zip.write_all(text.as_bytes()).map_err(plain)?;
        }
        zip.start_file("audio.pack", opts).map_err(plain)?;
        zip.write_all(&tar_buf).map_err(plain)?;
        zip.finish().map_err(plain)?;
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("The bundle folder could not be made: {e}"))?;
    let out = dir.join(format!("{name}.zip"));
    std::fs::write(&out, zip_buf).map_err(|e| format!("The bundle could not be saved: {e}"))?;
    return Ok(out);

    fn plain<E: std::fmt::Display>(e: E) -> String {
        format!("The bundle could not be packed: {e}")
    }
}
