//! The Train tile board: three summary tiles, and the layout with one tile open.
//!
//! Board tiles show numbers and state only. A click opens a tile. The open tile
//! holds the controls, and the other tiles move into a narrow column on the right.

use jeuvoice_core::train::StartFrom;
use rune::tok::{space, type_};
use rune::{El, ScrollSpec, col, el, row, text};

use crate::state::App;
use crate::style::{self, ButtonKind};
use crate::train::model_step::AdapterChoice;
use crate::train::run::RunState;
use crate::train::state::{TrainData, TrainTarget, TrainTile, fmt_minutes};
use crate::transcribe_tab::MEDIA;

/// Sources listed on the board Data tile before "+N more".
const BOARD_SOURCES: usize = 5;
/// The width of the side column when a tile is open.
const COLUMN_W: f32 = 190.0;

/// What the Data tile shows.
#[derive(Clone, Debug, PartialEq)]
pub struct DataSummary {
    pub clips: usize,
    pub minutes: f32,
    pub left_out: usize,
    pub band_khz: Option<u32>,
    /// Source names with their clip counts, in project order.
    pub sources: Vec<(String, usize)>,
}

pub fn data_summary(s: &TrainData) -> DataSummary {
    let included: Vec<_> = s.clips().iter().filter(|c| c.clip.excluded.is_none()).collect();
    let minutes = included.iter().map(|c| c.clip.duration_s).sum::<f32>() / 60.0;
    let mut bands: Vec<f32> = included.iter().map(|c| c.band_khz).collect();
    bands.sort_by(f32::total_cmp);
    DataSummary {
        clips: included.len(),
        minutes,
        left_out: s.excluded_count(),
        band_khz: bands.get(bands.len() / 2).map(|b| b.round() as u32),
        sources: s.sources().iter().map(|src| (src.name.clone(), s.clips_of(&src.name).len())).collect(),
    }
}

/// What the Run tile shows.
#[derive(Clone, Debug, PartialEq)]
pub enum RunSummary {
    /// No clips yet.
    NotReady,
    /// Clips are in; a Kaggle run takes about `hours`.
    Ready { hours: f32 },
    Running { step: u32, total: u32 },
    Done { checkpoints: usize },
    Failed,
}

pub fn run_summary(s: &TrainData) -> RunSummary {
    match s.run.as_ref().map(|r| r.state) {
        Some(RunState::Queued | RunState::Running) => RunSummary::Running {
            step: s.losses.last().map_or(0, |(step, _)| *step),
            total: s.project.settings.steps,
        },
        Some(RunState::Complete) => RunSummary::Done { checkpoints: s.checkpoints.len() },
        Some(RunState::Error | RunState::Cancelled) => RunSummary::Failed,
        None if s.included_clips().is_empty() => RunSummary::NotReady,
        None => RunSummary::Ready { hours: s.project.settings.steps as f32 / 1000.0 },
    }
}

fn run_line(s: &TrainData, r: &RunSummary) -> String {
    match r {
        RunSummary::NotReady => "Add clips first".into(),
        RunSummary::Ready { .. } => {
            format!("Ready to start · {}", crate::train::target_step::kaggle_estimate(s.project.settings.steps))
        }
        RunSummary::Running { step, total } => format!("Step {step} / {total}"),
        RunSummary::Done { checkpoints } => format!("Done · {checkpoints} checkpoints"),
        RunSummary::Failed => "The run stopped. Open it to see why.".into(),
    }
}

fn model_line(s: &TrainData) -> String {
    match &s.project.settings.start_from {
        StartFrom::Base => "Base OmniVoice".into(),
        StartFrom::Adapter { repo_or_dir } => {
            let name = repo_or_dir.rsplit(['/', '\\']).find(|p| !p.is_empty()).unwrap_or(repo_or_dir);
            name.to_string()
        }
    }
}

fn target_line(s: &TrainData) -> &'static str {
    match s.project.target {
        TrainTarget::Kaggle => "Kaggle · T4 ×2 · free",
        TrainTarget::ThisPc => "This PC",
        TrainTarget::RunPod => "RunPod",
    }
}

fn title(label: &str) -> El {
    text(label).font(type_::TITLE).color(style::palette().text)
}

fn secondary(label: &str) -> El {
    text(label).font(type_::LABEL).color(style::palette().text_secondary).truncate()
}

fn tile_key(tile: TrainTile) -> &'static str {
    match tile {
        TrainTile::Data => "train-tile-data",
        TrainTile::Setup => "train-tile-setup",
        TrainTile::Run => "train-tile-run",
    }
}

/// A pressable tile: the outer element has the tile key and the shared motion id, the
/// glass panel inside fills it.
fn tile(tile: TrainTile, panel: El) -> El {
    col()
        .key(tile_key(tile))
        .matched(tile_key(tile))
        .role("button")
        .cursor(rune::Cursor::Pointer)
        .on_press(move |s: &mut App| s.train.open(tile))
        .child(panel.w_full().h_full())
}

fn data_board(s: &TrainData) -> El {
    let p = style::palette();
    let d = data_summary(s);
    let mut pills = row().gap(space::S1).flex_wrap();
    pills = pills.child(style::pill(&format!("{} clips", d.clips), p.text));
    pills = pills.child(style::pill(&format!("{} min", fmt_minutes(d.minutes)), p.text));
    if d.left_out > 0 {
        pills = pills.child(style::pill(&format!("{} left out", d.left_out), p.text_secondary));
    }
    if let Some(b) = d.band_khz {
        pills = pills.child(style::pill(&format!("Band {b} kHz"), p.text_secondary));
    }
    let mut list = col().w_full().gap(space::S1);
    for (name, n) in d.sources.iter().take(BOARD_SOURCES) {
        list = list.child(
            row()
                .w_full()
                .gap(space::S2)
                .px(space::S2)
                .py(6.0)
                .radius(10.0)
                .fill(style::white_fill(0.04))
                .child(text(name.as_str()).font(type_::BODY).color(p.text).truncate().grow(1.0).min_w(0.0))
                .child(text(format!("{n} clips").as_str()).font(type_::LABEL).color(p.text_secondary)),
        );
    }
    if d.sources.len() > BOARD_SOURCES {
        list = list.child(secondary(&format!("+{} more", d.sources.len() - BOARD_SOURCES)));
    }
    if d.sources.is_empty() {
        list = list.child(secondary("Drop clips or a folder here"));
    }
    let body = col()
        .key("train-drop")
        .w_full()
        .h_full()
        .gap(space::S2)
        .padding(space::S3)
        .clip_rounded(14.0)
        .child(title("Data"))
        .child(pills)
        .child(list)
        .on_drop_files(|s: &mut App, files, _| s.train.add_files(files))
        .accepts_extensions(&MEDIA);
    tile(TrainTile::Data, style::glass_blur("glass-train-data", col().child(body)))
}

fn setup_board(s: &TrainData) -> El {
    let p = style::palette();
    let body = col()
        .gap(space::S2)
        .padding(space::S3)
        .child(title("Model and where"))
        .child(secondary("Start from"))
        .child(text(model_line(s).as_str()).font(type_::BODY).color(p.text).truncate())
        .child(secondary("Where"))
        .child(text(target_line(s)).font(type_::BODY).color(p.text).truncate());
    tile(TrainTile::Setup, style::glass_blur("glass-train-setup", body))
}

fn run_board(s: &TrainData) -> El {
    let p = style::palette();
    let r = run_summary(s);
    let mut body = col()
        .gap(space::S2)
        .padding(space::S3)
        .child(title("Run"))
        .child(text(run_line(s, &r).as_str()).font(type_::BODY).color(p.text).wrap());
    if let RunSummary::Running { step, total } = r {
        body = body.child(progress(step as f32 / total.max(1) as f32));
    }
    tile(TrainTile::Run, style::glass_blur("glass-train-run", body))
}

/// An orange progress bar from 0 to 1.
pub fn progress(f: f32) -> El {
    let p = style::palette();
    let f = f.clamp(0.0, 1.0);
    row()
        .w_full()
        .h(6.0)
        .radius(3.0)
        .fill(style::white_fill(0.08))
        .child(el().h_full().w_frac(f).radius(3.0).fill(p.accent))
}

/// The board: Data on the left at full height, Setup and Run on the right.
pub fn board(s: &TrainData) -> El {
    row()
        .key("train-board")
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .gap(space::S3)
        .child(col().grow(1.4).min_w(0.0).h_full().child(data_board(s).grow(1.0)))
        .child(
            col()
                .grow(1.0)
                .min_w(0.0)
                .h_full()
                .gap(space::S3)
                .child(setup_board(s).grow(1.0))
                .child(run_board(s).grow(1.0)),
        )
}

/// A narrow column tile: title and one line of state.
fn column_tile(s: &TrainData, t: TrainTile) -> El {
    let (name, line) = match t {
        TrainTile::Data => {
            let d = data_summary(s);
            ("Data", format!("{} clips · {} min", d.clips, fmt_minutes(d.minutes)))
        }
        TrainTile::Setup => ("Model and where", model_line(s)),
        TrainTile::Run => ("Run", run_line(s, &run_summary(s))),
    };
    let body = col()
        .gap(space::S1)
        .padding(space::S2)
        .child(text(name).font(type_::LABEL).color(style::palette().text))
        .child(secondary(&line));
    tile(t, style::glass(body))
}

/// One tile open on the left, the other two in a column on the right.
pub fn open_layout(s: &App, open: TrainTile, adapters: &[AdapterChoice]) -> El {
    let t = &s.train;
    let name = match open {
        TrainTile::Data => "Data",
        TrainTile::Setup => "Model and where",
        TrainTile::Run => "Run",
    };
    let content = match open {
        TrainTile::Data => crate::train::data_step::view(t),
        TrainTile::Setup => col()
            .w_full()
            .gap(space::S3)
            .child(crate::train::model_step::view(t, adapters))
            .child(crate::train::target_step::view(t, &s.transcribe.engine_state())),
        TrainTile::Run => crate::train::run_step::view(t),
    };
    let header = row()
        .w_full()
        .center_y()
        .gap(space::S2)
        .child(title(name).grow(1.0).min_w(0.0))
        .child(style::button_kind("train-tile-close", "Close", ButtonKind::Quiet, true, |s, _| s.train.close_tile()));
    let panel = col()
        .w_full()
        .h_full()
        .gap(space::S2)
        .padding(space::S3)
        .child(header)
        .child(col().w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).child(content));
    let open_el = col()
        .matched(tile_key(open))
        .grow(1.0)
        .min_w(0.0)
        .h_full()
        .child(style::glass_blur("glass-train-open", panel));
    let mut column = col().w(COLUMN_W).h_full().gap(space::S3);
    for other in [TrainTile::Data, TrainTile::Setup, TrainTile::Run] {
        if other != open {
            column = column.child(column_tile(t, other));
        }
    }
    row().w_full().grow(1.0).min_h(0.0).gap(space::S3).child(open_el).child(column)
}
