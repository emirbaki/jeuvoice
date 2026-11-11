//! The waveform view: one long wave with the Whisper words under it.
//!
//! A drag on the wave selects a time range, a click moves the playhead, the wheel zooms around
//! the pointer, and a click on a word calls back with its index.

use jeuvoice_core::peaks::Peaks;
use jeuvoice_core::transcript::Word;
use rune::tok::type_;
use rune::{Color, Cursor, Drag, El, LineCap, Phase, Shape, canvas, col, el, oklcha, text};

/// What part of the clip the wave shows, and what the user picked on it. Times are in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveView {
    pub start: f64,
    pub span: f64,
    pub selection: Option<(f64, f64)>,
    pub playhead: f64,
}

/// A note on one word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Missing,
    Added,
    Changed,
    LowProb,
}

/// Height of the wave in points.
const WAVE_H: f32 = 96.0;
/// Height of the row of words under the wave.
const WORDS_H: f32 = 26.0;
/// A press that travels less than this many points before release is a click, not a drag.
const CLICK_SLOP: f32 = 3.0;
/// The closest zoom, in seconds across the whole wave.
const MIN_SPAN: f64 = 2.0;
/// Wheel points that halve the span (a notched mouse wheel step is often 40 to 120 points).
const WHEEL_PER_HALVING: f64 = 240.0;
/// Space kept between two words that would touch.
const WORD_GAP: f32 = 4.0;
/// Width used for the wave before its first layout.
const DEFAULT_W: f32 = 800.0;

use crate::style::{green, orange, red};

/// The wave and the row of words under it. `key` names the wave element and its layout
/// read, so two waves (Transcribe, Voice) can live in one window. `duration` is the
/// clip length in seconds. `lens` finds the [`WaveView`] in the app state, `on_word`
/// hears the index of a clicked word, and `on_word_double` the index of a double-clicked
/// word.
/// `marks` pairs word indexes with marks: a word can have more than one. A `Missing` mark sits
/// on the heard word right after the gap and draws a red bar before it.
pub fn waveform<S: 'static>(
    key: &str,
    v: &WaveView,
    peaks: &Peaks,
    words: &[Word],
    marks: &[(usize, Mark)],
    duration: f64,
    lens: impl Fn(&mut S) -> &mut WaveView + Clone + 'static,
    on_word: impl Fn(&mut S, usize) + Clone + 'static,
    on_word_double: impl Fn(&mut S, usize) + Clone + 'static,
    // Replaces the word row while a word is being edited. The edit line sits
    // directly under the wave, in the same place as the words.
    editor: Option<El>,
) -> El {
    let width = rune::memo::layout_size(key).map_or(DEFAULT_W, |s| s[0]).max(1.0);
    let spans: Vec<(f64, f64)> = words.iter().map(|w| (w.start, w.end)).collect();
    let wave = canvas(wave_shape(v, peaks, words, width))
        .key(key)
        .w_full()
        .h(WAVE_H)
        .fill(crate::style::theme().card)
        .cursor(Cursor::Text)
        .on_drag({
            let lens = lens.clone();
            let on_word = on_word.clone();
            let on_edit = on_word_double.clone();
            let spans = spans.clone();
            move |s: &mut S, d: &mut Drag, tx: &mut rune::Tx| {
                drag(lens(s), d, duration);
                if d.phase == Phase::Up && !d.cancelled() && d.data == 0.0 {
                    let t = time_at(lens(s), d.x, d.w, duration);
                    if let Some(i) = word_at(&spans, t) {
                        on_word(s, i);
                        on_edit(s, i);
                        tx.scroll_into_view(format!("transcript-word-{i}"));
                    }
                }
            }
        })
        .on_wheel(move |s: &mut S, e, _tx| {
            if e.dy == 0.0 {
                return false;
            }
            zoom(lens(s), f64::from(e.x / width), f64::from(e.dy), duration);
            true
        });
    col()
        .w_full()
        .child(wave)
        .child(editor.unwrap_or_else(|| word_row(v, words, marks, width, on_word, on_word_double)))
}

/// Seconds at local point `x` of a wave `w` points wide, inside the clip.
fn time_at(v: &WaveView, x: f32, w: f32, duration: f64) -> f64 {
    (v.start + f64::from(x / w.max(1.0)) * v.span).clamp(0.0, duration)
}

/// One drag gesture on the wave. `Drag::data` is 1 once the press has moved far enough to be a
/// drag, so a press that wanders off and comes back still selects.
fn drag(v: &mut WaveView, d: &mut Drag, duration: f64) {
    let moved = (d.x - d.start_x).hypot(d.y - d.start_y) >= CLICK_SLOP;
    if d.phase == Phase::Move && moved {
        d.data = 1.0;
    }
    if d.phase == Phase::Down || (d.phase == Phase::Up && d.cancelled()) {
        return;
    }
    if d.data == 1.0 {
        let (a, b) = (time_at(v, d.start_x, d.w, duration), time_at(v, d.x, d.w, duration));
        v.selection = Some((a.min(b), a.max(b)));
    } else if d.phase == Phase::Up {
        v.playhead = time_at(v, d.x, d.w, duration);
        v.selection = None;
    }
}

/// Zoom by a wheel delta of `dy` points (positive zooms in), keeping the time at `frac` of the
/// wave's width where it is.
fn zoom(v: &mut WaveView, frac: f64, dy: f64, duration: f64) {
    let pinned = v.start + frac * v.span;
    let span = (v.span * (-dy / WHEEL_PER_HALVING).exp2()).clamp(MIN_SPAN.min(duration), duration);
    v.span = span;
    v.start = (pinned - frac * span).clamp(0.0, (duration - span).max(0.0));
}

/// The alpha of the wave after the play position.
pub const UNPLAYED_ALPHA: f32 = 0.30;
/// Crossing rates at or below this are pure orange; at or above `TEAL_RATE`, pure teal.
const ORANGE_RATE: f32 = 0.02;
const TEAL_RATE: f32 = 0.25;
/// A word end gets a gap only when the word is at least this wide on screen.
const MIN_GAP_WORD_W: f32 = 12.0;
const GAP_HALF_W: f32 = 1.5;

/// The color for a crossing rate: orange (low sounds) to pink to teal (hissing sounds).
#[cfg(test)]
pub fn pitch_color(rate: f32) -> Color {
    pitch_color_in(&crate::style::palette(), rate)
}

fn pitch_color_in(p: &crate::style::Palette, rate: f32) -> Color {
    let t = ((rate - ORANGE_RATE) / (TEAL_RATE - ORANGE_RATE)).clamp(0.0, 1.0);
    let c = if t <= 0.0 {
        p.accent
    } else if t >= 1.0 {
        p.teal
    } else if t < 0.5 {
        p.accent.mix(p.pink, t * 2.0)
    } else {
        p.pink.mix(p.teal, (t - 0.5) * 2.0)
    };
    Color { a: 1.0, ..c }
}

/// The pitch color, bright before the play position and dim after it.
#[cfg(test)]
fn column_color(col: usize, playhead_col: f32, rate: f32) -> Color {
    column_color_in(&crate::style::palette(), col, playhead_col, rate)
}

fn column_color_in(p: &crate::style::Palette, col: usize, playhead_col: f32, rate: f32) -> Color {
    let a = if (col as f32) < playhead_col { 1.0 } else { UNPLAYED_ALPHA };
    Color { a, ..pitch_color_in(p, rate) }
}

/// Vertical min-to-max strokes, one per point column, colored by pitch, with gaps at word
/// ends and the selection and playhead on top.
fn wave_shape(v: &WaveView, peaks: &Peaks, words: &[Word], width: f32) -> Shape {
    let cols = width.round().max(1.0) as usize;
    let mid = WAVE_H * 0.5;
    let x_of = |t: f64| ((t - v.start) / v.span * cols as f64) as f32;
    let playhead_col = x_of(v.playhead);
    let pal = crate::style::palette();
    // Words are in time order, so the gap positions are sorted and one cursor walks them.
    let gaps: Vec<f32> = words
        .iter()
        .filter(|w| w.end > v.start && w.end < v.start + v.span && x_of(w.end) - x_of(w.start) >= MIN_GAP_WORD_W)
        .map(|w| x_of(w.end))
        .collect();
    let rates = peaks.range_zcr(v.start, v.start + v.span, cols);
    let mut shape = Shape::new().line([0.0, mid], [cols as f32, mid], 1.0, LineCap::Butt, oklcha(1.0, 0.0, 0.0, 0.08));
    let mut next_gap = 0;
    for (i, (lo, hi)) in peaks.range(v.start, v.start + v.span, cols).into_iter().enumerate() {
        let x = i as f32 + 0.5;
        while next_gap < gaps.len() && gaps[next_gap] + GAP_HALF_W <= x {
            next_gap += 1;
        }
        if gaps.get(next_gap).is_some_and(|g| (g - x).abs() < GAP_HALF_W) {
            continue;
        }
        let top = mid - hi.clamp(-1.0, 1.0) * (mid - 2.0);
        let bottom = (mid - lo.clamp(-1.0, 1.0) * (mid - 2.0)).max(top + 1.0);
        let rate = rates.get(i).copied().unwrap_or(0.0);
        shape = shape.line([x, top], [x, bottom], 1.0, LineCap::Butt, column_color_in(&pal, i, playhead_col, rate));
    }
    if let Some((a, b)) = v.selection {
        let (a, b) = (x_of(a).max(0.0), x_of(b).min(cols as f32));
        if b > a {
            let tint = Color { a: 0.22, ..pal.accent };
            shape = shape.fill(vec![[a, 0.0], [b, 0.0], [b, WAVE_H], [a, WAVE_H]], tint);
        }
    }
    if (0.0..=cols as f32).contains(&playhead_col) {
        let p = playhead_col;
        // A soft glow under the playhead line.
        shape = shape.line([p, 0.0], [p, WAVE_H], 5.0, LineCap::Butt, oklcha(1.0, 0.0, 0.0, 0.25));
        shape = shape.line([p, 0.0], [p, WAVE_H], 1.5, LineCap::Butt, pal.text);
    }
    shape.view_box(cols as f32, WAVE_H)
}

/// The word at `t`, or the next one, or the last one when `t` is past the end.
pub(crate) fn word_at(spans: &[(f64, f64)], t: f64) -> Option<usize> {
    if let Some(i) = spans.iter().position(|&(a, b)| t >= a && t <= b) {
        return Some(i);
    }
    if let Some(i) = spans.iter().position(|&(a, _)| a >= t) {
        return Some(i);
    }
    spans.len().checked_sub(1)
}

/// The words in the visible range, each at the x of its start time, or just right of the word
/// before when they would overlap.
fn word_row<S: 'static>(
    v: &WaveView,
    words: &[Word],
    marks: &[(usize, Mark)],
    width: f32,
    on_word: impl Fn(&mut S, usize) + Clone + 'static,
    on_word_double: impl Fn(&mut S, usize) + Clone + 'static,
) -> El {
    let t = crate::style::theme();
    let end = v.start + v.span;
    let spans: Vec<(f64, f64)> = words.iter().map(|w| (w.start, w.end)).collect();
    let (start, span) = (v.start, v.span);
    let on_row_click = {
        let seek = on_word.clone();
        let edit = on_word_double.clone();
        move |s: &mut S, d: &mut Drag, tx: &mut rune::Tx| {
            if d.phase != Phase::Up || d.cancelled() || d.data > 0.0 {
                return;
            }
            let t = start + f64::from((d.x / d.w.max(1.0)).clamp(0.0, 1.0)) * span;
            if let Some(i) = word_at(&spans, t) {
                seek(s, i);
                edit(s, i);
                tx.scroll_into_view(format!("transcript-word-{i}"));
            }
        }
    };
    let mut row = el()
        .key("wave-words")
        .w_full()
        .h(WORDS_H)
        .clip_rounded(0.0)
        .cursor(Cursor::Text)
        .on_drag(on_row_click);
    let mut free_x = f32::NEG_INFINITY;
    // Words are in time order, so the first visible word is the first that ends after `start`.
    let first = words.partition_point(|w| w.end <= v.start);
    for (i, w) in words.iter().enumerate().skip(first) {
        if w.start >= end {
            break;
        }
        let has = |m: Mark| marks.iter().any(|&(j, k)| j == i && k == m);
        let key = format!("word-{i}");
        let target = (((w.start - v.start) / v.span) as f32 * width).max(0.0);
        let x = target.max(free_x);
        // The width it had last layout; a guess the first time it shows.
        let w_pts = rune::memo::layout_size(&key).map_or(w.text.chars().count() as f32 * 7.0 + 4.0, |s| s[0]);
        free_x = x + w_pts + WORD_GAP;
        let color = if has(Mark::Changed) {
            red()
        } else if has(Mark::Added) {
            green()
        } else {
            t.text
        };
        let (on_word, on_word_double) = (on_word.clone(), on_word_double.clone());
        let mut word = col()
            .key(key)
            .absolute(x, 2.0)
            .px(2.0)
            .gap(1.0)
            .radius(4.0)
            .cursor(Cursor::Pointer)
            .hover(crate::style::hover_fill())
            .label(w.text.as_str())
            .on_press(move |s: &mut S| on_word(s, i))
            .on_double_press(move |s: &mut S| on_word_double(s, i))
            .child(text(w.text.as_str()).font(type_::BODY).color(color));
        if has(Mark::LowProb) {
            word = word.child(el().w_full().h(2.0).radius(1.0).fill(orange()));
        }
        if has(Mark::Missing) {
            row = row.child(missing_bar(format!("missing-{i}"), x - 3.0));
        }
        row = row.child(word);
    }
    // A gap after the last word (the speaker stopped early): the bar goes right after that word.
    let trailing = marks.iter().any(|&(j, k)| j == words.len() && k == Mark::Missing);
    if let Some(last) = words.last()
        && trailing
        && last.end > v.start
        && last.end <= end
    {
        let x = if free_x.is_finite() { free_x - WORD_GAP + 1.0 } else { 0.0 };
        row = row.child(missing_bar("missing-end".into(), x));
    }
    row
}

/// The red bar that marks a gap where a script word was not heard, with its left edge at `x`.
fn missing_bar(key: String, x: f32) -> El {
    el().key(key).absolute(x.max(0.0), 3.0).size(3.0, 16.0).radius(1.5).fill(red())
}

#[cfg(test)]
mod tests {
    use rune::Color;
    use super::{Mark, WaveView, waveform};
    use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks};
    use jeuvoice_core::transcript::Word;
    use rune::{Cx, El, Env, Runtime, WheelEvent, col};

    const DURATION: f64 = 10.0;

    struct Fixture {
        view: WaveView,
        peaks: Peaks,
        words: Vec<Word>,
        marks: Vec<(usize, Mark)>,
        picked: Option<usize>,
        doubled: Option<usize>,
    }

    fn word(text: &str, start: f64, end: f64, prob: f32) -> Word {
        Word { text: text.into(), start, end, prob }
    }

    /// 10 s of a tone that swells and fades in syllable-sized bursts.
    fn synthetic_peaks() -> Peaks {
        let n = (DURATION * PEAKS_PER_SECOND as f64) as usize;
        let level = |k: usize| {
            let t = k as f32 / PEAKS_PER_SECOND as f32;
            let burst = (t * 3.1).sin().abs() * (0.4 + 0.6 * (t * 0.7).sin().abs());
            (burst * (1.0 + 0.15 * (t * 53.0).sin())).min(1.0)
        };
        Peaks {
            per_second: PEAKS_PER_SECOND,
            min: (0..n).map(|k| -level(k)).collect(),
            max: (0..n).map(level).collect(),
            zcr: (0..n).map(|k| (k as f32 * 0.05).sin().abs() * 0.3).collect(),
        }
    }

    fn fixture() -> Fixture {
        Fixture {
            view: WaveView { start: 0.0, span: DURATION, selection: None, playhead: 0.0 },
            peaks: synthetic_peaks(),
            words: vec![
                word("Merhaba,", 0.4, 0.9, 0.95),
                word("bugün", 1.1, 1.5, 0.9),
                word("size", 1.6, 1.9, 0.4),
                word("küçük", 2.6, 3.0, 0.85),
                word("bir", 3.05, 3.2, 0.9),
                word("hikâye", 3.25, 3.8, 0.8),
                word("anlatacağım.", 5.0, 6.1, 0.9),
                word("Dinleyin.", 7.2, 7.9, 0.3),
            ],
            marks: vec![(1, Mark::Changed), (2, Mark::LowProb), (4, Mark::Added), (6, Mark::Missing), (7, Mark::LowProb), (7, Mark::Changed)],
            picked: None,
            doubled: None,
        }
    }

    fn view(s: &Fixture, _: &mut Cx<Fixture>) -> El {
        let t = crate::style::theme();
        col().w_full().h_full().fill(t.bg).child(waveform(
            "wave",
            &s.view,
            &s.peaks,
            &s.words,
            &s.marks,
            DURATION,
            |s: &mut Fixture| &mut s.view,
            |s: &mut Fixture, i| s.picked = Some(i),
            |s: &mut Fixture, i| s.doubled = Some(i),
            None,
        ))
    }

    fn run() -> Runtime<Fixture> {
        let mut rt = Runtime::new(fixture(), view, (1000.0, 200.0), Env { system_dark: true, scale: 2.0 });
        rt.frame(0.0);
        rt.frame(1.0 / 60.0);
        rt
    }

    /// The wave's middle row, in window points.
    fn wave_y(rt: &Runtime<Fixture>) -> f32 {
        let (rect, _) = rt.presented("wave").expect("the wave is on screen");
        assert_eq!(rect[2], 1000.0, "the wave fills the window's width");
        rect[1] + rect[3] * 0.5
    }

    /// The old drawing: the same strokes in one color, without rates or gaps.
    fn old_wave_shape(v: &WaveView, peaks: &Peaks, width: f32) -> rune::Shape {
        let cols = width.round().max(1.0) as usize;
        let mid = super::WAVE_H * 0.5;
        let ink = rune::tok::accent(0, true);
        let mut shape = rune::Shape::new();
        for (i, (lo, hi)) in peaks.range(v.start, v.start + v.span, cols).into_iter().enumerate() {
            let x = i as f32 + 0.5;
            let top = mid - hi.clamp(-1.0, 1.0) * (mid - 2.0);
            let bottom = (mid - lo.clamp(-1.0, 1.0) * (mid - 2.0)).max(top + 1.0);
            shape = shape.line([x, top], [x, bottom], 1.0, rune::LineCap::Butt, ink);
        }
        shape.view_box(cols as f32, super::WAVE_H)
    }

    /// `scripts\cargo.cmd test -p jeuvoice-app --release wave_shape_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn wave_shape_speed() {
        let n = 600 * PEAKS_PER_SECOND as usize;
        let peaks = Peaks {
            per_second: PEAKS_PER_SECOND,
            min: (0..n).map(|k| -((k as f32 * 0.01).sin().abs())).collect(),
            max: (0..n).map(|k| (k as f32 * 0.01).sin().abs()).collect(),
            zcr: (0..n).map(|k| (k as f32 * 0.003).sin().abs() * 0.3).collect(),
        };
        let words: Vec<Word> = (0..1200).map(|i| word("söz", i as f64 * 0.5, i as f64 * 0.5 + 0.4, 0.9)).collect();
        let v = WaveView { start: 0.0, span: 600.0, selection: None, playhead: 300.0 };
        let time = |f: &dyn Fn()| {
            let t = std::time::Instant::now();
            for _ in 0..100 {
                f();
            }
            t.elapsed().as_secs_f64() * 1000.0 / 100.0
        };
        let before = time(&|| drop(old_wave_shape(&v, &peaks, 1600.0)));
        let after = time(&|| drop(super::wave_shape(&v, &peaks, &words, 1600.0)));
        println!("wave_shape 10 min, 1600 columns: before {before:.3} ms, after {after:.3} ms");
        assert!(after < 2.0 * before, "after {after:.3} ms is more than twice before {before:.3} ms");
    }

    #[test]
    fn pitch_color_runs_orange_to_teal() {
        let p = crate::style::palette();
        assert_eq!(super::pitch_color(0.0), Color { a: 1.0, ..p.accent });
        assert_eq!(super::pitch_color(0.5), Color { a: 1.0, ..p.teal });
        let mid = super::pitch_color(0.1);
        assert_ne!(mid, Color { a: 1.0, ..p.accent });
        assert_ne!(mid, Color { a: 1.0, ..p.teal });
    }

    #[test]
    fn columns_after_the_playhead_are_dim() {
        assert_eq!(super::column_color(0, 50.0, 0.1).a, 1.0);
        assert_eq!(super::column_color(99, 50.0, 0.1).a, super::UNPLAYED_ALPHA);
    }

    #[test]
    fn drag_selects_a_time_range() {
        let mut rt = run();
        let y = wave_y(&rt);
        rt.pointer_down(100.0, y);
        rt.frame(1.0 / 60.0);
        rt.pointer_move(200.0, y);
        rt.frame(1.0 / 60.0);
        rt.pointer_move(300.0, y);
        rt.frame(1.0 / 60.0);
        rt.pointer_up(300.0, y);
        rt.frame(1.0 / 60.0);
        let (a, b) = rt.state.view.selection.expect("a drag selects");
        assert!((a - 1.0).abs() < 0.05 && (b - 3.0).abs() < 0.05, "selection {a}..{b}");
    }

    #[test]
    fn drag_to_the_left_orders_and_clamps_the_selection() {
        let mut rt = run();
        let y = wave_y(&rt);
        rt.pointer_down(300.0, y);
        rt.frame(1.0 / 60.0);
        rt.pointer_move(-50.0, y);
        rt.frame(1.0 / 60.0);
        rt.pointer_up(-50.0, y);
        rt.frame(1.0 / 60.0);
        let (a, b) = rt.state.view.selection.expect("a drag selects");
        assert!(a == 0.0 && (b - 3.0).abs() < 0.05, "selection {a}..{b}");
    }

    #[test]
    fn click_on_wave_moves_playhead_and_clears_selection() {
        let mut rt = run();
        rt.update(|s, _| s.view.selection = Some((1.0, 2.0)));
        rt.frame(1.0 / 60.0);
        let y = wave_y(&rt);
        rt.pointer_down(500.0, y);
        rt.frame(1.0 / 60.0);
        rt.pointer_move(502.0, y);
        rt.frame(1.0 / 60.0);
        rt.pointer_up(502.0, y);
        rt.frame(1.0 / 60.0);
        assert!((rt.state.view.playhead - 5.0).abs() < 0.05, "playhead {}", rt.state.view.playhead);
        assert_eq!(rt.state.view.selection, None);
    }

    #[test]
    fn click_on_word_calls_handler() {
        let mut rt = run();
        let (x, y) = rt.find("word-2").expect("word 2 is on screen");
        rt.pointer_down(x, y);
        rt.pointer_up(x, y);
        rt.frame(1.0 / 60.0);
        assert_eq!(rt.state.picked, Some(2));
    }

    #[test]
    fn double_click_on_word_calls_handler() {
        let mut rt = run();
        let (x, y) = rt.find("word-3").expect("word 3 is on screen");
        for _ in 0..2 {
            rt.pointer_down(x, y);
            rt.frame(1.0 / 120.0);
            rt.pointer_up(x, y);
            rt.frame(1.0 / 120.0);
        }
        assert_eq!((rt.state.picked, rt.state.doubled), (Some(3), Some(3)));
    }

    #[test]
    fn words_sit_under_their_sound_and_do_not_overlap() {
        let rt = run();
        let rect = |i: usize| rt.presented(&format!("word-{i}")).unwrap().0;
        // "Merhaba," starts at 0.4 s: 40 points into a 1000-point, 10 s wave.
        assert!((rect(0)[0] - 40.0).abs() < 1.0, "word 0 at {}", rect(0)[0]);
        // "küçük", "bir" and "hikâye" start 5 to 20 points apart, closer than they are wide.
        for i in 3..5 {
            let (a, b) = (rect(i), rect(i + 1));
            assert!(b[0] >= a[0] + a[2], "word {} overlaps word {}", i + 1, i);
        }
        // "anlatacağım." has room, so it stays at its start time.
        assert!((rect(6)[0] - 500.0).abs() < 1.0, "word 6 at {}", rect(6)[0]);
    }

    #[test]
    fn missing_bar_sits_left_of_its_word() {
        let rt = run();
        let bar = rt.presented("missing-6").expect("word 6 has a Missing bar").0;
        let word = rt.presented("word-6").unwrap().0;
        assert!(bar[0] + bar[2] <= word[0] + 0.5 && bar[0] >= word[0] - 6.0, "bar {bar:?} word {word:?}");
    }

    #[test]
    fn trailing_missing_bar_sits_right_of_the_last_word() {
        let mut rt = run();
        rt.update(|s, _| s.marks.push((s.words.len(), Mark::Missing)));
        rt.frame(1.0 / 60.0);
        rt.frame(1.0 / 60.0);
        assert!(rt.find("missing-end").is_some());
        let bar = rt.presented("missing-end").unwrap().0;
        let last = rt.presented("word-7").unwrap().0;
        assert!(bar[0] >= last[0] + last[2] && bar[0] <= last[0] + last[2] + 6.0, "bar {bar:?} last word {last:?}");
    }

    #[test]
    fn words_outside_the_visible_range_are_not_built() {
        let mut rt = run();
        rt.update(|s, _| (s.view.start, s.view.span) = (4.0, 2.0));
        rt.frame(1.0 / 60.0);
        rt.frame(1.0 / 60.0);
        assert!(rt.find("word-0").is_none());
        assert!(rt.find("word-7").is_none());
        assert!(rt.find("word-6").is_some());
        rt.update(|s, _| s.marks.push((s.words.len(), Mark::Missing)));
        rt.frame(1.0 / 60.0);
        assert!(rt.find("missing-end").is_none(), "the last word ends outside the view");
    }

    #[test]
    fn wheel_zooms_around_pointer() {
        let mut rt = run();
        let y = wave_y(&rt);
        let x = 300.0;
        let at = |v: &WaveView| v.start + f64::from(x) / 1000.0 * v.span;
        let before = at(&rt.state.view);
        rt.wheel_event(WheelEvent::new(0.0, 120.0).at(x, y));
        rt.frame(1.0 / 60.0);
        let v = &rt.state.view;
        assert!(v.span < DURATION, "the wheel zooms in, span {}", v.span);
        assert!((at(v) - before).abs() < 0.05, "time under the pointer moved from {before} to {}", at(v));
    }

    #[test]
    fn zoom_stays_between_two_seconds_and_the_full_clip() {
        let mut rt = run();
        let y = wave_y(&rt);
        for _ in 0..40 {
            rt.wheel_event(WheelEvent::new(0.0, 120.0).at(900.0, y));
            rt.frame(1.0 / 60.0);
        }
        let v = &rt.state.view;
        assert_eq!(v.span, 2.0);
        assert!(v.start >= 0.0 && v.start + v.span <= DURATION, "view {}..{}", v.start, v.start + v.span);
        for _ in 0..40 {
            rt.wheel_event(WheelEvent::new(0.0, -120.0).at(100.0, y));
            rt.frame(1.0 / 60.0);
        }
        let v = &rt.state.view;
        assert_eq!((v.start, v.span), (0.0, DURATION));
    }

    /// Renders the fixture to `target/wave.png` to look at:
    /// `scripts\cargo.cmd test -p jeuvoice-app snap_wave -- --ignored`.
    #[test]
    #[ignore]
    fn snap_wave() {
        fn init() -> Fixture {
            let mut f = fixture();
            f.view.selection = Some((2.4, 4.1));
            f.view.playhead = 5.5;
            f
        }
        let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/wave.png");
        let spec = rune::AppSpec::new("Wave", (1000.0, 200.0), init, view);
        rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
        println!("wrote {out}");
    }
}
