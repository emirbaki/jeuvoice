//! The waveform view: one long wave with the Whisper words under it.
//!
//! A drag on the wave selects a time range, a click moves the playhead, the wheel zooms around
//! the pointer, and a click on a word calls back with its index.

use jeuvoice_core::peaks::Peaks;
use jeuvoice_core::transcript::Word;
use rune::tok::{self, type_};
use rune::{Color, Cursor, Drag, El, LineCap, Phase, Shape, canvas, col, el, oklch, oklcha, text};

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

fn red() -> Color {
    oklch(0.66, 0.2, 25.0)
}
fn green() -> Color {
    oklch(0.74, 0.17, 150.0)
}
fn orange() -> Color {
    oklch(0.78, 0.16, 65.0)
}

/// The wave and the row of words under it. `duration` is the clip length in seconds. `lens`
/// finds the [`WaveView`] in the app state, and `on_word` hears the index of a clicked word.
/// `marks` pairs word indexes with marks: a word can have more than one. A `Missing` mark sits
/// on the heard word right after the gap and draws a red bar before it.
pub fn waveform<S: 'static>(
    v: &WaveView,
    peaks: &Peaks,
    words: &[Word],
    marks: &[(usize, Mark)],
    duration: f64,
    lens: impl Fn(&mut S) -> &mut WaveView + Clone + 'static,
    on_word: impl Fn(&mut S, usize) + Clone + 'static,
) -> El {
    let width = rune::memo::layout_size("wave").map_or(DEFAULT_W, |s| s[0]).max(1.0);
    let wave = canvas(wave_shape(v, peaks, width))
        .key("wave")
        .w_full()
        .h(WAVE_H)
        .fill(tok::theme(true).card)
        .cursor(Cursor::Text)
        .on_drag({
            let lens = lens.clone();
            move |s: &mut S, d: &mut Drag, _tx| drag(lens(s), d, duration)
        })
        .on_wheel(move |s: &mut S, e, _tx| {
            if e.dy == 0.0 {
                return false;
            }
            zoom(lens(s), f64::from(e.x / width), f64::from(e.dy), duration);
            true
        });
    col().w_full().child(wave).child(word_row(v, words, marks, width, on_word))
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

/// Vertical min-to-max strokes, one per point column, with the selection and playhead on top.
fn wave_shape(v: &WaveView, peaks: &Peaks, width: f32) -> Shape {
    let cols = width.round().max(1.0) as usize;
    let mid = WAVE_H * 0.5;
    let ink = tok::accent(0, true);
    let mut shape = Shape::new().line([0.0, mid], [cols as f32, mid], 1.0, LineCap::Butt, oklcha(1.0, 0.0, 0.0, 0.08));
    for (i, (lo, hi)) in peaks.range(v.start, v.start + v.span, cols).into_iter().enumerate() {
        let x = i as f32 + 0.5;
        let top = mid - hi.clamp(-1.0, 1.0) * (mid - 2.0);
        let bottom = (mid - lo.clamp(-1.0, 1.0) * (mid - 2.0)).max(top + 1.0);
        shape = shape.line([x, top], [x, bottom], 1.0, LineCap::Butt, ink);
    }
    let x_of = |t: f64| ((t - v.start) / v.span * cols as f64) as f32;
    if let Some((a, b)) = v.selection {
        let (a, b) = (x_of(a).max(0.0), x_of(b).min(cols as f32));
        if b > a {
            let tint = Color { a: 0.28, ..tok::accent(0, true) };
            shape = shape.fill(vec![[a, 0.0], [b, 0.0], [b, WAVE_H], [a, WAVE_H]], tint);
        }
    }
    let p = x_of(v.playhead);
    if (0.0..=cols as f32).contains(&p) {
        shape = shape.line([p, 0.0], [p, WAVE_H], 1.5, LineCap::Butt, tok::theme(true).text);
    }
    shape.view_box(cols as f32, WAVE_H)
}

/// The words in the visible range, each at the x of its start time, or just right of the word
/// before when they would overlap.
fn word_row<S: 'static>(
    v: &WaveView,
    words: &[Word],
    marks: &[(usize, Mark)],
    width: f32,
    on_word: impl Fn(&mut S, usize) + Clone + 'static,
) -> El {
    let t = tok::theme(true);
    let end = v.start + v.span;
    let mut row = el().w_full().h(WORDS_H).clip_rounded(0.0);
    let mut free_x = f32::NEG_INFINITY;
    for (i, w) in words.iter().enumerate() {
        if w.end <= v.start || w.start >= end {
            continue;
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
        let on_word = on_word.clone();
        let mut word = col()
            .key(key)
            .absolute(x, 2.0)
            .px(2.0)
            .gap(1.0)
            .cursor(Cursor::Pointer)
            .label(w.text.as_str())
            .on_press(move |s: &mut S| on_word(s, i))
            .child(text(w.text.as_str()).font(type_::BODY).color(color));
        if has(Mark::LowProb) {
            word = word.child(el().w_full().h(2.0).radius(1.0).fill(orange()));
        }
        if has(Mark::Missing) {
            row = row.child(el().absolute((x - 3.0).max(0.0), 3.0).size(3.0, 16.0).radius(1.5).fill(red()));
        }
        row = row.child(word);
    }
    row
}

#[cfg(test)]
mod tests {
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
        }
    }

    fn view(s: &Fixture, _: &mut Cx<Fixture>) -> El {
        let t = rune::tok::theme(true);
        col().w_full().h_full().fill(t.bg).child(waveform(
            &s.view,
            &s.peaks,
            &s.words,
            &s.marks,
            DURATION,
            |s: &mut Fixture| &mut s.view,
            |s: &mut Fixture, i| s.picked = Some(i),
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
    fn words_outside_the_visible_range_are_not_built() {
        let mut rt = run();
        rt.update(|s, _| (s.view.start, s.view.span) = (4.0, 2.0));
        rt.frame(1.0 / 60.0);
        rt.frame(1.0 / 60.0);
        assert!(rt.find("word-0").is_none());
        assert!(rt.find("word-7").is_none());
        assert!(rt.find("word-6").is_some());
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
