//! The Frost look in one place: colors, glass panels, button kinds and motion.
//!
//! Every page takes its colors and panels from here. Do not call
//! `tok::theme(true)` in a page.

use rune::tok::{space, type_};
use rune::{Color, El, Transition, Tx, col, el, oklch, text};

use crate::state::App;

/// The Frost colors. All colors are linear sRGB, as rune stores them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    pub bg_top: Color,
    pub bg_bottom: Color,
    /// The soft light spot at the top left of the page.
    pub glow: Color,
    pub glass: Color,
    pub glass_border: Color,
    /// The thin light line at the top edge of a glass panel.
    pub highlight: Color,
    pub text: Color,
    pub text_secondary: Color,
    pub accent: Color,
    /// Text on an accent fill.
    pub accent_text: Color,
    pub red: Color,
    pub green: Color,
    pub teal: Color,
    pub pink: Color,
}

fn hex(s: &str) -> Color {
    rune::color::hex(s).expect("a valid hex color")
}

/// White with the look of CSS `rgba(255,255,255,css_alpha)` over the page background.
/// CSS blends in sRGB, but rune blends in linear light, where the same alpha looks much
/// brighter. This finds the linear alpha that gives the same result over `bg_bottom`.
fn white(css_alpha: f32) -> Color {
    let to_lin = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    let bg_srgb = 0x10 as f32 / 255.0;
    let target = to_lin(bg_srgb + css_alpha * (1.0 - bg_srgb));
    let bg = to_lin(bg_srgb);
    let a = ((target - bg) / (1.0 - bg)).clamp(0.0, 1.0);
    Color { r: 1.0, g: 1.0, b: 1.0, a }
}

pub fn palette() -> Palette {
    Palette {
        bg_top: hex("#18191d"),
        bg_bottom: hex("#101114"),
        glow: white(0.08),
        glass: white(0.055),
        glass_border: white(0.09),
        highlight: white(0.16),
        text: hex("#e9e9ec"),
        text_secondary: hex("#8d8e96"),
        accent: hex("#ff9f5a"),
        accent_text: hex("#1a1208"),
        red: oklch(0.68, 0.19, 25.0),
        green: oklch(0.76, 0.15, 150.0),
        teal: hex("#45d6e8"),
        pink: hex("#ff5fa2"),
    }
}

pub fn red() -> Color {
    palette().red
}

/// The accent. The old orange marks (low-probability words) use it too.
pub fn orange() -> Color {
    palette().accent
}

pub fn green() -> Color {
    palette().green
}

/// WCAG contrast ratio. Rune colors are linear already, so the luminance needs no
/// gamma step. Alpha is ignored: blend first.
#[cfg(test)]
pub fn contrast(fg: Color, bg: Color) -> f32 {
    let lum = |c: Color| 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    let (a, b) = (lum(fg), lum(bg));
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

/// White fill with the look of CSS `rgba(255,255,255,css_alpha)` on the page.
pub fn white_fill(css_alpha: f32) -> Color {
    white(css_alpha)
}

/// The rune theme with the Frost colors, for code that reads theme fields
/// (`t.card`, `t.text`). Pages call this, never `tok::theme(true)`.
pub fn theme() -> rune::tok::Theme {
    let p = palette();
    let mut t = rune::tok::theme(true);
    t.bg = p.bg_bottom;
    t.card = white(0.05);
    t.card_border = p.glass_border;
    t.text = p.text;
    t.text_secondary = p.text_secondary;
    t.separator = white(0.07);
    t.control_track = white(0.08);
    t.control_off = white(0.12);
    t.hover = white(0.03);
    t.press = white(0.06);
    t
}

/// The page background: a dark vertical gradient with a soft light spot at the top left.
pub fn background(child: El) -> El {
    let p = palette();
    let glow = el().absolute(-160.0, -220.0).size(720.0, 520.0).radius(360.0).fill(p.glow).blur(120.0);
    col().w_full().h_full().gradient(p.bg_top, p.bg_bottom).clip_rounded(0.0).child(glow).child(child)
}

/// A glass panel without blur: glass fill, a light top edge, round corners.
pub fn glass(e: El) -> El {
    let p = palette();
    e.fill(p.glass).border_gradient(1.0, p.highlight, p.glass_border).radius(14.0)
}

/// A glass panel with backdrop blur. Rune blurs only 4 elements per frame, so every
/// blurred panel has a `glass-` key and the tests count them.
pub fn glass_blur(key: &str, e: El) -> El {
    debug_assert!(key.starts_with("glass-"), "a blurred panel needs a glass- key, not {key}");
    glass(e.key(key)).backdrop_blur(18.0)
}

/// How much a button stands out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonKind {
    /// The one main action of a page: orange.
    Main,
    /// A gray glass button.
    Normal,
    /// No fill, gray text.
    Quiet,
}

#[cfg(test)]
thread_local! {
    static MAIN_KEYS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The keys of the main buttons built on this thread since the last call (tests only).
#[cfg(test)]
pub fn take_main_keys() -> Vec<String> {
    MAIN_KEYS.with(|k| std::mem::take(&mut *k.borrow_mut()))
}

/// A button of `kind`. `on` runs on press. A disabled button ignores presses.
pub fn button_kind(key: &str, label: &str, kind: ButtonKind, enabled: bool, on: impl Fn(&mut App, &mut Tx) + 'static) -> El {
    let p = palette();
    #[cfg(test)]
    if kind == ButtonKind::Main {
        MAIN_KEYS.with(|k| k.borrow_mut().push(key.to_string()));
    }
    let (fill, ink) = match kind {
        ButtonKind::Main => (p.accent, p.accent_text),
        ButtonKind::Normal => (white(0.08), p.text),
        ButtonKind::Quiet => (white(0.0), p.text_secondary),
    };
    let b = el()
        .key(key)
        .role("button")
        .label(label)
        .focusable(enabled)
        .disabled(!enabled)
        .px(space::S3)
        .py(space::S2)
        .radius(999.0)
        .fill(fill)
        .opacity(if enabled { 1.0 } else { 0.45 })
        .child(text(label).font(type_::LABEL).color(ink));
    let b = if kind == ButtonKind::Normal { b.border(1.0, p.glass_border) } else { b };
    if enabled { b.on_press_tx(on) } else { b }
}

/// A gray glass button.
pub fn button(key: &str, label: &str, enabled: bool, on: impl Fn(&mut App, &mut Tx) + 'static) -> El {
    button_kind(key, label, ButtonKind::Normal, enabled, on)
}

/// A small rounded label for totals and script differences.
pub fn pill(label: &str, color: Color) -> El {
    el()
        .label(label)
        .min_w(0.0)
        .shrink()
        .px(space::S2)
        .py(2.0)
        .radius(999.0)
        .fill(color.alpha(0.14))
        .border(1.0, color.alpha(0.35))
        .child(text(label).font(type_::LABEL).color(color).truncate().min_w(0.0))
}

/// How a new page enters: it fades in and moves up from 8 points lower.
pub const PAGE_ENTER: (f32, f32) = (0.0, 8.0);

/// How the content of a new page enters. Put it on the content, inside a page wrapper
/// that has `page_exit()`.
pub fn page_transition() -> Transition {
    Transition::offset(PAGE_ENTER.0, PAGE_ENTER.1).combined(Transition::opacity())
}

/// How a page wrapper leaves: at once. An exiting page would still draw its blurred
/// panels, and rune blurs only 4 elements per frame.
pub fn page_exit() -> Transition {
    Transition::none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rune::el;

    #[test]
    fn the_accent_is_warm_orange() {
        assert_eq!(palette().accent, rune::color::hex("#ff9f5a").unwrap());
        assert_eq!(orange(), palette().accent);
    }

    #[test]
    fn body_text_is_readable_on_glass() {
        let p = palette();
        let panel = p.bg_bottom.mix(Color { a: 1.0, ..p.glass }, p.glass.a);
        assert!(contrast(p.text, panel) >= 7.0, "{}", contrast(p.text, panel));
        assert!(contrast(p.text_secondary, panel) >= 4.5, "{}", contrast(p.text_secondary, panel));
    }

    #[test]
    fn main_button_text_is_readable() {
        let p = palette();
        assert!(contrast(p.accent_text, p.accent) >= 4.5, "{}", contrast(p.accent_text, p.accent));
    }

    #[test]
    #[should_panic]
    fn glass_blur_rejects_other_keys() {
        let _ = glass_blur("panel", el());
    }

    #[test]
    fn main_buttons_are_recorded() {
        let _ = take_main_keys();
        let _ = button_kind("k-main", "Go", ButtonKind::Main, true, |_, _| {});
        let _ = button_kind("k-normal", "Other", ButtonKind::Normal, true, |_, _| {});
        assert_eq!(take_main_keys(), vec!["k-main".to_string()]);
    }

    #[test]
    fn a_long_pill_stays_in_a_narrow_row() {
        let view = |_: &(), _: &mut rune::Cx<()>| {
            // The root fills the window, so the narrow row sits inside it.
            col().child(rune::row()
                .key("narrow")
                .w(190.0)
                .gap(space::S2)
                .child(text("Changed").font(type_::LABEL).w(64.0))
                .child(pill("geliştirilebilecekmiş → geliştirilebilecek", palette().red).key("long-pill")))
        };
        let mut rt = rune::Runtime::new((), view, (400.0, 200.0), rune::Env { system_dark: true, scale: 2.0 });
        rt.frame(0.0);
        let rect = |k: &str| rt.elements().into_iter().find(|e| e.key.as_deref() == Some(k)).unwrap().target;
        let (pill, row) = (rect("long-pill"), rect("narrow"));
        assert!(pill[0] + pill[2] <= row[0] + row[2] + 0.5, "pill {pill:?} inside row {row:?}");
    }
}
