//! App state: which tab is open.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Voice,
    Transcribe,
    Train,
    Settings,
}

pub struct App {
    pub tab: Tab,
}

impl Default for App {
    fn default() -> Self {
        Self { tab: Tab::Transcribe }
    }
}
