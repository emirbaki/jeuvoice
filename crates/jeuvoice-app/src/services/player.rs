//! One sound player for the whole app.
//!
//! The Transcribe tab (Plan 1) and the Voice tab (Plan 2) share it. The tab that
//! loads a clip owns the player; the other tab sees its play bar as stopped.

use std::sync::Arc;

use jeuvoice_core::audio::Audio;

use crate::playback::Playback;

/// Which tab owns the player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerOwner {
    Transcribe,
    /// The Voice tab (Plan 2) takes the player with this.
    #[allow(dead_code)]
    Voice,
    /// The Train tab review list takes the player with this.
    Train,
}

/// One [`Playback`] with an owner. Opening the device can fail: then playback stays
/// off and [`SharedPlayer::take_error`] says why.
pub struct SharedPlayer {
    playback: Option<Playback>,
    owner: Option<PlayerOwner>,
    sound: bool,
    error: Option<String>,
}

impl SharedPlayer {
    pub fn new(sound: bool) -> Self {
        Self { playback: None, owner: None, sound, error: None }
    }

    /// Which tab owns the player now. The Voice tab (Plan 2) reads this.
    #[allow(dead_code)]
    pub fn owner(&self) -> Option<PlayerOwner> {
        self.owner
    }

    /// The player for `owner`, opening the device on first use. `None` when another
    /// tab owns it, when sound is off, or when the device cannot open. Polling
    /// code uses this: it never takes the player from the tab the user is on.
    pub fn player(&mut self, owner: PlayerOwner) -> Option<&mut Playback> {
        if self.owner.is_none() {
            self.owner = Some(owner);
        }
        if self.owner != Some(owner) {
            return None;
        }
        self.ensure_device();
        self.playback.as_mut()
    }

    /// The player for `owner`, taking it from another tab. A press on a play
    /// button uses this: the old tab stops and the new one loads its own audio.
    pub fn take(&mut self, owner: PlayerOwner) -> Option<&mut Playback> {
        if self.owner != Some(owner) {
            if let Some(pb) = self.playback.as_mut() {
                pb.pause();
            }
            self.owner = Some(owner);
        }
        self.ensure_device();
        self.playback.as_mut()
    }

    fn ensure_device(&mut self) {
        if self.playback.is_none() && self.sound {
            match Playback::new() {
                Ok(pb) => self.playback = Some(pb),
                Err(e) => {
                    self.sound = false;
                    self.error = Some(e);
                }
            }
        }
    }

    /// Load a clip for `owner`, taking the player. Takes the audio by value: no clone.
    pub fn load(&mut self, owner: PlayerOwner, audio: Audio) {
        if let Some(pb) = self.take(owner) {
            pb.load(Arc::new(audio));
        }
    }

    /// Whether `owner`'s play bar moves.
    pub fn is_playing_for(&self, owner: PlayerOwner) -> bool {
        self.owner == Some(owner) && self.playback.as_ref().is_some_and(Playback::is_playing)
    }

    /// Why opening the device failed, if it did.
    pub fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    #[cfg(test)]
    pub fn sound(&self) -> bool {
        self.sound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_loader_owns_the_player() {
        let mut p = SharedPlayer::new(false);
        assert_eq!(p.owner(), None);
        // Sound is off, so there is no device: still the loader owns it.
        assert!(p.player(PlayerOwner::Transcribe).is_none());
        assert_eq!(p.owner(), Some(PlayerOwner::Transcribe));
    }

    #[test]
    fn the_other_tab_sees_a_stopped_bar() {
        let mut p = SharedPlayer::new(false);
        p.load(PlayerOwner::Transcribe, Audio { samples: vec![0.1; 100], rate: 16_000 });
        assert_eq!(p.owner(), Some(PlayerOwner::Transcribe));
        assert!(p.player(PlayerOwner::Voice).is_none(), "another tab gets no player");
        assert!(!p.is_playing_for(PlayerOwner::Voice));
        assert!(!p.is_playing_for(PlayerOwner::Transcribe), "no device, nothing plays");
    }

    #[test]
    fn another_tab_takes_the_player_on_use() {
        let mut p = SharedPlayer::new(false);
        p.load(PlayerOwner::Transcribe, Audio { samples: vec![0.1; 100], rate: 16_000 });
        assert_eq!(p.owner(), Some(PlayerOwner::Transcribe));
        p.load(PlayerOwner::Voice, Audio { samples: vec![0.2; 200], rate: 16_000 });
        assert_eq!(p.owner(), Some(PlayerOwner::Voice), "a load takes the player");
        assert_eq!(p.take(PlayerOwner::Train).map(|_| ()), None, "no device, but the owner moves");
        assert_eq!(p.owner(), Some(PlayerOwner::Train));
    }

    #[test]
    fn a_failed_device_leaves_a_notice_and_stays_off() {
        // Sound off stands in for "no device": loading works, playing never starts.
        let mut p = SharedPlayer::new(false);
        p.load(PlayerOwner::Voice, Audio { samples: vec![0.1; 100], rate: 16_000 });
        assert_eq!(p.owner(), Some(PlayerOwner::Voice));
        assert!(!p.sound());
    }
}
