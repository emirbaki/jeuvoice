//! Embeds the JeuVoice icon in the Windows exe, so Explorer, the taskbar
//! shortcut and the window share one mark. The icon is drawn by
//! `packaging/windows/make-icon.py`; cargo-packager uses the same file for the
//! installer. Other platforms get their icon from the app bundle (Plan 4).

fn main() {
    println!("cargo:rerun-if-changed=../../packaging/windows/icon.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../packaging/windows/icon.ico");
        res.compile().expect(
            "the app icon needs rc.exe from the Visual Studio C++ tools; run scripts\\cargo.cmd",
        );
    }
}
