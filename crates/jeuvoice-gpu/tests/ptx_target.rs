// The release runs on Turing (sm_75), Ampere and Ada cards, and Gpu::open accepts 7.5.
// The driver compiles PTX only for its own target or newer cards, so the PTX must be sm_75.

#[test]
fn ptx_targets_turing() {
    let ptx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/oxide-ptx/jeuvoice_gpu.ptx");
    let text = std::fs::read_to_string(&ptx).expect("read the built PTX (scripts\\cargo.cmd oxide sets the folder)");
    let target = text.lines().find(|l| l.starts_with(".target")).expect(".target line");
    assert_eq!(target.trim(), ".target sm_75", "the PTX must run on Turing cards");
}
