use jeuvoice_core::queue::{spawn_worker, ItemState, Job, Queue, Update};
use jeuvoice_core::transcript::Transcript;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

fn recv(updates: &Receiver<(usize, Update)>) -> (usize, Update) {
    updates
        .recv_timeout(Duration::from_secs(5))
        .expect("no update arrived in time")
}

fn paths(names: &[&str]) -> Vec<PathBuf> {
    names.iter().map(PathBuf::from).collect()
}

#[test]
fn items_run_in_order_and_a_failure_does_not_stop_the_batch() {
    let mut q = Queue::default();
    q.add(&paths(&["a.mp4", "b.mp4", "c.mp4"]));
    let (job_tx, job_rx) = channel::<(usize, Job)>();
    let (up_tx, up_rx) = channel();
    let worker = spawn_worker(job_rx, up_tx, || {});

    for _ in 0..3 {
        let i = q.take_next().expect("an item is waiting");
        let job: Job = if i == 1 {
            Box::new(|_| Err("The file b.mp4 has no sound track.".to_string()))
        } else {
            Box::new(|_| Ok(Transcript::default()))
        };
        job_tx.send((i, job)).unwrap();
    }
    let mut finished = Vec::new();
    while finished.len() < 3 {
        let (i, update) = recv(&up_rx);
        if !matches!(update, Update::Progress(_)) {
            finished.push(i);
        }
        q.apply(i, update);
    }
    assert_eq!(finished, vec![0, 1, 2]);
    assert_eq!(q.items[0].1, ItemState::Done(Transcript::default()));
    assert_eq!(
        q.items[1].1,
        ItemState::Failed("The file b.mp4 has no sound track.".to_string())
    );
    assert_eq!(q.items[2].1, ItemState::Done(Transcript::default()));
    assert_eq!(q.next_waiting(), None);
    drop(job_tx);
    worker.join().unwrap();
}

#[test]
fn adding_the_same_file_twice_keeps_one() {
    let mut q = Queue::default();
    q.add(&paths(&["a.mp4", "b.mp4"]));
    q.add(&paths(&["a.mp4", "c.mp4", "c.mp4"]));
    assert_eq!(q.items.len(), 3);
    assert_eq!(q.next_waiting(), Some(0));
}

#[test]
fn progress_updates_reach_the_item() {
    let mut q = Queue::default();
    q.add(&paths(&["a.mp4"]));
    let (job_tx, job_rx) = channel::<(usize, Job)>();
    let (up_tx, up_rx) = channel();
    let worker = spawn_worker(job_rx, up_tx, || {});

    let i = q.take_next().unwrap();
    assert_eq!(q.items[i].1, ItemState::Running(0.0));
    job_tx
        .send((
            i,
            Box::new(|progress: &mut dyn FnMut(f32)| {
                progress(0.5);
                Ok(Transcript::default())
            }),
        ))
        .unwrap();

    let (j, update) = recv(&up_rx);
    assert_eq!((j, &update), (0, &Update::Progress(0.5)));
    q.apply(j, update);
    assert_eq!(q.items[0].1, ItemState::Running(0.5));
    let (j, update) = recv(&up_rx);
    q.apply(j, update);
    assert_eq!(q.items[0].1, ItemState::Done(Transcript::default()));
    drop(job_tx);
    worker.join().unwrap();
}

#[test]
fn a_panicking_job_fails_and_the_worker_continues() {
    let (job_tx, job_rx) = channel::<(usize, Job)>();
    let (up_tx, up_rx) = channel();
    let worker = spawn_worker(job_rx, up_tx, || {});
    job_tx.send((0, Box::new(|_| panic!("boom")))).unwrap();
    job_tx
        .send((1, Box::new(|_| Ok(Transcript::default()))))
        .unwrap();
    assert_eq!(
        recv(&up_rx),
        (
            0,
            Update::Failed("Transcription stopped because of an internal error.".to_string())
        )
    );
    assert_eq!(recv(&up_rx), (1, Update::Done(Transcript::default())));
    drop(job_tx);
    worker.join().unwrap();
}

#[test]
fn apply_out_of_range_does_nothing() {
    let mut q = Queue::default();
    q.apply(5, Update::Progress(0.1));
    assert!(q.items.is_empty());
}
