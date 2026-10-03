use rverdict_engine::{BackendChoice, select};

/// CI sets `RVERDICT_EXPECT_BACKEND` on runners that have a given backend
/// (Mesa's lavapipe on Linux, Metal on macOS). Selecting it must pass the
/// self-test, which runs a tiny model and compares it with the CPU, instead of
/// falling back.
#[test]
fn the_expected_backend_passes_its_self_test() {
    let Ok(expected) = std::env::var("RVERDICT_EXPECT_BACKEND") else {
        return;
    };
    let choice: BackendChoice = expected.parse().unwrap();
    let selected = select(choice);
    assert!(
        selected.skipped.is_empty(),
        "skipped: {:?}",
        selected.skipped
    );
    assert!(
        !selected.name.starts_with("cpu"),
        "fell back to {}",
        selected.name
    );
}

#[test]
fn cpu_is_always_available() {
    assert!(select(BackendChoice::Cpu).name.starts_with("cpu"));
}

/// Hosts share one engine across threads (`spawn_blocking`, a server pool).
#[test]
fn engine_can_be_shared_across_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<rverdict_engine::Engine>();
}
