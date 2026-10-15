use sha2::{Digest, Sha256};
/// All source files defining generation, execution, verification and CLI behavior.
pub fn hash() -> String {
    let mut sha = Sha256::new();
    sha.update(crate::MODEL.as_bytes());
    #[cfg(feature = "compact")]
    sha.update(include_str!("compact.rs").as_bytes());
    #[cfg(feature = "explore")]
    for source in [
        include_str!("explore.rs"),
        include_str!("bin/vnext-explore.rs"),
    ] {
        sha.update(source.as_bytes());
    }
    #[cfg(feature = "concurrent")]
    for source in [
        include_str!("v09_hardware.rs"),
        include_str!("v09_memory.rs"),
        include_str!("v09_schedule.rs"),
        include_str!("v09_submission.rs"),
        include_str!("bin/vnext-concurrent.rs"),
    ] {
        sha.update(source.as_bytes());
    }
    for source in [
        include_str!("main.rs"),
        include_str!("bin/vnext-release.rs"),
        include_str!("lib.rs"),
        include_str!("source.rs"),
        include_str!("hardware.rs"),
        include_str!("isa.rs"),
        include_str!("machine.rs"),
        include_str!("timing.rs"),
        include_str!("shared.rs"),
        include_str!("power.rs"),
        include_str!("baseline.rs"),
        include_str!("reference.rs"),
        include_str!("submission.rs"),
        include_str!("../Cargo.toml"),
        include_str!("../Cargo.lock"),
    ] {
        sha.update(source.as_bytes());
    }
    format!("{:x}", sha.finalize())
}
