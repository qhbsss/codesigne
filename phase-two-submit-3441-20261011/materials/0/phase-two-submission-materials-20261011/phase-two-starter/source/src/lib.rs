pub mod baseline;
pub mod hardware;
pub mod isa;
pub mod machine;
pub mod power;
pub mod reference;
mod shared;
pub mod source;
pub mod submission;
pub mod timing;
#[cfg(not(feature = "explore"))]
pub const MODEL: &str = "vnext-blocking-sh-v0.6";
pub type Result<T> = std::result::Result<T, String>;

#[cfg(all(feature = "explore", not(feature = "concurrent")))]
pub const MODEL: &str = "vnext-explore-v0.8";
#[cfg(feature = "explore")]
pub mod explore;

#[cfg(all(feature = "concurrent", not(feature = "long-waves")))]
pub const MODEL: &str = "vnext-concurrent-v0.9";
#[cfg(feature = "concurrent")]
pub mod v09_hardware;
#[cfg(feature = "concurrent")]
pub mod v09_memory;
#[cfg(feature = "concurrent")]
pub mod v09_schedule;
#[cfg(feature = "concurrent")]
pub mod v09_submission;

/// Experimental admission limits only; hardware timing/costs are unchanged.
#[cfg(all(feature = "long-waves", not(feature = "compact")))]
pub const MODEL: &str = "phase-two-long-waves-experiment-v1";
pub const MAX_WAVE_PRIMITIVES: usize = if cfg!(feature = "compact") {
    1_048_576
} else if cfg!(feature = "long-waves") {
    262_144
} else {
    16_384
};
pub const MAX_WAVE_COMMANDS: usize = 2 * MAX_WAVE_PRIMITIVES;

#[cfg(feature = "compact")]
pub const MODEL: &str = "phase-two-compact-v2";
#[cfg(feature = "compact")]
pub mod compact;
