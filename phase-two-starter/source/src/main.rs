use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
use vnext_sim::{
    Result,
    baseline::{self, Fixture, Shape},
    hardware::Hardware,
    isa::Program,
    machine::Machine,
    reference,
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    shape: Shape,
    hardware: Hardware,
    tile: usize,
    seed: u64,
    #[serde(default)]
    fixture_profile: baseline::FixtureProfile,
    #[serde(default)]
    prefill_policy: Option<baseline::Policy>,
    #[serde(default)]
    decode_policy: Option<baseline::Policy>,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn source_hash() -> String {
    vnext_sim::source::hash()
}
fn output_hash(o: &baseline::Outputs) -> String {
    let mut s = Sha256::new();
    for v in o.hidden.iter().chain(&o.keys).chain(&o.values) {
        s.update((v.len() as u64).to_le_bytes());
        for x in v {
            s.update(x.to_le_bytes());
        }
    }
    format!("{:x}", s.finalize())
}
fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    let data = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    fs::write(path, data).map_err(|e| e.to_string())
}
fn new_output_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(path).map_err(|e| e.to_string())
}
fn read_json(path: &str) -> Result<Vec<u8>> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > 64 * 1024 * 1024 {
        return Err("JSON input exceeds 64 MiB; use the streaming Rust API".into());
    }
    fs::read(path).map_err(|e| e.to_string())
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 || !matches!(args[1].as_str(), "check" | "estimate" | "program") {
        return Err("usage: vnext-sim <check|estimate> <smoke|main|stress|config.json> <seed> <new-output-dir>\n       vnext-sim program <program.json> 0 <new-output-dir>".into());
    }
    let out = Path::new(&args[4]);
    if out.exists() {
        return Err("output directory exists; refusing overwrite".into());
    }
    if args[1] == "program" {
        let bytes = read_json(&args[2])?;
        let p: Program = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if p.model != vnext_sim::MODEL {
            return Err("unsupported model version".into());
        }
        let mut m = Machine::new(p.hardware, true)?;
        for (i, t) in p.tensors.into_iter().enumerate() {
            m.alloc(&format!("tensor{i}"), t)?;
        }
        for wave in p.waves {
            m.wave(wave)?;
        }
        let report = m.report();
        new_output_dir(out)?;
        write(&out.join("report.json"), &report)?;
        write(
            &out.join("tensors.json"),
            &m.tensors.iter().map(|t| &t.data).collect::<Vec<_>>(),
        )?;
        write(
            &out.join("manifest.json"),
            &serde_json::json!({"model":vnext_sim::MODEL,"source_sha256":source_hash(),"program_sha256":hash(&bytes)}),
        )?;
        return Ok(());
    }
    let seed = args[3].parse::<u64>().map_err(|e| e.to_string())?;
    let mut config = match args[2].as_str() {
        "smoke" => Config {
            shape: Shape::smoke(),
            hardware: Hardware::default(),
            tile: 32,
            fixture_profile: baseline::FixtureProfile::default(),
            prefill_policy: None,
            decode_policy: None,
            seed,
        },
        "main" => Config {
            shape: Shape::main(),
            hardware: Hardware::default(),
            tile: 32,
            fixture_profile: baseline::FixtureProfile::default(),
            prefill_policy: None,
            decode_policy: None,
            seed,
        },
        "stress" => Config {
            shape: Shape::stress(),
            hardware: Hardware::default(),
            tile: 32,
            fixture_profile: baseline::FixtureProfile::default(),
            prefill_policy: None,
            decode_policy: None,
            seed,
        },
        path => serde_json::from_slice::<Config>(&read_json(path)?).map_err(|e| e.to_string())?,
    };
    config.seed = seed;
    config.shape.validate()?;
    config.hardware.validate()?;
    if ![8, 16, 32].contains(&config.tile) {
        return Err("invalid baseline tile".into());
    }
    for policy in [config.prefill_policy, config.decode_policy]
        .into_iter()
        .flatten()
    {
        policy.validate()?;
    }
    let start = Instant::now();
    let fixture = Fixture::new_profile(config.shape, seed, config.fixture_profile)?;
    let fixture_seconds = start.elapsed().as_secs_f64();
    new_output_dir(out)?;
    write(&out.join("config.json"), &config)?;
    let mut reports = vec![];
    for decode in [false, true] {
        let name = if decode { "decode" } else { "prefill" };
        eprintln!("{name}: starting {}", args[1]);
        let mut m = Machine::new(config.hardware.clone(), args[1] == "check")?;
        m.timing.cycle_reference = std::env::var("VNEXT_CYCLE_REFERENCE").is_ok_and(|s| s == "1");
        let start = Instant::now();
        let policy = if decode {
            config.decode_policy
        } else {
            config.prefill_policy
        }
        .unwrap_or_else(|| baseline::Policy::square(config.tile));
        let actual = baseline::run_policy(&mut m, &fixture, decode, policy)?;
        let simulation_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let comparison = if args[1] == "check" {
            Some(reference::compare(
                &actual,
                &reference::run(&fixture, decode),
            )?)
        } else {
            None
        };
        let reference_seconds = start.elapsed().as_secs_f64();
        let report = m.report();
        eprintln!(
            "{name}: {} cycles, simulation {simulation_seconds:.3}s, reference {reference_seconds:.3}s, power {:.3}/{:.3} W",
            report.stats.cycles, report.short_power_w, report.long_power_w
        );
        reports.push(serde_json::json!({"scenario":name,"simulation_seconds":simulation_seconds,"reference_seconds":reference_seconds,"comparison":comparison,"output_sha256":if args[1]=="check"{Some(output_hash(&actual))}else{None},"report":report}));
    }
    write(&out.join("report.json"), &reports)?;
    write(
        &out.join("manifest.json"),
        &serde_json::json!({"model":vnext_sim::MODEL,"source_sha256":source_hash(),"config_sha256":hash(&serde_json::to_vec(&config).unwrap()),"fixture_version":"splitmix64-fp32-v2","fixture_profile":config.fixture_profile,"fixture_seconds":fixture_seconds,"arch":std::env::consts::ARCH,"os":std::env::consts::OS,"official_score":null,"cycle_reference":std::env::var("VNEXT_CYCLE_REFERENCE").is_ok_and(|s|s=="1"),"status":"complete"}),
    )?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
