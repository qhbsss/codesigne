use std::{fs, path::Path, time::Instant};
use vnext_sim::{
    Result,
    baseline::{Fixture, FixtureProfile},
    submission::{self, ExportConfig},
};
fn write(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn usage() -> String {
    "vnext-release export CONFIG NEW_PROGRAM_DIR\nvnext-release check PROGRAM_DIR SEED <legacy|attention_stress> NEW_RESULT_DIR\nvnext-release grade PROGRAM_DIR SEED NEW_RESULT_DIR\ngrade: fixed M, two seeds × two profiles × P/D; check: diagnostic only".into()
}
fn run() -> Result<()> {
    if cfg!(feature = "explore") {
        return Err("explore build cannot grade frozen release programs".into());
    }
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "export") && args.len() == 4 {
        let path = Path::new(&args[2]);
        if fs::metadata(path).map_err(|e| e.to_string())?.len() > 1024 * 1024 {
            return Err("config exceeds 1 MiB".into());
        }
        let config: ExportConfig =
            serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        submission::export(&config, Path::new(&args[3]))?;
        return write(
            &Path::new(&args[3]).join("export.json"),
            &serde_json::json!({"contract":submission::CONTRACT,"source_sha256":vnext_sim::source::hash(),"config":config,"note":"offline static generator; no fixture values or seed included in programs"}),
        );
    }
    let grade = args.get(1).is_some_and(|s| s == "grade");
    if !(grade && args.len() == 5 || args.get(1).is_some_and(|s| s == "check") && args.len() == 6) {
        return Err(usage());
    }
    let dir = Path::new(&args[2]);
    let seed: u64 = args[3].parse().map_err(|_| "invalid seed")?;
    let out = Path::new(args.last().unwrap());
    let (p, d) = submission::pair_headers(dir, grade)?;
    let cases = if grade {
        vec![
            (seed, FixtureProfile::Legacy),
            (seed, FixtureProfile::AttentionStress),
            (seed ^ 0x9e3779b97f4a7c15, FixtureProfile::Legacy),
            (seed ^ 0x9e3779b97f4a7c15, FixtureProfile::AttentionStress),
        ]
    } else {
        let profile = match args[4].as_str() {
            "legacy" => FixtureProfile::Legacy,
            "attention_stress" => FixtureProfile::AttentionStress,
            _ => return Err("unknown profile".into()),
        };
        vec![(seed, profile)]
    };
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let mut reports = vec![];
    let result: Result<()> = (|| {
        for (seed, profile) in cases {
            let fixture = Fixture::new_profile(p.shape, seed, profile)?;
            for (scenario, header) in [("prefill", &p), ("decode", &d)] {
                eprintln!("{scenario}: seed={seed}, profile={profile:?}");
                let begin = Instant::now();
                let eval = submission::evaluate(&dir.join(format!("{scenario}.jsonl")), &fixture)?;
                if serde_json::to_value(&eval.header).unwrap()
                    != serde_json::to_value(header).unwrap()
                {
                    return Err("header changed during evaluation".into());
                }
                if let Some(previous) = reports
                    .iter()
                    .find(|v: &&serde_json::Value| v["scenario"] == scenario)
                    && (previous["evaluation"]["program_sha256"] != eval.program_sha256
                        || previous["evaluation"]["report"]
                            != serde_json::to_value(&eval.report).unwrap())
                {
                    return Err("program or timing changed between fixtures".into());
                }
                reports.push(serde_json::json!({"scenario":scenario,"seed":seed,"profile":profile,"host_seconds":begin.elapsed().as_secs_f64(),"evaluation":eval}));
                write(&out.join("evaluations.json"), &reports)?;
            }
        }
        Ok(())
    })();
    let cycles = |scenario: &str| {
        reports
            .iter()
            .find(|v| v["scenario"] == scenario)
            .and_then(|v| v["evaluation"]["report"]["stats"]["cycles"].as_u64())
    };
    let score = if grade && result.is_ok() {
        Some(submission::score(
            cycles("prefill").unwrap(),
            cycles("decode").unwrap(),
        )?)
    } else {
        None
    };
    write(
        &out.join("result.json"),
        &serde_json::json!({
            "contract":submission::CONTRACT,"model":vnext_sim::MODEL,"source_sha256":vnext_sim::source::hash(),
            "status":if result.is_ok(){"complete"}else{"failed"},"error":result.as_ref().err(),
            "qualified":grade&&result.is_ok(),"experimental_score":score,"official_score":null,
            "reference_cycles":{"prefill":submission::REFERENCE_PREFILL,"decode":submission::REFERENCE_DECODE},
            "host_seconds":start.elapsed().as_secs_f64(),"completed_evaluations":reports.len(),
            "note":"finite fixture verification; official acceptance requires organizer rerun of these exact program hashes"
        }),
    )?;
    result
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
