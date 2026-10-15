use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
use vnext_sim::{
    Result,
    explore::{Config, run_batched},
    machine::Machine,
    reference,
};
fn write(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 || !matches!(args[1].as_str(), "check" | "estimate" | "probe") {
        return Err("vnext-explore <check|estimate|probe> CONFIG SEED NEW_OUTPUT_DIR".into());
    }
    let seed: u64 = args[3].parse().map_err(|_| "invalid seed")?;
    let path = Path::new(&args[2]);
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > 1024 * 1024 {
        return Err("config exceeds 1 MiB".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let c: Config = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    c.validate()?;
    let out = Path::new(&args[4]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    write(&out.join("config.json"), &c)?;
    let total = Instant::now();
    let mut stage = "fixture";
    let result: Result<serde_json::Value> = (|| {
        let f = c.fixture(seed)?;
        let fixture_seconds = total.elapsed().as_secs_f64();
        let start = Instant::now();
        // Reference temporaries die before allocating the simulated machine.
        let expected = if args[1] != "estimate" {
            Some(reference::run_batched(
                &f,
                c.decode,
                c.batch,
                c.reference_threads,
            ))
        } else {
            None
        };
        let reference_seconds = start.elapsed().as_secs_f64();
        if args[1] == "probe" {
            let wrong =
                reference::uniform_attention_control(&f, c.decode, c.batch, c.reference_threads);
            let rejection = reference::compare(&wrong, expected.as_ref().unwrap()).err();
            if rejection.is_none() {
                return Err("uniform attention escaped the numerical contract".into());
            }
            return Ok(
                serde_json::json!({"negative_control":"uniform_attention","rejected":true,"reason":rejection,
                "fixture_seconds":fixture_seconds,"reference_seconds":reference_seconds}),
            );
        }
        stage = "simulation";
        let start = Instant::now();
        let mut m = Machine::new(c.hardware.clone(), expected.is_some())?;
        let actual = run_batched(&mut m, &f, c.decode, c.policy, c.batch)?;
        let simulation_seconds = start.elapsed().as_secs_f64();
        stage = "comparison";
        let comparison = expected
            .as_ref()
            .map(|e| reference::compare(&actual, e))
            .transpose()?;
        let report = m.report();
        let mut hash = Sha256::new();
        for row in actual
            .hidden
            .iter()
            .chain(&actual.keys)
            .chain(&actual.values)
        {
            hash.update((row.len() as u64).to_le_bytes());
            for x in row {
                hash.update(x.to_le_bytes());
            }
        }
        Ok(
            serde_json::json!({"fixture_seconds":fixture_seconds,"reference_seconds":reference_seconds,
            "simulation_seconds":simulation_seconds,"comparison":comparison,"report":report,
            "output_sha256":if expected.is_some(){Some(format!("{:x}",hash.finalize()))}else{None}}),
        )
    })();
    write(
        &out.join("result.json"),
        &serde_json::json!({
            "model":vnext_sim::MODEL,"source_sha256":vnext_sim::source::hash(),
            "fixture_version":"batch-shared-weights-v1","config_sha256":format!("{:x}",Sha256::digest(&bytes)),"seed":seed,"mode":args[1],
            "status":if result.is_ok(){"complete"}else{"failed"},"failed_stage":if result.is_err(){Some(stage)}else{None},
            "error":result.as_ref().err(),"host_seconds":total.elapsed().as_secs_f64(),"evaluation":result.as_ref().ok(),
            "power_integrator":if std::env::var("VNEXT_POWER_SPARSE").as_deref()==Ok("1"){"sparse"}else{"dense"},
        "official_score":null,"note":"trusted research generator; not an untrusted-submission grading entry point"
        }),
    )?;
    result.map(|_| ())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
