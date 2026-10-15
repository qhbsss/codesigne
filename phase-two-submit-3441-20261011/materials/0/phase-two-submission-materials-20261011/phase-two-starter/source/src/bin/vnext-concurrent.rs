use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
use vnext_sim::{
    Result,
    baseline::FixtureProfile,
    v09_submission::{self as submission, Config},
};
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "vnext-concurrent <check|export|evaluate|grade> CONFIG_OR_PROGRAM SEED NEW_OUTPUT_DIR"
                .into(),
        );
    }
    let seed = args[3].parse::<u64>().map_err(|_| "invalid seed")?;
    let out = Path::new(&args[4]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let input = Path::new(&args[2]);
    let mut config_hash = None;
    let result: Result<serde_json::Value> = (|| match args[1].as_str() {
        "contract" => Ok(
            serde_json::json!({"limits":{"program_file_bytes":vnext_sim::submission::MAX_FILE_BYTES,"line_bytes":vnext_sim::submission::MAX_LINE_BYTES,"wave_primitives":vnext_sim::MAX_WAVE_PRIMITIVES,"wave_commands":vnext_sim::MAX_WAVE_COMMANDS,"wave_distinct_access_intervals":vnext_sim::machine::MAX_WAVE_ACCESS_INTERVALS,"area_mm2_per_au":0.25},"cases":submission::CASES.iter().enumerate().map(|(i,n)|{let(shape,batch,decode)=submission::known_shape(n).unwrap();(n.to_string(),serde_json::json!({"shape":shape,"batch":batch,"decode":decode,"reference_cycles":submission::REFERENCE_CYCLES[i],"weight":if i<2 {0.25}else{1./6.}}))}).collect::<std::collections::BTreeMap<_,_>>() }),
        ),
        "check" | "export" => {
            if fs::metadata(input).map_err(|e| e.to_string())?.len() > 1024 * 1024 {
                return Err("config too large".into());
            }
            let bytes = fs::read(input).map_err(|e| e.to_string())?;
            config_hash = Some(format!("{:x}", Sha256::digest(&bytes)));
            let c: Config = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            if args[1] == "export" {
                submission::export(&c, &out.join("program.jsonl"))?;
                Ok(serde_json::json!({"exported":"program.jsonl"}))
            } else {
                submission::check(&c, seed)
            }
        }
        "grade" => serde_json::to_value(submission::grade(
            input,
            seed,
            FixtureProfile::AttentionStress,
        )?)
        .map_err(|e| e.to_string()),
        "evaluate" => serde_json::to_value(submission::evaluate(
            input,
            seed,
            FixtureProfile::AttentionStress,
        )?)
        .map_err(|e| e.to_string()),
        _ => Err("unknown command".into()),
    })();
    let value = serde_json::json!({"model":vnext_sim::MODEL,"contract":submission::CONTRACT,"source_sha256":vnext_sim::source::hash(),"config_sha256":config_hash,"seed":seed,"mode":args[1],"status":if result.is_ok(){"complete"}else{"failed"},"error":result.as_ref().err(),"evaluation":result.as_ref().ok(),"host_seconds":start.elapsed().as_secs_f64(),"official_score":if args[1]=="grade"{result.as_ref().ok().and_then(|v|v.get("score")).cloned()}else{None}});
    fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    result.map(|_| ())
}
