#![cfg(feature = "compact")]
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use vnext_sim::{
    baseline::Shape,
    compact::Reader,
    v09_hardware::Hardware,
    v09_submission::{CONTRACT, Header, Record},
};
struct Program(PathBuf);
impl Drop for Program {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn program(lines: &[String]) -> Program {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "compact-test-{}-{}.jsonl",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let h = Header {
        contract: CONTRACT.into(),
        model: vnext_sim::MODEL.into(),
        hardware: Hardware::default(),
        shape: Shape::smoke(),
        batch: 1,
        decode: false,
    };
    let s = serde_json::to_string(&h).unwrap() + "\n" + &lines.join("\n") + "\n";
    fs::write(&path, s).unwrap();
    Program(path)
}
fn records(values: Vec<Value>) -> Vec<Value> {
    let p = program(&values.iter().map(Value::to_string).collect::<Vec<_>>());
    let mut reader = Reader::open(&p.0).unwrap();
    let mut out = vec![];
    while let Some(r) = reader.next_record(false).unwrap() {
        out.push(serde_json::to_value(r).unwrap());
    }
    out
}
fn repeat(tag: &str, count: Value, body: Value) -> Value {
    let mut v = json!({"var":"i","start":0,"count":count,"step":1,"body":body});
    v[tag] = "repeat".into();
    v
}
#[test]
fn nested_loops_expand_exactly_and_preserve_wave_boundary() {
    let commands = json!([{"command":"repeat","var":"j","start":0,"count":3,"step":1,"body":[{"command":"run","instruction":{"op":"fill","dst":{"add":[{"mul":[{"var":"i"},3]},{"var":"j"}]},"len":1,"value":1.0}}]}]);
    let wave =
        json!({"record":"wave","groups":[{"sm":0,"rf_kib":1,"sh_kib":0,"commands":commands}]});
    let out = records(vec![repeat("record", json!(2), json!([wave]))]);
    assert_eq!(out.len(), 2);
    for (i, r) in out.iter().enumerate() {
        let cs = r["groups"][0]["commands"].as_array().unwrap();
        assert_eq!(cs.len(), 3);
        for (j, c) in cs.iter().enumerate() {
            assert_eq!(c["instruction"]["dst"], i * 3 + j);
        }
    }
}
#[test]
fn templates_have_explicit_integer_arguments_and_no_implicit_capture() {
    let t = json!({"record":"template","name":"allocate","kind":"record","params":["size"],"body":[{"record":"alloc","len":{"var":"size"}}]});
    let call = json!({"record":"call","name":"allocate","args":[{"add":[4,{"var":"i"}]}]});
    let out = records(vec![t, repeat("record", json!(3), json!([call]))]);
    assert_eq!(
        out,
        vec![
            json!({"record":"alloc","len":4}),
            json!({"record":"alloc","len":5}),
            json!({"record":"alloc","len":6})
        ]
    );
}
fn rejects(lines: Vec<String>) {
    let p = program(&lines);
    let mut reader = Reader::open(&p.0).unwrap();
    for _ in 0..10 {
        match reader.next_record(false) {
            Err(_) => return,
            Ok(None) => panic!("invalid source accepted"),
            Ok(Some(_)) => {}
        }
    }
    panic!("failed to reject promptly");
}
#[test]
fn malformed_or_unbounded_constructs_fail() {
    for value in [
        repeat(
            "record",
            json!(1_000_001),
            json!([{"record":"alloc","len":1}]),
        ),
        repeat("record", json!(-1), json!([{"record":"alloc","len":1}])),
        json!({"record":"alloc","len":{"var":"unbound"}}),
        json!({"record":"alloc","len":{"add":[9223372036854775807i64,1]}}),
        json!({"record":"alloc","len":{"div":[1,0]}}),
        json!({"record":"call","name":"absent","args":[]}),
        json!({"record":"repeat","var":"i","start":0,"count":1,"step":0,"body":[{"record":"alloc","len":1}]}),
        json!({"record":"template","name":"t","kind":"record","params":["a","a"],"body":[{"record":"alloc","len":1}]}),
    ] {
        rejects(vec![value.to_string()]);
    }
    rejects(vec![r#"{"record":"alloc","len":1,"len":2}"#.into()]);
    rejects(vec![r#"{"record":"template","name":"t","kind":"record","params":[],"body":[{"record":"alloc","len":1,"len":2}]}"#.into()]);
    let recursive = json!({"record":"template","name":"t","kind":"record","params":[],"body":[{"record":"call","name":"t","args":[]}]});
    rejects(vec![
        recursive.to_string(),
        json!({"record":"call","name":"t","args":[]}).to_string(),
    ]);
}
#[test]
fn extra_source_after_commit_is_rejected_even_if_it_emits_no_instruction() {
    let p = program(&[repeat("record", json!(0), json!([{"record":"alloc","len":1}])).to_string()]);
    let mut r = Reader::open(&p.0).unwrap();
    assert!(r.next_record(true).is_err());
}
#[test]
fn source_hash_is_of_original_bytes_not_expanded_stream() {
    use sha2::{Digest, Sha256};
    let p = program(&[repeat("record", json!(3), json!([{"record":"alloc","len":1}])).to_string()]);
    let mut r = Reader::open(&p.0).unwrap();
    let mut count = 0;
    while let Some(Record::Alloc { .. }) = r.next_record(false).unwrap() {
        count += 1;
    }
    assert_eq!(count, 3);
    assert_eq!(
        r.hash(),
        format!("{:x}", Sha256::digest(fs::read(&p.0).unwrap()))
    );
}
