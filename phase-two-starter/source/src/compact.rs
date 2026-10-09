//! Bounded, data-independent source loops. No host code or tensor-value expressions.
use crate::{
    Result,
    v09_schedule::{Command, Group},
    v09_submission::{Header, Record},
};
use serde::{
    Deserialize,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path, sync::Arc};
const DEPTH: usize = 16;
const CONTROL: u64 = 2_000_000;
type Env = BTreeMap<String, i64>;
// Value's normal deserializer silently overwrites duplicate keys. Reject them at
// every nesting level before any expansion, including unused template fields.
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Unique;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON with unique keys")
            }
            fn visit_bool<E: de::Error>(self, x: bool) -> std::result::Result<Unique, E> {
                Ok(Unique(x.into()))
            }
            fn visit_i64<E: de::Error>(self, x: i64) -> std::result::Result<Unique, E> {
                Ok(Unique(x.into()))
            }
            fn visit_u64<E: de::Error>(self, x: u64) -> std::result::Result<Unique, E> {
                Ok(Unique(x.into()))
            }
            fn visit_f64<E: de::Error>(self, x: f64) -> std::result::Result<Unique, E> {
                serde_json::Number::from_f64(x)
                    .map(|n| Unique(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite JSON"))
            }
            fn visit_str<E: de::Error>(self, x: &str) -> std::result::Result<Unique, E> {
                Ok(Unique(x.into()))
            }
            fn visit_string<E: de::Error>(self, x: String) -> std::result::Result<Unique, E> {
                Ok(Unique(x.into()))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut v = vec![];
                while let Some(x) = a.next_element::<Unique>()? {
                    v.push(x.0);
                }
                Ok(Unique(v.into()))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut v = Map::new();
                while let Some(k) = a.next_key::<String>()? {
                    if v.contains_key(&k) {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                    v.insert(k, a.next_value::<Unique>()?.0);
                }
                Ok(Unique(v.into()))
            }
        }
        d.deserialize_any(V)
    }
}
fn object(v: &Value) -> Result<&Map<String, Value>> {
    v.as_object().ok_or("expected object".into())
}
fn fields(v: &Value, names: &[&str]) -> Result<()> {
    let o = object(v)?;
    if o.len() != names.len() || names.iter().any(|n| !o.contains_key(*n)) {
        Err("missing/unknown compact field".into())
    } else {
        Ok(())
    }
}
fn name(v: &Value) -> Result<String> {
    let s = v.as_str().ok_or("name must be a string")?;
    if s.is_empty() || s.len() > 32 || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return Err("invalid compact name".into());
    }
    Ok(s.into())
}
fn integer(v: &Value, e: &Env, d: usize) -> Result<i64> {
    fn inner(v: &Value, e: &Env, d: usize, nodes: &mut usize) -> Result<i64> {
        *nodes += 1;
        if *nodes > 512 {
            return Err("integer expression node budget".into());
        }
        if d > DEPTH {
            return Err("expression nesting budget".into());
        }
        if let Some(x) = v.as_i64() {
            return Ok(x);
        }
        let o = object(v)?;
        if o.len() != 1 {
            return Err("invalid integer expression".into());
        }
        let (op, a) = o.iter().next().unwrap();
        if op == "var" {
            return e.get(&name(a)?).copied().ok_or("unbound variable".into());
        }
        let args = a
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or("binary expression needs two operands")?;
        let x = inner(&args[0], e, d + 1, nodes)?;
        let y = inner(&args[1], e, d + 1, nodes)?;
        match op.as_str() {
            "add" => x.checked_add(y),
            "sub" => x.checked_sub(y),
            "mul" => x.checked_mul(y),
            "div" => x.checked_div(y),
            "mod" => x.checked_rem(y),
            "min" => Some(x.min(y)),
            "max" => Some(x.max(y)),
            _ => return Err("unknown integer operator".into()),
        }
        .ok_or("integer overflow or division by zero".into())
    }
    inner(v, e, d, &mut 0)
}
fn resolve(v: &Value, e: &Env, d: usize) -> Result<Value> {
    if d > 64 {
        return Err("payload nesting budget".into());
    }
    match v {
        Value::Object(o)
            if o.len() == 1
                && o.keys().any(|k| {
                    matches!(
                        k.as_str(),
                        "var" | "add" | "sub" | "mul" | "div" | "mod" | "min" | "max"
                    )
                }) =>
        {
            Ok(integer(v, e, 0)?.into())
        }
        Value::Object(o) => Ok(Value::Object(
            o.iter()
                .map(|(k, v)| Ok((k.clone(), resolve(v, e, d + 1)?)))
                .collect::<Result<_>>()?,
        )),
        Value::Array(a) => Ok(Value::Array(
            a.iter()
                .map(|v| resolve(v, e, d + 1))
                .collect::<Result<_>>()?,
        )),
        _ => Ok(v.clone()),
    }
}
#[derive(Clone)]
struct Template {
    kind: String,
    params: Vec<String>,
    body: Arc<Vec<Value>>,
}
struct Frame {
    body: Arc<Vec<Value>>,
    index: usize,
    iteration: i64,
    count: i64,
    var: Option<String>,
    start: i64,
    step: i64,
    env: Env,
}
fn repetition(v: &Value, e: &Env, tag: &str) -> Result<Frame> {
    fields(v, &[tag, "var", "start", "count", "step", "body"])?;
    let var = name(&v["var"])?;
    if e.contains_key(&var) {
        return Err("loop variable shadowing".into());
    }
    let start = integer(&v["start"], e, 0)?;
    let count = integer(&v["count"], e, 0)?;
    let step = integer(&v["step"], e, 0)?;
    if !(0..=1_000_000).contains(&count) || step == 0 {
        return Err("invalid loop count/step".into());
    }
    if count > 0 {
        start
            .checked_add((count - 1).checked_mul(step).ok_or("loop overflow")?)
            .ok_or("loop overflow")?;
    }
    let body = v["body"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("empty/nonarray loop body")?
        .clone();
    Ok(Frame {
        body: Arc::new(body),
        index: 0,
        iteration: 0,
        count,
        var: Some(var),
        start,
        step,
        env: e.clone(),
    })
}
fn invocation(v: &Value, e: &Env, tag: &str, t: &BTreeMap<String, Template>) -> Result<Frame> {
    fields(v, &[tag, "name", "args"])?;
    let n = name(&v["name"])?;
    let t = t.get(&n).ok_or("undefined template")?;
    if t.kind != tag {
        return Err("template scope mismatch".into());
    }
    let a = v["args"]
        .as_array()
        .filter(|a| a.len() == t.params.len())
        .ok_or("template argument count")?;
    let env = t
        .params
        .iter()
        .zip(a)
        .map(|(n, v)| Ok((n.clone(), integer(v, e, 0)?)))
        .collect::<Result<_>>()?;
    Ok(Frame {
        body: t.body.clone(),
        index: 0,
        iteration: 0,
        count: 1,
        var: None,
        start: 0,
        step: 1,
        env,
    })
}
fn item(stack: &mut Vec<Frame>) -> Option<(Value, Env)> {
    loop {
        let f = stack.last_mut()?;
        if f.index == f.body.len() {
            f.index = 0;
            f.iteration += 1;
        }
        if f.iteration >= f.count {
            stack.pop();
            continue;
        }
        let mut env = f.env.clone();
        if let Some(n) = &f.var {
            env.insert(n.clone(), f.start + f.iteration * f.step);
        }
        let v = f.body[f.index].clone();
        f.index += 1;
        return Some((v, env));
    }
}
fn commands(
    body: Vec<Value>,
    env: Env,
    templates: &BTreeMap<String, Template>,
    count: &mut usize,
    control: &mut u64,
) -> Result<Vec<Command>> {
    let mut stack = vec![Frame {
        body: Arc::new(body),
        index: 0,
        iteration: 0,
        count: 1,
        var: None,
        start: 0,
        step: 1,
        env,
    }];
    let mut out = vec![];
    while let Some((v, e)) = item(&mut stack) {
        *control += 1;
        if *control > CONTROL + 100_000_000 {
            return Err("expanded control-work budget".into());
        }
        let tag = v
            .get("command")
            .and_then(Value::as_str)
            .ok_or("missing command")?;
        let frame = match tag {
            "repeat" => Some(repetition(&v, &e, "command")?),
            "call" => Some(invocation(&v, &e, "command", templates)?),
            _ => None,
        };
        if let Some(f) = frame {
            if stack.len() >= DEPTH {
                return Err("command nesting budget".into());
            }
            stack.push(f);
        } else {
            *count += 1;
            if *count > crate::MAX_WAVE_COMMANDS {
                return Err("expanded wave command budget".into());
            }
            out.push(serde_json::from_value(resolve(&v, &e, 0)?).map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}
fn record(
    v: &Value,
    e: &Env,
    templates: &BTreeMap<String, Template>,
    control: &mut u64,
) -> Result<Record> {
    if v.get("record").and_then(Value::as_str) != Some("wave") {
        return serde_json::from_value(resolve(v, e, 0)?).map_err(|e| e.to_string());
    }
    fields(v, &["record", "groups"])?;
    let gs = v["groups"]
        .as_array()
        .filter(|a| a.len() <= 256)
        .ok_or("group limit")?;
    let mut count = 0;
    let mut out = vec![];
    for g in gs {
        fields(g, &["sm", "rf_kib", "sh_kib", "commands"])?;
        let index = |s: &str| -> Result<usize> {
            usize::try_from(integer(&g[s], e, 0)?).map_err(|_| "negative/index overflow".into())
        };
        let body = g["commands"]
            .as_array()
            .ok_or("commands must be an array")?
            .clone();
        out.push(Group {
            sm: index("sm")?,
            rf_kib: index("rf_kib")?,
            sh_kib: index("sh_kib")?,
            commands: commands(body, e.clone(), templates, &mut count, control)?,
        });
    }
    Ok(Record::Wave { groups: out })
}
pub struct Reader {
    reader: BufReader<File>,
    buffer: Vec<u8>,
    total: u64,
    hash: Sha256,
    templates: BTreeMap<String, Template>,
    stack: Vec<Frame>,
    physical: u64,
    template_bytes: usize,
    control: u64,
    pub header: Header,
}
impl Reader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        let mut buffer = vec![];
        let mut total = 0;
        if !crate::submission::line(&mut reader, &mut buffer, &mut total)? {
            return Err("empty program".into());
        }
        let header = serde_json::from_slice(&buffer).map_err(|e| e.to_string())?;
        let mut hash = Sha256::new();
        hash.update(&buffer);
        Ok(Self {
            reader,
            buffer,
            total,
            hash,
            templates: BTreeMap::new(),
            stack: vec![],
            physical: 0,
            template_bytes: 0,
            control: 0,
            header,
        })
    }
    pub fn hash(&self) -> String {
        format!("{:x}", self.hash.clone().finalize())
    }
    pub fn next_record(&mut self, finished: bool) -> Result<Option<Record>> {
        loop {
            let (v, e) = if let Some(x) = item(&mut self.stack) {
                x
            } else {
                if !crate::submission::line(&mut self.reader, &mut self.buffer, &mut self.total)? {
                    return Ok(None);
                }
                self.hash.update(&self.buffer);
                self.physical += 1;
                if self.physical > 200_000 {
                    return Err("source record budget".into());
                }
                (
                    serde_json::from_slice::<Unique>(&self.buffer)
                        .map_err(|e| e.to_string())?
                        .0,
                    Env::new(),
                )
            };
            if finished {
                return Err("extra records after final commit".into());
            }
            self.control += 1;
            if self.control > CONTROL + 100_000_000 {
                return Err("expanded control-work budget".into());
            }
            match v
                .get("record")
                .and_then(Value::as_str)
                .ok_or("missing record tag")?
            {
                "template" => {
                    if !self.stack.is_empty() {
                        return Err("templates must be top-level source records".into());
                    }
                    fields(&v, &["record", "name", "kind", "params", "body"])?;
                    let n = name(&v["name"])?;
                    if self.templates.contains_key(&n) || self.templates.len() >= 4096 {
                        return Err("duplicate/template count budget".into());
                    }
                    let kind = v["kind"]
                        .as_str()
                        .filter(|s| *s == "record" || *s == "command")
                        .ok_or("invalid template scope")?
                        .to_string();
                    let params = v["params"]
                        .as_array()
                        .filter(|a| a.len() <= 64)
                        .ok_or("parameter budget")?
                        .iter()
                        .map(name)
                        .collect::<Result<Vec<_>>>()?;
                    let mut names = params.clone();
                    names.sort();
                    names.dedup();
                    if names.len() != params.len() {
                        return Err("duplicate parameter".into());
                    }
                    let body = v["body"]
                        .as_array()
                        .filter(|a| !a.is_empty())
                        .ok_or("empty/nonarray template body")?
                        .clone();
                    self.template_bytes = self
                        .template_bytes
                        .checked_add(serde_json::to_vec(&v).map_err(|e| e.to_string())?.len())
                        .ok_or("template bytes overflow")?;
                    if self.template_bytes > 8 * 1024 * 1024 {
                        return Err("retained template source exceeds 8 MiB".into());
                    }
                    self.templates.insert(
                        n,
                        Template {
                            kind,
                            params,
                            body: Arc::new(body),
                        },
                    );
                }
                "repeat" | "call" => {
                    if self.stack.len() >= DEPTH {
                        return Err("record nesting budget".into());
                    }
                    let f = if v["record"] == "repeat" {
                        repetition(&v, &e, "record")?
                    } else {
                        invocation(&v, &e, "record", &self.templates)?
                    };
                    self.stack.push(f);
                }
                _ => return Ok(Some(record(&v, &e, &self.templates, &mut self.control)?)),
            }
        }
    }
}
