use crate::{
    Result,
    isa::{Arg, Instr, Op, View, Wave},
    machine::Machine,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub layers: usize,
    pub d: usize,
    pub heads: usize,
    pub f: usize,
    pub prefill: usize,
    pub history: usize,
    pub steps: usize,
}
impl Shape {
    pub fn smoke() -> Self {
        Self {
            layers: 2,
            d: 32,
            heads: 2,
            f: 80,
            prefill: 7,
            history: 11,
            steps: 3,
        }
    }
    pub fn main() -> Self {
        Self {
            layers: 4,
            d: 768,
            heads: 12,
            f: 3072,
            prefill: 256,
            history: 1024,
            steps: 8,
        }
    }
    pub fn stress() -> Self {
        Self {
            layers: 8,
            d: 1024,
            heads: 16,
            f: 4096,
            prefill: 512,
            history: 2048,
            steps: 32,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.layers == 0
            || self.layers > 8
            || self.d == 0
            || self.d
                > if cfg!(feature = "explore") {
                    1536
                } else {
                    1024
                }
            || self.heads == 0
            || !self.d.is_multiple_of(self.heads)
            || self.f == 0
            || self.f
                > if cfg!(feature = "explore") {
                    6144
                } else {
                    4096
                }
            || self.prefill == 0
            || self.prefill > if cfg!(feature = "explore") { 2048 } else { 512 }
            || self.history == 0
            || self.history
                > if cfg!(feature = "explore") {
                    4096
                } else {
                    2048
                }
            || self.steps == 0
            || self.steps > 32
        {
            return Err("unsupported/invalid workload shape".into());
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct Layer {
    pub qkv: Vec<f32>,
    pub qb: Vec<f32>,
    pub out: Vec<f32>,
    pub ob: Vec<f32>,
    pub up: Vec<f32>,
    pub ub: Vec<f32>,
    pub down: Vec<f32>,
    pub db: Vec<f32>,
    pub g1: Vec<f32>,
    pub b1: Vec<f32>,
    pub g2: Vec<f32>,
    pub b2: Vec<f32>,
    pub past_k: Vec<f32>,
    pub past_v: Vec<f32>,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureProfile {
    Legacy,
    #[default]
    AttentionStress,
}
pub struct Fixture {
    pub shape: Shape,
    pub layers: Vec<Layer>,
    pub prompt: Vec<f32>,
    pub inputs: Vec<Vec<f32>>,
}
struct Random(u64);
impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 40) as f32 / 8388608. - 1.
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}
impl Fixture {
    pub fn new_profile(shape: Shape, seed: u64, profile: FixtureProfile) -> Result<Self> {
        let mut fixture = Self::new(shape, seed)?;
        if matches!(profile, FixtureProfile::AttentionStress) {
            for layer in &mut fixture.layers {
                for row in layer.qkv.chunks_mut(3 * shape.d) {
                    for v in &mut row[..2 * shape.d] {
                        *v *= 4.;
                    }
                }
                for v in &mut layer.past_k {
                    *v *= 4.;
                }
            }
        }
        Ok(fixture)
    }

    pub fn new(shape: Shape, seed: u64) -> Result<Self> {
        shape.validate()?;
        let mut rng = Random(seed);
        let d = shape.d;
        let f = shape.f;
        let mut layers = vec![];
        for _ in 0..shape.layers {
            layers.push(Layer {
                qkv: rng.vec(d * 3 * d, 0.5 / (d as f32).sqrt()),
                qb: rng.vec(3 * d, 0.01),
                out: rng.vec(d * d, 0.5 / (d as f32).sqrt()),
                ob: rng.vec(d, 0.01),
                up: rng.vec(d * f, 0.5 / (d as f32).sqrt()),
                ub: rng.vec(f, 0.01),
                down: rng.vec(f * d, 0.5 / (f as f32).sqrt()),
                db: rng.vec(d, 0.01),
                g1: rng.vec(d, 0.1).into_iter().map(|v| v + 1.).collect(),
                b1: rng.vec(d, 0.01),
                g2: rng.vec(d, 0.1).into_iter().map(|v| v + 1.).collect(),
                b2: rng.vec(d, 0.01),
                past_k: rng.vec(shape.history * d, 0.25),
                past_v: rng.vec(shape.history * d, 0.25),
            });
        }
        Ok(Self {
            shape,
            layers,
            prompt: rng.vec(shape.prefill * d, 0.5),
            inputs: (0..shape.steps).map(|_| rng.vec(d, 0.5)).collect(),
        })
    }
}
#[derive(Clone, Copy)]
pub struct Matrix {
    pub tensor: usize,
    pub rows: usize,
    pub cols: usize,
    pub base: usize,
    pub rs: usize,
    pub cs: usize,
}
impl Matrix {
    pub fn plain(tensor: usize, rows: usize, cols: usize) -> Self {
        Self {
            tensor,
            rows,
            cols,
            base: 0,
            rs: cols,
            cs: 1,
        }
    }
    fn view(&self, r: usize, c: usize, rows: usize, cols: usize) -> View {
        View {
            base: self.base + r * self.rs + c * self.cs,
            rows,
            cols,
            row_stride: self.rs,
            col_stride: self.cs,
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    #[serde(default)]
    pub vector_gemv: bool,
    #[serde(default)]
    pub pack_transpose: bool,
    #[serde(default = "one")]
    pub split_k: usize,
    #[serde(default = "one")]
    pub head_group: usize,
    #[serde(default)]
    pub persistent_keys: bool,
    #[serde(default)]
    pub sh_rows: usize,
    #[serde(default)]
    pub sh_direct: bool,
}
fn one() -> usize {
    1
}
impl Policy {
    pub fn square(tile: usize) -> Self {
        Self {
            m: tile,
            n: tile,
            k: 64,
            vector_gemv: false,
            pack_transpose: false,
            split_k: 1,
            head_group: 1,
            persistent_keys: false,
            sh_rows: 0,
            sh_direct: false,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.m == 0 || self.m > 64 || self.n == 0 || self.n > 64 || self.k == 0 || self.k > 256 {
            return Err("policy requires M,N in 1..64 and K in 1..256".into());
        }
        if self.sh_rows > 8 {
            return Err("sh_rows must be 0..8".into());
        }
        if self.head_group == 0 || self.head_group > 32 {
            return Err("head_group must be 1..32".into());
        }
        if ![1, 2, 4, 8].contains(&self.split_k) {
            return Err("split_k must be 1/2/4/8".into());
        }
        Ok(())
    }
}
pub struct Builder<'a> {
    pub machine: &'a mut Machine,
    pub policy: Policy,
}
impl Builder<'_> {
    fn empty_wave(&self) -> Wave {
        vec![vec![]; self.machine.timing.hw.sms]
    }
    fn submit(&mut self, wave: Wave) -> Result<()> {
        self.machine.wave(wave)
    }
    pub fn gemm(&mut self, a: Matrix, b: Matrix, c: Matrix) -> Result<()> {
        self.gemm_many(&[(a, b, c)])
    }
    /// Schedule independent GEMMs together. A stage must not read another job's output.
    pub fn gemm_many(&mut self, jobs: &[(Matrix, Matrix, Matrix)]) -> Result<()> {
        if jobs.len() > 32 {
            return Err("independent GEMM batch exceeds 32".into());
        }
        self.policy.validate()?;
        if jobs.len() > 1
            && jobs.iter().enumerate().any(|(i, (_, _, c))| {
                jobs.iter()
                    .enumerate()
                    .any(|(j, (a, b, _))| i != j && (c.tensor == a.tensor || c.tensor == b.tensor))
            })
        {
            return Err("batched GEMMs have cross-job tensor dependencies".into());
        }
        let policy = self.policy;
        let mut plans = vec![];
        let mut scratch = vec![];
        for &(a, b, c) in jobs {
            if a.rows == 0
                || a.cols == 0
                || b.cols == 0
                || a.cols != b.rows
                || a.rows != c.rows
                || b.cols != c.cols
            {
                return Err("GEMM shape mismatch".into());
            }
            self.policy.validate()?;
            let packed = if self.policy.pack_transpose && b.cs > 1 {
                let matrix = self.pack(b)?;
                scratch.push(matrix.tensor);
                Some(matrix)
            } else {
                None
            };
            let b = packed.unwrap_or(b);
            let requested = if a.rows == 1 {
                policy.split_k.min(a.cols)
            } else {
                1
            };
            let chunk = a.cols.div_ceil(requested);
            let parts = a.cols.div_ceil(chunk);
            let partial = if parts > 1 {
                Some(Matrix::plain(
                    self.machine.empty("split_partials", parts * c.cols)?,
                    parts,
                    c.cols,
                ))
            } else {
                None
            };
            if let Some(p) = partial {
                scratch.push(p.tensor);
            }
            plans.push((a, b, c, parts, chunk, partial));
        }
        if policy.sh_rows > 0 && plans.iter().all(|p| p.0.rows > 1) {
            self.gemm_shared(&plans.iter().map(|p| (p.0, p.1, p.2)).collect::<Vec<_>>())?;
            for id in scratch.into_iter().rev() {
                self.machine.release_last(id)?;
            }
            return Ok(());
        }
        let mut wave = self.empty_wave();
        let mut sm = 0;
        for &(a, b, c, parts, chunk, partial) in &plans {
            for row in (0..c.rows).step_by(policy.m) {
                for col in (0..c.cols).step_by(policy.n) {
                    for part in 0..parts {
                        let begin = part * chunk;
                        let finish = ((part + 1) * chunk).min(a.cols);
                        if begin >= finish {
                            continue;
                        }
                        let m = (c.rows - row).min(policy.m);
                        let n = (c.cols - col).min(policy.n);
                        let aa = m * n;
                        let bb = aa + m * policy.k;
                        if bb + policy.k * n > self.machine.timing.hw.rf_kib * 256 {
                            return Err("GEMM tile working set exceeds RF capacity".into());
                        }
                        let chunks = (finish - begin).div_ceil(policy.k);
                        let instructions = 2
                            + 2 * chunks
                            + if policy.vector_gemv && m == 1 {
                                finish - begin
                            } else {
                                chunks
                            };
                        if wave.iter().map(Vec::len).sum::<usize>() + instructions > 16384 {
                            self.submit(wave)?;
                            wave = self.empty_wave();
                            sm = 0;
                        }
                        let program = &mut wave[sm];
                        program.push(Instr::Fill {
                            dst: 0,
                            len: m * n,
                            value: 0.,
                        });
                        for kk in (begin..finish).step_by(policy.k) {
                            let k = (finish - kk).min(policy.k);
                            program.push(Instr::Load {
                                tensor: a.tensor,
                                global: a.view(row, kk, m, k),
                                local: View::contiguous(aa, m, k),
                            });
                            program.push(Instr::Load {
                                tensor: b.tensor,
                                global: b.view(kk, col, k, n),
                                local: View::contiguous(bb, k, n),
                            });
                            if policy.vector_gemv && m == 1 {
                                for step in 0..k {
                                    program.push(Instr::Vector {
                                        kind: Op::Fma,
                                        dst: 0,
                                        len: n,
                                        a: Arg::scalar(aa + step),
                                        b: Arg::reg(bb + step * n),
                                    });
                                }
                            } else {
                                program.push(Instr::Mma {
                                    a: aa,
                                    b: bb,
                                    c: 0,
                                    m,
                                    n,
                                    k,
                                });
                            }
                        }
                        program.push(Instr::Store {
                            tensor: partial.unwrap_or(c).tensor,
                            global: if let Some(p) = partial {
                                p.view(part, col, m, n)
                            } else {
                                c.view(row, col, m, n)
                            },
                            local: View::contiguous(0, m, n),
                        });
                        sm += 1;
                        if sm == wave.len() {
                            self.submit(wave)?;
                            wave = self.empty_wave();
                            sm = 0;
                        }
                    }
                }
            }
        }
        if sm > 0 {
            self.submit(wave)?;
        }
        for &(_, _, c, _, _, partial) in &plans {
            if let Some(p) = partial {
                self.sum_partials(p, c)?;
            }
        }
        for id in scratch.into_iter().rev() {
            self.machine.release_last(id)?;
        }
        Ok(())
    }
    // K-major traversal keeps several output tiles in SH. A B panel is loaded
    // once and reused by those row tiles; all accumulator spills are explicit.
    fn gemm_shared(&mut self, jobs: &[(Matrix, Matrix, Matrix)]) -> Result<()> {
        let p = self.policy;
        if p.k * p.n + p.sh_rows * p.m * p.n > self.machine.timing.hw.sh_kib * 256 {
            return Err("shared GEMM panel/accumulators exceed SH capacity".into());
        }
        if p.m * p.n + p.m * p.k + if p.sh_direct { 0 } else { p.k * p.n }
            > self.machine.timing.hw.rf_kib * 256
        {
            return Err("shared GEMM staging exceeds RF capacity".into());
        }
        let mut wave = self.empty_wave();
        let mut sm = 0;
        let mut wave_instructions = 0;
        for &(a, b, c) in jobs {
            for row in (0..c.rows).step_by(p.m * p.sh_rows) {
                let groups = (c.rows - row).div_ceil(p.m).min(p.sh_rows);
                let instructions =
                    a.cols.div_ceil(p.k) * (1 + if p.sh_direct { 4 } else { 5 } * groups);
                if instructions > 16384 {
                    return Err(
                        "shared GEMM task exceeds wave limit; increase K block or reduce sh_rows"
                            .into(),
                    );
                }
                for col in (0..c.cols).step_by(p.n) {
                    if wave_instructions + instructions > 16384 {
                        self.submit(wave)?;
                        wave = self.empty_wave();
                        sm = 0;
                        wave_instructions = 0;
                    }
                    let n = (c.cols - col).min(p.n);
                    for kk in (0..a.cols).step_by(p.k) {
                        let k = (a.cols - kk).min(p.k);
                        wave[sm].push(Instr::LoadShared {
                            tensor: b.tensor,
                            global: b.view(kk, col, k, n),
                            local: View::contiguous(0, k, n),
                        });
                        for group in 0..groups {
                            let r = row + group * p.m;
                            let m = (c.rows - r).min(p.m);
                            let aa = m * n;
                            let bb = aa + m * p.k;
                            let acc = View::contiguous(p.k * p.n + group * p.m * p.n, m, n);
                            if kk == 0 {
                                wave[sm].push(Instr::Fill {
                                    dst: 0,
                                    len: m * n,
                                    value: 0.,
                                });
                            } else {
                                wave[sm].push(Instr::SharedRead {
                                    shared: acc,
                                    local: View::contiguous(0, m, n),
                                });
                            }
                            wave[sm].push(Instr::Load {
                                tensor: a.tensor,
                                global: a.view(r, kk, m, k),
                                local: View::contiguous(aa, m, k),
                            });
                            if p.sh_direct {
                                wave[sm].push(Instr::MmaShared {
                                    a: aa,
                                    b: View::contiguous(0, k, n),
                                    c: 0,
                                    m,
                                    n,
                                    k,
                                });
                            } else {
                                wave[sm].push(Instr::SharedRead {
                                    shared: View::contiguous(0, k, n),
                                    local: View::contiguous(bb, k, n),
                                });
                                wave[sm].push(Instr::Mma {
                                    a: aa,
                                    b: bb,
                                    c: 0,
                                    m,
                                    n,
                                    k,
                                });
                            }
                            if kk + k == a.cols {
                                wave[sm].push(Instr::Store {
                                    tensor: c.tensor,
                                    global: c.view(r, col, m, n),
                                    local: View::contiguous(0, m, n),
                                });
                            } else {
                                wave[sm].push(Instr::SharedWrite {
                                    shared: acc,
                                    local: View::contiguous(0, m, n),
                                });
                            }
                        }
                    }
                    wave_instructions += instructions;
                    sm += 1;
                    if sm == wave.len() {
                        self.submit(wave)?;
                        wave = self.empty_wave();
                        sm = 0;
                        wave_instructions = 0;
                    }
                }
            }
        }
        if sm > 0 {
            self.submit(wave)?;
        }
        Ok(())
    }
    fn sum_partials(&mut self, source: Matrix, dest: Matrix) -> Result<()> {
        let mut wave = self.empty_wave();
        let mut sm = 0;
        for col in (0..dest.cols).step_by(256) {
            let n = (dest.cols - col).min(256);
            wave[sm].push(Instr::Load {
                tensor: source.tensor,
                global: source.view(0, col, 1, n),
                local: View::contiguous(0, 1, n),
            });
            for part in 1..source.rows {
                wave[sm].push(Instr::Load {
                    tensor: source.tensor,
                    global: source.view(part, col, 1, n),
                    local: View::contiguous(n, 1, n),
                });
                wave[sm].push(Instr::Vector {
                    kind: Op::Add,
                    dst: 0,
                    len: n,
                    a: Arg::reg(0),
                    b: Arg::reg(n),
                });
            }
            wave[sm].push(Instr::Store {
                tensor: dest.tensor,
                global: dest.view(0, col, 1, n),
                local: View::contiguous(0, 1, n),
            });
            sm += 1;
            if sm == wave.len() {
                self.submit(wave)?;
                wave = self.empty_wave();
                sm = 0;
            }
        }
        if sm > 0 {
            self.submit(wave)?;
        }
        Ok(())
    }
    // Tiled transpose: read in the contiguous source direction, transpose the
    // RF view on store. Both HBM passes and RF port traffic are simulated.
    fn pack(&mut self, source: Matrix) -> Result<Matrix> {
        let id = self.machine.empty("packed", source.rows * source.cols)?;
        let dest = Matrix::plain(id, source.rows, source.cols);
        self.pack_into(source, dest)?;
        Ok(dest)
    }
    pub(crate) fn pack_into(&mut self, source: Matrix, dest: Matrix) -> Result<()> {
        if (source.rows, source.cols) != (dest.rows, dest.cols) {
            return Err("transpose destination shape mismatch".into());
        }
        let mut wave = self.empty_wave();
        let mut sm = 0;
        for r in (0..source.rows).step_by(16) {
            for c in (0..source.cols).step_by(16) {
                let nr = (source.rows - r).min(16);
                let nc = (source.cols - c).min(16);
                wave[sm].push(Instr::Load {
                    tensor: source.tensor,
                    global: View {
                        base: source.base + r * source.rs + c * source.cs,
                        rows: nc,
                        cols: nr,
                        row_stride: source.cs,
                        col_stride: source.rs,
                    },
                    local: View::contiguous(0, nc, nr),
                });
                wave[sm].push(Instr::Store {
                    tensor: dest.tensor,
                    global: dest.view(r, c, nr, nc),
                    local: View {
                        base: 0,
                        rows: nr,
                        cols: nc,
                        row_stride: 1,
                        col_stride: nr,
                    },
                });
                sm += 1;
                if sm == wave.len() {
                    self.submit(wave)?;
                    wave = self.empty_wave();
                    sm = 0;
                }
            }
        }
        if sm > 0 {
            self.submit(wave)?;
        }
        Ok(())
    }
    pub fn copy(&mut self, src: Matrix, dst: Matrix) -> Result<()> {
        if (src.rows, src.cols) != (dst.rows, dst.cols) {
            return Err("copy shape mismatch".into());
        }
        let mut wave = self.empty_wave();
        let mut sm = 0;
        for r in 0..src.rows {
            for c in (0..src.cols).step_by(4096) {
                let n = (src.cols - c).min(4096);
                wave[sm].push(Instr::Load {
                    tensor: src.tensor,
                    global: src.view(r, c, 1, n),
                    local: View::contiguous(0, 1, n),
                });
                wave[sm].push(Instr::Store {
                    tensor: dst.tensor,
                    global: dst.view(r, c, 1, n),
                    local: View::contiguous(0, 1, n),
                });
                sm += 1;
                if sm == wave.len() {
                    self.submit(wave)?;
                    wave = self.empty_wave();
                    sm = 0;
                }
            }
        }
        if sm > 0 {
            self.submit(wave)?;
        }
        Ok(())
    }
    fn rows(&mut self, rows: usize, mut make: impl FnMut(usize) -> Vec<Instr>) -> Result<()> {
        for r in (0..rows).step_by(self.machine.timing.hw.sms) {
            let mut wave = self.empty_wave();
            for (i, p) in wave.iter_mut().enumerate() {
                if r + i < rows {
                    *p = make(r + i);
                }
            }
            self.submit(wave)?;
        }
        Ok(())
    }
    fn load(tensor: usize, base: usize, len: usize, dst: usize) -> Instr {
        Instr::Load {
            tensor,
            global: View::contiguous(base, 1, len),
            local: View::contiguous(dst, 1, len),
        }
    }
    fn store(tensor: usize, base: usize, len: usize, src: usize) -> Instr {
        Instr::Store {
            tensor,
            global: View::contiguous(base, 1, len),
            local: View::contiguous(src, 1, len),
        }
    }
    fn v(kind: Op, dst: usize, len: usize, a: Arg, b: Arg) -> Instr {
        Instr::Vector {
            kind,
            dst,
            len,
            a,
            b,
        }
    }
    pub fn norm(
        &mut self,
        x: usize,
        y: usize,
        g: usize,
        b: usize,
        rows: usize,
        d: usize,
    ) -> Result<()> {
        use Arg::Imm as I;
        let r = Arg::reg;
        let s = Arg::scalar;
        self.rows(rows, |row| {
            vec![
                Self::load(x, row * d, d, 0),
                Self::load(g, 0, d, d),
                Self::load(b, 0, d, 2 * d),
                Instr::Reduce {
                    scratch: 4 * d + 1,
                    dst: 4 * d,
                    src: 0,
                    len: d,
                    max: false,
                },
                Self::v(Op::Div, 4 * d, 1, s(4 * d), I { value: d as f32 }),
                Self::v(Op::Sub, 0, d, r(0), s(4 * d)),
                Self::v(Op::Square, 3 * d, d, r(0), I { value: 0. }),
                Instr::Reduce {
                    scratch: 4 * d + 1,
                    dst: 4 * d,
                    src: 3 * d,
                    len: d,
                    max: false,
                },
                Self::v(Op::Div, 4 * d, 1, s(4 * d), I { value: d as f32 }),
                Self::v(Op::Add, 4 * d, 1, s(4 * d), I { value: 1e-5 }),
                Self::v(Op::Rsqrt, 4 * d, 1, s(4 * d), I { value: 0. }),
                Self::v(Op::Mul, 0, d, r(0), s(4 * d)),
                Self::v(Op::Mul, 0, d, r(0), r(d)),
                Self::v(Op::Add, 0, d, r(0), r(2 * d)),
                Self::store(y, row * d, d, 0),
            ]
        })
    }
    pub fn bias(&mut self, x: usize, b: usize, rows: usize, d: usize) -> Result<()> {
        self.rows(rows, |row| {
            vec![
                Self::load(x, row * d, d, 0),
                Self::load(b, 0, d, d),
                Self::v(Op::Add, 0, d, Arg::reg(0), Arg::reg(d)),
                Self::store(x, row * d, d, 0),
            ]
        })
    }
    pub fn residual(&mut self, x: usize, y: usize, rows: usize, d: usize) -> Result<()> {
        self.rows(rows, |row| {
            vec![
                Self::load(x, row * d, d, 0),
                Self::load(y, row * d, d, d),
                Self::v(Op::Add, 0, d, Arg::reg(0), Arg::reg(d)),
                Self::store(y, row * d, d, 0),
            ]
        })
    }
    pub fn gelu(&mut self, x: usize, rows: usize, d: usize) -> Result<()> {
        use Arg::Imm as I;
        let r = Arg::reg;
        // Chunking keeps 3*chunk RF elements below even the 32 KiB menu.
        self.rows(rows * d.div_ceil(1024), |task| {
            let row = task / d.div_ceil(1024);
            let start = (task % d.div_ceil(1024)) * 1024;
            let n = (d - start).min(1024);
            vec![
                Self::load(x, row * d + start, n, 0),
                Self::v(Op::Square, n, n, r(0), I { value: 0. }),
                Self::v(Op::Mul, n, n, r(n), r(0)),
                Self::v(Op::Mul, n, n, r(n), I { value: 0.044715 }),
                Self::v(Op::Add, n, n, r(n), r(0)),
                Self::v(
                    Op::Mul,
                    n,
                    n,
                    r(n),
                    I {
                        value: std::f32::consts::FRAC_2_PI.sqrt(),
                    },
                ),
                Self::v(Op::Tanh, n, n, r(n), I { value: 0. }),
                Self::v(Op::Add, n, n, r(n), I { value: 1. }),
                Self::v(Op::Mul, n, n, r(n), I { value: 0.5 }),
                Self::v(Op::Mul, 0, n, r(0), r(n)),
                Self::store(x, row * d + start, n, 0),
            ]
        })
    }
    pub fn softmax(
        &mut self,
        x: usize,
        rows: usize,
        context: usize,
        history: usize,
        head: usize,
    ) -> Result<()> {
        self.softmax_many(&[x], rows, context, history, head)
    }
    pub fn softmax_many(
        &mut self,
        xs: &[usize],
        rows: usize,
        context: usize,
        history: usize,
        head: usize,
    ) -> Result<()> {
        use Arg::Imm as I;
        let r = Arg::reg;
        let s = Arg::scalar;
        self.rows(rows * xs.len(), |i| {
            let x = xs[i / rows];
            let row = i % rows;
            let len = history + row + 1;
            vec![
                Self::load(x, row * context, len, 0),
                Self::v(
                    Op::Mul,
                    0,
                    len,
                    r(0),
                    I {
                        value: 1. / (head as f32).sqrt(),
                    },
                ),
                Instr::Reduce {
                    scratch: context + 1,
                    dst: context,
                    src: 0,
                    len,
                    max: true,
                },
                Self::v(Op::Sub, 0, len, r(0), s(context)),
                Self::v(Op::Exp, 0, len, r(0), I { value: 0. }),
                Instr::Reduce {
                    scratch: context + 1,
                    dst: context,
                    src: 0,
                    len,
                    max: false,
                },
                Self::v(Op::Div, 0, len, r(0), s(context)),
                Self::store(x, row * context, len, 0),
            ]
        })?;
        if rows > 1 {
            self.rows((rows - 1) * xs.len(), |i| {
                let x = xs[i / (rows - 1)];
                let row = i % (rows - 1);
                let start = history + row + 1;
                let len = context - start;
                vec![
                    Instr::Fill {
                        dst: 0,
                        len,
                        value: 0.,
                    },
                    Self::store(x, row * context + start, len, 0),
                ]
            })?;
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct Outputs {
    pub hidden: Vec<Vec<f32>>,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}
pub fn run(machine: &mut Machine, fixture: &Fixture, decode: bool, tile: usize) -> Result<Outputs> {
    run_policy(machine, fixture, decode, Policy::square(tile))
}
pub fn run_policy(
    machine: &mut Machine,
    fixture: &Fixture,
    decode: bool,
    policy: Policy,
) -> Result<Outputs> {
    policy.validate()?;
    let shape = fixture.shape;
    let d = shape.d;
    let f = shape.f;
    let rows = if decode { 1 } else { shape.prefill };
    let steps = if decode { shape.steps } else { 1 };
    let mut weights = vec![];
    let mut caches = vec![];
    for (i, l) in fixture.layers.iter().enumerate() {
        let mut ids = vec![];
        for (name, data) in [
            ("qkv", &l.qkv),
            ("qb", &l.qb),
            ("out", &l.out),
            ("ob", &l.ob),
            ("up", &l.up),
            ("ub", &l.ub),
            ("down", &l.down),
            ("db", &l.db),
            ("g1", &l.g1),
            ("b1", &l.b1),
            ("g2", &l.g2),
            ("b2", &l.b2),
        ] {
            ids.push(machine.bind_input(&format!("layer{i}.{name}"), data, data.len())?);
        }
        weights.push(ids);
        let capacity = if decode { shape.history + steps } else { rows };
        let (k, v) = if decode {
            (
                machine.bind_input(&format!("layer{i}.past_k"), &l.past_k, capacity * d)?,
                machine.bind_input(&format!("layer{i}.past_v"), &l.past_v, capacity * d)?,
            )
        } else {
            (
                machine.empty("K", capacity * d)?,
                machine.empty("V", capacity * d)?,
            )
        };
        let kt = if policy.persistent_keys {
            let kt = machine.empty("persistent_KT", capacity * d)?;
            if decode {
                // Initial cache format conversion is part of measured execution.
                Builder { machine, policy }.pack_into(
                    Matrix {
                        tensor: k,
                        rows: d,
                        cols: shape.history,
                        base: 0,
                        rs: 1,
                        cs: d,
                    },
                    Matrix {
                        tensor: kt,
                        rows: d,
                        cols: shape.history,
                        base: 0,
                        rs: capacity,
                        cs: 1,
                    },
                )?;
            }
            Some(kt)
        } else {
            None
        };
        caches.push((k, v, kt, capacity));
    }
    let mut outputs = Outputs {
        hidden: vec![],
        keys: vec![vec![]; shape.layers],
        values: vec![vec![]; shape.layers],
    };
    for step in 0..steps {
        // Future teacher-forced inputs do not exist in machine memory until
        // all preceding outputs and KV stores have completed at wave fences.
        let input = if decode {
            fixture.inputs[step].clone()
        } else {
            fixture.prompt.clone()
        };
        let mut x = machine.bind_input(&format!("input{step}"), &input, input.len())?;
        let history = if decode { shape.history + step } else { 0 };
        let context = history + rows;
        for layer in 0..shape.layers {
            let w = &weights[layer];
            let (kcache, vcache, kt, capacity) = caches[layer];
            let ln = machine.empty("ln", rows * d)?;
            let qkv = machine.empty("qkv", rows * 3 * d)?;
            let att = machine.empty("attention", rows * d)?;
            let out = machine.empty("projected", rows * d)?;
            let ln2 = machine.empty("ln2", rows * d)?;
            let up = machine.empty("up", rows * f)?;
            let down = machine.empty("down", rows * d)?;
            let mut b = Builder { machine, policy };
            b.norm(x, ln, w[8], w[9], rows, d)?;
            b.gemm(
                Matrix::plain(ln, rows, d),
                Matrix::plain(w[0], d, 3 * d),
                Matrix::plain(qkv, rows, 3 * d),
            )?;
            b.bias(qkv, w[1], rows, 3 * d)?;
            for (offset, cache) in [(d, kcache), (2 * d, vcache)] {
                b.copy(
                    Matrix {
                        tensor: qkv,
                        rows,
                        cols: d,
                        base: offset,
                        rs: 3 * d,
                        cs: 1,
                    },
                    Matrix {
                        tensor: cache,
                        rows,
                        cols: d,
                        base: history * d,
                        rs: d,
                        cs: 1,
                    },
                )?;
            }
            if let Some(kt) = kt {
                b.pack_into(
                    Matrix {
                        tensor: kcache,
                        rows: d,
                        cols: rows,
                        base: history * d,
                        rs: 1,
                        cs: d,
                    },
                    Matrix {
                        tensor: kt,
                        rows: d,
                        cols: rows,
                        base: history,
                        rs: capacity,
                        cs: 1,
                    },
                )?;
            }
            let key_view = |h: usize, hd: usize| {
                if let Some(kt) = kt {
                    Matrix {
                        tensor: kt,
                        rows: hd,
                        cols: context,
                        base: h * hd * capacity,
                        rs: capacity,
                        cs: 1,
                    }
                } else {
                    Matrix {
                        tensor: kcache,
                        rows: hd,
                        cols: context,
                        base: h * hd,
                        rs: 1,
                        cs: d,
                    }
                }
            };
            if policy.head_group > 1 {
                let hd = d / shape.heads;
                for first in (0..shape.heads).step_by(policy.head_group) {
                    let mut scores = vec![];
                    let mut qk = vec![];
                    let mut av = vec![];
                    for h in first..(first + policy.head_group).min(shape.heads) {
                        let score = b.machine.empty("score", rows * context)?;
                        scores.push(score);
                        qk.push((
                            Matrix {
                                tensor: qkv,
                                rows,
                                cols: hd,
                                base: h * hd,
                                rs: 3 * d,
                                cs: 1,
                            },
                            key_view(h, hd),
                            Matrix::plain(score, rows, context),
                        ));
                        av.push((
                            Matrix::plain(score, rows, context),
                            Matrix {
                                tensor: vcache,
                                rows: context,
                                cols: hd,
                                base: h * hd,
                                rs: d,
                                cs: 1,
                            },
                            Matrix {
                                tensor: att,
                                rows,
                                cols: hd,
                                base: h * hd,
                                rs: d,
                                cs: 1,
                            },
                        ));
                    }
                    b.gemm_many(&qk)?;
                    b.softmax_many(&scores, rows, context, history, hd)?;
                    b.gemm_many(&av)?;
                }
            } else {
                for h in 0..shape.heads {
                    let hd = d / shape.heads;
                    let score = b.machine.empty("score", rows * context)?;
                    b.gemm(
                        Matrix {
                            tensor: qkv,
                            rows,
                            cols: hd,
                            base: h * hd,
                            rs: 3 * d,
                            cs: 1,
                        },
                        key_view(h, hd),
                        Matrix::plain(score, rows, context),
                    )?;
                    b.softmax(score, rows, context, history, hd)?;
                    b.gemm(
                        Matrix::plain(score, rows, context),
                        Matrix {
                            tensor: vcache,
                            rows: context,
                            cols: hd,
                            base: h * hd,
                            rs: d,
                            cs: 1,
                        },
                        Matrix {
                            tensor: att,
                            rows,
                            cols: hd,
                            base: h * hd,
                            rs: d,
                            cs: 1,
                        },
                    )?;
                }
            }
            b.gemm(
                Matrix::plain(att, rows, d),
                Matrix::plain(w[2], d, d),
                Matrix::plain(out, rows, d),
            )?;
            b.bias(out, w[3], rows, d)?;
            b.residual(x, out, rows, d)?;
            b.norm(out, ln2, w[10], w[11], rows, d)?;
            b.gemm(
                Matrix::plain(ln2, rows, d),
                Matrix::plain(w[4], d, f),
                Matrix::plain(up, rows, f),
            )?;
            b.bias(up, w[5], rows, f)?;
            b.gelu(up, rows, f)?;
            b.gemm(
                Matrix::plain(up, rows, f),
                Matrix::plain(w[6], f, d),
                Matrix::plain(down, rows, d),
            )?;
            b.bias(down, w[7], rows, d)?;
            b.residual(out, down, rows, d)?;
            x = down;
            if b.machine.functional {
                outputs.keys[layer]
                    .extend_from_slice(&b.machine.tensors[kcache].data[history * d..context * d]);
                outputs.values[layer]
                    .extend_from_slice(&b.machine.tensors[vcache].data[history * d..context * d]);
            }
        }
        if machine.functional {
            outputs.hidden.push(machine.tensors[x].data.clone());
        }
        // Explicit STEP.COMMIT control issue, with no outstanding operations.
        use crate::submission::{Checkpoint, OutputView};
        machine.checkpoint(Checkpoint {
            hidden: OutputView {
                tensor: x,
                base: 0,
                len: rows * d,
            },
            keys: caches
                .iter()
                .map(|&(k, _, _, _)| OutputView {
                    tensor: k,
                    base: history * d,
                    len: rows * d,
                })
                .collect(),
            values: caches
                .iter()
                .map(|&(_, v, _, _)| OutputView {
                    tensor: v,
                    base: history * d,
                    len: rows * d,
                })
                .collect(),
        })?;
    }
    Ok(outputs)
}
