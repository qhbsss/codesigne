use serde::{Deserialize, Serialize};

// Views use FP32 element addresses. Stride zero expresses broadcast/transpose.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    pub base: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
}
impl View {
    pub fn contiguous(base: usize, rows: usize, cols: usize) -> Self {
        Self {
            base,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
        }
    }
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }
    pub fn at(&self, i: usize) -> usize {
        self.base + (i / self.cols) * self.row_stride + (i % self.cols) * self.col_stride
    }
    pub fn end(&self) -> Option<usize> {
        self.base
            .checked_add(self.rows.checked_sub(1)?.checked_mul(self.row_stride)?)?
            .checked_add(self.cols.checked_sub(1)?.checked_mul(self.col_stride)?)?
            .checked_add(1)
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Add,
    Fma,
    Sub,
    Mul,
    Div,
    Exp,
    Tanh,
    Rsqrt,
    Square,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Arg {
    Reg { base: usize, stride: usize },
    Imm { value: f32 },
}
impl Arg {
    pub fn reg(base: usize) -> Self {
        Self::Reg { base, stride: 1 }
    }
    pub fn scalar(base: usize) -> Self {
        Self::Reg { base, stride: 0 }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Instr {
    Load {
        tensor: usize,
        global: View,
        local: View,
    },
    Store {
        tensor: usize,
        global: View,
        local: View,
    },
    LoadShared {
        tensor: usize,
        global: View,
        local: View,
    },
    StoreShared {
        tensor: usize,
        global: View,
        local: View,
    },
    SharedRead {
        shared: View,
        local: View,
    },
    SharedWrite {
        shared: View,
        local: View,
    },
    Fill {
        dst: usize,
        len: usize,
        value: f32,
    },
    Vector {
        kind: Op,
        dst: usize,
        len: usize,
        a: Arg,
        b: Arg,
    },
    Reduce {
        scratch: usize,
        dst: usize,
        src: usize,
        len: usize,
        max: bool,
    },
    MmaShared {
        a: usize,
        b: View,
        c: usize,
        m: usize,
        n: usize,
        k: usize,
    },
    Mma {
        a: usize,
        b: usize,
        c: usize,
        m: usize,
        n: usize,
        k: usize,
    },
}
pub type Wave = Vec<Vec<Instr>>;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub model: String,
    pub hardware: crate::hardware::Hardware,
    pub tensors: Vec<Vec<f32>>,
    pub waves: Vec<Wave>,
}
