# Public HBM ABI (v0.7)

The two cases use separate deterministic layouts in a 2 GiB FP32 HBM address space. Every offset in the tables is an **FP32 element offset**, not a byte address. Symbols start on 64-byte boundaries and are row-major. The `scratch` symbol covers unused HBM and can be reused after a value is no longer live.

For a symbol with base `B` and shape `[d0,d1,...]`, flatten the indices in row-major order and add the result to `B`. An assembly HBM view with `shape` and `strides` selects a noncontiguous tile; `count` is its logical number of FP32 values. `abi.py` in the starter package is the executable source of truth.

`M1_P1` has two independent batch elements. It must write both prompt results and every layer’s prompt K/V before `STEP.COMMIT {"step":0}` releases `input/step0`. The first new position is then computed for **both** batch elements. `M2_D1` starts with read-only K/V for 128 historical positions and releases `input/stepN` only after earlier steps commit. Every new hidden and K/V value must be written once.

For a K/V tensor shaped `[batch, heads, positions, head_width]`, the element `(b,h,t,c)` is at `B + (((b*heads + h)*positions + t)*head_width + c)`. P1 new K/V uses `[2,8,65,32]`; D1 historical K/V uses `[1,4,128,32]`; D1 new K/V uses `[1,4,8,32]`. The different position strides matter when copying K/V into attention tiles.

## M1_P1

| Symbol | FP32 offset | Shape | Access | Release step |
| --- | ---: | --- | --- | ---: |
| `layer0/ln1_g` | 0 | 256 | read only | 0 |
| `layer0/ln1_b` | 256 | 256 | read only | 0 |
| `layer0/wqkv` | 512 | 256 × 768 | read only | 0 |
| `layer0/wo` | 197,120 | 256 × 256 | read only | 0 |
| `layer0/ln2_g` | 262,656 | 256 | read only | 0 |
| `layer0/ln2_b` | 262,912 | 256 | read only | 0 |
| `layer0/w1` | 263,168 | 256 × 1024 | read only | 0 |
| `layer0/b1` | 525,312 | 1024 | read only | 0 |
| `layer0/w2` | 526,336 | 1024 × 256 | read only | 0 |
| `layer0/b2` | 788,480 | 256 | read only | 0 |
| `layer1/ln1_g` | 788,736 | 256 | read only | 0 |
| `layer1/ln1_b` | 788,992 | 256 | read only | 0 |
| `layer1/wqkv` | 789,248 | 256 × 768 | read only | 0 |
| `layer1/wo` | 985,856 | 256 × 256 | read only | 0 |
| `layer1/ln2_g` | 1,051,392 | 256 | read only | 0 |
| `layer1/ln2_b` | 1,051,648 | 256 | read only | 0 |
| `layer1/w1` | 1,051,904 | 256 × 1024 | read only | 0 |
| `layer1/b1` | 1,314,048 | 1024 | read only | 0 |
| `layer1/w2` | 1,315,072 | 1024 × 256 | read only | 0 |
| `layer1/b2` | 1,577,216 | 256 | read only | 0 |
| `layer2/ln1_g` | 1,577,472 | 256 | read only | 0 |
| `layer2/ln1_b` | 1,577,728 | 256 | read only | 0 |
| `layer2/wqkv` | 1,577,984 | 256 × 768 | read only | 0 |
| `layer2/wo` | 1,774,592 | 256 × 256 | read only | 0 |
| `layer2/ln2_g` | 1,840,128 | 256 | read only | 0 |
| `layer2/ln2_b` | 1,840,384 | 256 | read only | 0 |
| `layer2/w1` | 1,840,640 | 256 × 1024 | read only | 0 |
| `layer2/b1` | 2,102,784 | 1024 | read only | 0 |
| `layer2/w2` | 2,103,808 | 1024 × 256 | read only | 0 |
| `layer2/b2` | 2,365,952 | 256 | read only | 0 |
| `input/prompt` | 2,366,208 | 2 × 64 × 256 | read only | 0 |
| `input/step0` | 2,398,976 | 2 × 256 | read only | 1 |
| `output/hidden` | 2,399,488 | 2 × 65 × 256 | writable | 0 |
| `layer0/new_k` | 2,432,768 | 2 × 8 × 65 × 32 | writable | 0 |
| `layer0/new_v` | 2,466,048 | 2 × 8 × 65 × 32 | writable | 0 |
| `layer1/new_k` | 2,499,328 | 2 × 8 × 65 × 32 | writable | 0 |
| `layer1/new_v` | 2,532,608 | 2 × 8 × 65 × 32 | writable | 0 |
| `layer2/new_k` | 2,565,888 | 2 × 8 × 65 × 32 | writable | 0 |
| `layer2/new_v` | 2,599,168 | 2 × 8 × 65 × 32 | writable | 0 |
| `scratch` | 2,632,448 | 534238464 | writable | 0 |

Total addressed space: 2,147,483,648 bytes (including scratch).

## M2_D1

| Symbol | FP32 offset | Shape | Access | Release step |
| --- | ---: | --- | --- | ---: |
| `layer0/ln1_g` | 0 | 128 | read only | 0 |
| `layer0/ln1_b` | 128 | 128 | read only | 0 |
| `layer0/wqkv` | 256 | 128 × 384 | read only | 0 |
| `layer0/wo` | 49,408 | 128 × 128 | read only | 0 |
| `layer0/ln2_g` | 65,792 | 128 | read only | 0 |
| `layer0/ln2_b` | 65,920 | 128 | read only | 0 |
| `layer0/w1` | 66,048 | 128 × 512 | read only | 0 |
| `layer0/b1` | 131,584 | 512 | read only | 0 |
| `layer0/w2` | 132,096 | 512 × 128 | read only | 0 |
| `layer0/b2` | 197,632 | 128 | read only | 0 |
| `layer0/history_k` | 197,760 | 1 × 4 × 128 × 32 | read only | 0 |
| `layer0/history_v` | 214,144 | 1 × 4 × 128 × 32 | read only | 0 |
| `layer1/ln1_g` | 230,528 | 128 | read only | 0 |
| `layer1/ln1_b` | 230,656 | 128 | read only | 0 |
| `layer1/wqkv` | 230,784 | 128 × 384 | read only | 0 |
| `layer1/wo` | 279,936 | 128 × 128 | read only | 0 |
| `layer1/ln2_g` | 296,320 | 128 | read only | 0 |
| `layer1/ln2_b` | 296,448 | 128 | read only | 0 |
| `layer1/w1` | 296,576 | 128 × 512 | read only | 0 |
| `layer1/b1` | 362,112 | 512 | read only | 0 |
| `layer1/w2` | 362,624 | 512 × 128 | read only | 0 |
| `layer1/b2` | 428,160 | 128 | read only | 0 |
| `layer1/history_k` | 428,288 | 1 × 4 × 128 × 32 | read only | 0 |
| `layer1/history_v` | 444,672 | 1 × 4 × 128 × 32 | read only | 0 |
| `input/step0` | 461,056 | 1 × 128 | read only | 0 |
| `input/step1` | 461,184 | 1 × 128 | read only | 1 |
| `input/step2` | 461,312 | 1 × 128 | read only | 2 |
| `input/step3` | 461,440 | 1 × 128 | read only | 3 |
| `input/step4` | 461,568 | 1 × 128 | read only | 4 |
| `input/step5` | 461,696 | 1 × 128 | read only | 5 |
| `input/step6` | 461,824 | 1 × 128 | read only | 6 |
| `input/step7` | 461,952 | 1 × 128 | read only | 7 |
| `output/hidden` | 462,080 | 1 × 8 × 128 | writable | 0 |
| `layer0/new_k` | 463,104 | 1 × 4 × 8 × 32 | writable | 0 |
| `layer0/new_v` | 464,128 | 1 × 4 × 8 × 32 | writable | 0 |
| `layer1/new_k` | 465,152 | 1 × 4 × 8 × 32 | writable | 0 |
| `layer1/new_v` | 466,176 | 1 × 4 × 8 × 32 | writable | 0 |
| `scratch` | 467,200 | 536403712 | writable | 0 |

Total addressed space: 2,147,483,648 bytes (including scratch).

The published baseline programs use exactly these offsets. If you change the compiler, derive addresses from `abi.py` or this table and keep the output and input-release rules intact.
