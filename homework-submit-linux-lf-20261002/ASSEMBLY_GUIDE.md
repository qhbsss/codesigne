# How to read and modify the complete assembly examples

The [P1 listing](/examples/M1_P1.asm) and [D1 listing](/examples/M2_D1.asm) are complete working programs, not pseudocode. P1 has 1,843 source lines; D1 has 2,618. Both files, an editable compiler at `project/compiler.py`, its frozen reference source at `codesign/challenge/baseline.py`, and the public checker are in the [starter ZIP](/downloads/transformer-codesign-starter.zip). This guide explains their repeated structure so that you can navigate or regenerate them without reading thousands of lines in one pass.

Use the [assignment statement](/statement/) for the required work, the [ISA](/isa/) for exact instruction fields, and the [ABI](/abi/) for every HBM symbol. Line numbers below apply to the frozen example files; generated or edited files will differ.

## 1. The three levels of the program

1. **Scenario and step.** P1 has two 64-position prompt phases followed by one new position for each batch element. D1 contains eight sequential decode steps. Step boundaries control which external input is readable.
2. **Layer.** Each P1 phase computes layers 0, 1, and 2; each D1 step computes layers 0 and 1. The output of one layer becomes the input of the next.
3. **Instruction tiles.** Within a layer, `FOR` loops tile matrix multiplication, move HBM data into RF, compute with `MMA.ACC`/`VEC`/`REDUCE`/`SFU`, and store results. The loops are expanded before execution; the source does not contain a runtime branch.

The baseline chooses one workgroup named `g` on SM 0, with zero shared-memory quota. It uses RF partitions (the `lane` field) and HBM scratch for intermediate tensors. The hardware JSON has eight SMs, but this simple program does not distribute its work across all of them. It is a correctness starting point, not a claim of optimal scheduling.

## 2. Locate a block by its name

The baseline generator embeds batch element, phase, decode step, layer, and operation in loop names:

| Example prefix | Meaning |
| --- | --- |
| `pb0l0...` / `pb1l2...` | P1 prompt, batch 0 layer 0 / batch 1 layer 2 |
| `sb0l0...` / `sb1l2...` | P1 first new position, batch 0 layer 0 / batch 1 layer 2 |
| `d7l1...` | D1 decode step 7, layer 1 |
| `...ln1` / `...ln2` | First / second LayerNorm |
| `...qkv`, `...wo`, `...w1`, `...w2` | The four weight matrix multiplications |
| `...a...` | K/V export, attention scores, softmax, and context |
| `...res`, `...bias1`, `...gelu`, `...out` | Residual, FFN bias, activation, and final layer output |

Search for a loop prefix such as `"pb1l2qkvm"` or `"d7l1qkvm"` to locate a matrix tile. `STEP.COMMIT` marks release boundaries. Event names interpolate static loop indices into unique strings; they are not tensor names.

### P1 source map

| Lines | Meaning |
| --- | --- |
| 1 | `WG.BEGIN` for workgroup `g` |
| 2–457 | Batch 0 prompt: layers 0–2 |
| 458–461 | Copy batch 0's 64 final prompt hidden vectors |
| 462–917 | Batch 1 prompt: layers 0–2 |
| 918–921 | Copy batch 1's 64 final prompt hidden vectors |
| 922 | `STEP.COMMIT {"step":0}` checks **both** prompt hidden/KV sets and releases the two `input/step0` vectors |
| 923–1378 | Batch 0 first new position: layers 0–2 |
| 1379–1382 | Copy batch 0's new hidden vector |
| 1383–1838 | Batch 1 first new position: layers 0–2 |
| 1839–1842 | Copy batch 1's new hidden vector |
| 1843 | `WG.END` |

P1 needs one intermediate commit after **both** prompts. The two first-new-position results are checked at case end. For `output/hidden` shaped `[2,65,256]`, batch `b`, position `t`, component `c` is at `base + (b*65+t)*256+c`. The K/V tensors have a different head-major order described in the [ABI](/abi/).

### D1 source map

Each step is written out separately. Within a step, two layer blocks appear in order, followed by the final hidden store and a commit.

| Decode step | Lines through its `STEP.COMMIT` |
| ---: | ---: |
| 0 | 2–328 |
| 1 | 329–655 |
| 2 | 656–982 |
| 3 | 983–1309 |
| 4 | 1310–1636 |
| 5 | 1637–1963 |
| 6 | 1964–2290 |
| 7 | 2291–2617 |

Line 1 starts the workgroup; line 2618 ends it. In step 0, layers begin at lines 2 and 164. `STEP.COMMIT {"step":s}` checks the current output and K/V coverage, then releases `input/step{s+1}` when `s < 7`. D1 requires commits numbered **0 through 7**.

## 3. Follow one layer's dataflow

For both cases, one layer follows this order. All weight tensors are read-only; temporary tensors are allocated within `scratch`.

| Stage | Input → output | What the listing does |
| --- | --- | --- |
| First normalization | incoming `x` → `ln1` | `REDUCE sum` computes mean and variance; `SFU rsqrt` and `VEC` apply gamma/beta. Epsilon is `1e-5`. |
| QKV projection | `ln1 × wqkv` → `qkv` | Tiled `MMA.ACC`; P1 has 768 output columns: Q (0–255), K (256–511), V (512–767), split into eight heads of 32. D1 has 384 columns, split into four heads of 32. |
| K/V export | `qkv` → `layerN/new_k`, `layerN/new_v` | `ST` writes new keys and values for **this layer** in head-major order. All layers need their own KV writes. |
| Attention | Q and permitted K/V → `ctx` | Compute scaled Q·K scores, subtract the row maximum, exponentiate, normalize, then compute the weighted V sum. Causal limits change by query position. |
| Attention projection and residual | `ctx × wo + x` → `res` | Project context back to the case's model width, then add the incoming hidden vector. |
| FFN | `LayerNorm(res) × w1 + b1` → GELU → `× w2 + b2 + res` | P1 expands 256→1024→256; D1 expands 128→512→128. The output becomes the next layer's `x`. |
| Final layer only | last-layer output → `output/hidden` | Store every required final hidden vector before its commit or case end. |

The baseline stores many intermediates in HBM scratch and allocates a fresh range for each one. That makes the dataflow easy to inspect and leaves opportunities to fuse or reuse storage. If you change this, preserve the same observable final hidden and K/V values.

### Attention differs between the two cases

- **P1 prompt:** for query position `t` within either 64-position prompt, read only that batch element's keys `0..t`; future positions and the other batch element are excluded. Each first new query sees its own prompt positions `0..63` and position `64`.
- **D1 step `s`:** read 128 historical keys from `history_k/v` and `s+1` generated keys from `new_k/v`, including the current one. The two HBM arrays are separate. Do not overwrite history, and do not omit earlier generated KV when decoding a later step.
- The baseline stores attention scores and probabilities in scratch and uses a maximum-subtracted softmax. Another stable algorithm is allowed if the numerical check passes.

## 4. Read HBM operands as element addresses

The [ABI table](/abi/) gives **FP32 element offsets**. Do not multiply the listed offset by four in assembly. For a row-major tensor with base `B` and shape `[rows, columns]`, element `(i,j)` is at `B + i*columns + j`.

For example, P1 `layer0/wqkv` begins at element offset **512** and has shape `[256,768]`. Its element `(2,16)` is at `512 + 2*768 + 16 = 2064`. A tile with 48 rows and 16 columns starting at the base can use this HBM view:

```json
{"space":"HBM","offset":512,"count":768,"wg":null,"lane":0,"shape":[48,16],"strides":[768,1]}
```

`count` is the **logical** number of values, `48*16`, while `strides` describe their physical spacing in HBM. A two-dimensional tile must supply both `shape` and `strides`; using `count:768` alone would read 768 contiguous elements and therefore the wrong matrix tile. The matching RF destination can be contiguous with the same count.

K/V use a different flattening order. For `layer0/new_k` in P1, shape is `[2,8,65,32]`; the FP32 offset of batch `b`, head `h`, position `t`, component `c` is `base + (((b*8+h)*65+t)*32+c)`. D1 `new_k` is `[1,4,8,32]`, while its historical K/V are `[1,4,128,32]`; use the corresponding position strides and their separate ABI bases. The exact bases for every layer are in the ABI table.

`RF` and `SH` offsets are also element offsets, relative to a workgroup. In the baseline hardware, a workgroup has 64 KiB of RF split across 16 `lane` partitions: each partition holds **1024 FP32 words**. `lane` identifies the partition; it is not an HBM address. The baseline uses RF lanes 0 and 1 for loaded operands and lane 2 for the accumulator.

## 5. Understand the repeated GEMM pattern

A baseline matrix multiplication is tiled as **up to 8 rows × 16 output columns × 48 K elements**:

1. `FOR ...m` selects up to eight rows; `FOR ...n` selects 16 output columns.
2. `VEC add` of two zero immediates initializes the RF accumulator. `MMA.ACC` adds to the existing accumulator; skipping initialization changes the answer or reads unwritten RF.
3. `FOR ...k` visits the K dimension in 48-element chunks. Two `LD` instructions bring the A and B tiles from HBM to RF. Their `shape`/`strides` preserve the source matrices' row-major layout.
4. `MMA.ACC` multiplies those RF tiles and accumulates into RF lane 2. The final K chunk uses `min(48, remaining K)` to avoid out-of-range elements.
5. `ST` writes the accumulated tile to its HBM scratch output.

A long expression such as `{"min":[48,{"add":[256,{"mul":[-1,{"var":"...k"}]}]}]}` means `min(48, 256 - k)`. It is a **compile-time integer expression** used for the final tile size. `{"var":"...k"}` substitutes the current static loop index; braces in an event name interpolate that index into a unique string. See the [ISA expression rules](/isa/#static-loops-and-expressions) before editing generated expressions.

## 6. Commit, event, and workgroup rules that often cause failures

- **Input release:** P1 `input/step0` is unreadable before the prompt commit. D1 `input/step{s}` is unreadable before all earlier steps commit. A correct final HBM result cannot excuse an early read.
- **Output coverage:** each required final hidden and new K/V element must be written once. A commit fails when current required output/KV positions are missing. Writing an output position twice also fails.
- **Initialized storage:** RF/SH reads require a prior write. An `MMA.ACC` accumulator must start with an initialized value. Scratch HBM must be written before it is read.
- **Events:** each expanded operation needs a unique event name. `WAIT` names earlier events and a specific workgroup; `BARRIER` names participating workgroups. Reusing an event inside expanded loops fails.
- **Cross-workgroup HBM:** two workgroups' source-line order alone does not order overlapping HBM accesses. Use synchronization with the producer's event, a barrier, or `STEP.COMMIT` as appropriate.
- **Transfer source reuse:** an `ST` may still be reading RF/SH after issue. Wait for its service to finish before overwriting an overlapping source. The timing checker enforces this dependency.
- **Program completion:** end every workgroup. P1 needs one intermediate commit; D1 needs all eight commits. Static `FOR` bodies must be well-formed even when they execute zero times.

## 7. Generate, check, and submit

The starter ZIP already contains valid files. Treat the long `.asm` listings as generated output. `project/compiler.py` is an editable copy of the baseline Python generator; leave `codesign/challenge/baseline.py` unchanged because the frozen baseline manifest checks that reference source. No C program or general-purpose compiler frontend is required.

The compiler's **inputs** are its Python source (the baseline Transformer operations and chosen software optimizations), the public model/scenario definitions in `workload_v07.json`, and HBM symbol offsets built by `abi.py`. The `Builder` methods emit ISA instructions and allocate scratch. `generate_m1_p1()` and `generate_m1_d1()` each return `(assembly_text, layout, scratch_end)`; the command below writes the first result into each `programs/*.asm` file. The compiler does not read input tensor values or automatically derive a schedule from `hardware.json`. The public checker later loads those values and evaluates your generated assembly with your selected hardware.

From the extracted starter directory, regenerate the published assembly with:

```sh
python -m project.compiler
```

Modify `project/compiler.py` or replace it with your own generator to change tiling, loop order, RF/shared-memory reuse, fusion, workgroups, or scheduling. The current `Builder.gemm` handles partial M and K tiles but only accepts GEMM N tiles of 8 or 16 that divide the output width. A compiler you extend can emit N tail tiles by using the remaining width in its load, RF, `MMA.ACC`, and store views. The ISA accepts positive MMA dimensions; correctness and hardware resource checks still apply. You may add a fast syntax or shape checker to your compiler to reject mistakes before the full public `check` run.

After changing the generator or program, run a public correctness check **before** comparing performance:

```sh
python challenge.py check --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --seed 7 --report check-1.json
python challenge.py estimate --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --seed 7 --report estimate-1.json
python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report local-grade.json
```

`estimate` measures the simulated cost without proving correctness. `check` compares outputs with public reference inputs. `grade` writes a public experimental score to the JSON file named by `--report`; attach that unedited file as `local-grade.json`. The website can display its score as Local, and a later verified result replaces it. Each command needs a fresh report path. Use Python 3.12 and NumPy 2.x, then ZIP `hardware.json`, both `programs/*.asm` files, and `local-grade.json` at the paths shown in the statement. Add your complete agent trace to the final ZIP before the deadline; its file name and format are your choice.
