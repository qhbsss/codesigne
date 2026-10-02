# From Layered Design to Agent-Guided Hardware–Program Co-design

## Why this assignment exists

Modern AI infrastructure is built as a stack. A model defines the computation. A compiler chooses kernels, tiling, data movement, and instruction order. A runtime schedules work. A hardware architect chooses compute units, memory capacity, bandwidth, and interconnect.

These layers are useful, but they are not independent. A larger matrix engine helps only if the program can supply enough work and data. More on-chip memory helps only if the compiler finds reuse that avoids expensive off-chip accesses. A schedule that is efficient on one accelerator may perform poorly on another.

In principle, we could describe all of these choices in one optimization problem. In practice, the joint search space is enormous. Hardware choices multiply with program choices, and program behavior depends on the hardware on which it runs. Traditional system design therefore separates the problem into layers and relies heavily on abstractions, heuristics, and human experience. This is not a conceptual failure. It is a practical response to a problem that has been too large and uncertain to search as a whole.

This assignment asks whether that boundary can now move.

## Why AI inference changes the problem

General-purpose programs often contain input-dependent branches, irregular memory access, unpredictable synchronization, and rapidly changing workloads. These behaviors make it difficult to predict how a program will use a proposed machine before the program actually runs.

Transformer inference is different in several important ways. For a fixed model and deployment scenario:

- the operator graph is largely known;
- tensor shapes and dependencies are explicit;
- the same layer structure is executed repeatedly;
- data movement and storage requirements can be modeled;
- important deployment cases, such as prompt prefill and token-by-token decoding, can be specified precisely.

Inference is not perfectly static. Sequence lengths vary, memory contention matters, and prefill and decode expose very different kinds of parallelism. Nevertheless, the workload is regular enough that a simulator can evaluate many hardware–program pairs under a common model of correctness, latency, area, power, and energy.

This regularity does not make co-design easy. It makes a larger part of the problem explicit and measurable.

## What agent-guided optimization adds

Even for a fixed Transformer, exhaustively enumerating every legal accelerator and every legal program remains impractical. The opportunity is therefore not to replace the entire design process with one monolithic solver. It is to search selectively.

An AI agent can maintain a history of proposed designs and their measured outcomes. It can use that evidence to generate new candidates, reject invalid or unpromising regions, and decide which experiments are most informative. Mathematical optimization methods, including mixed-integer linear programming when appropriate, can solve bounded subproblems or enforce exact constraints. A compiler or program generator can turn each software candidate into executable assembly. A deterministic simulator can then provide the feedback needed to revise both the hardware and the program.

The resulting loop is:

1. **Specify the workload.** Define the model, tensor shapes, numerical behavior, and deployment scenarios.
2. **Specify the legal design space.** Define the available hardware resources, program operations, and system constraints.
3. **Propose a joint design.** Choose both an accelerator configuration and a program mapping for that accelerator.
4. **Verify correctness.** Reject candidates that do not compute the required result.
5. **Measure the design.** Estimate latency, area, power, energy, and other relevant costs.
6. **Learn from the result.** Use the evidence to refine the search and propose the next candidate.

The agent does not remove the combinatorial search problem, and the simulator does not guarantee that the best observed design is globally optimal. Together, however, they make it possible to explore across boundaries that a conventional layer-by-layer process would normally hold fixed.

## The co-design problem in this homework

This homework gives you a bounded instance of agent-guided hardware–program co-design for Transformer inference.

You are given:

- a multilayer, decoder-only Transformer workload;
- two inference scenarios with different performance demands;
- a menu of legal GPU-like hardware resources;
- an assembly language for expressing computation, data movement, synchronization, and placement;
- a functional checker and a deterministic performance and cost model.

You must produce one joint design with two coupled parts.

### Hardware design

Choose a single accelerator configuration shared by both workloads. The available decisions include the number and shape of compute units, vector and special-function capacity, local storage, register-file bandwidth, DMA resources, network bandwidth, HBM channels, and cache capacity.

### Program design

Produce one program for prompt prefill and one for autoregressive decoding. The programs may choose tiling, loop order, workgroup placement, data movement, reuse, synchronization, fusion, recomputation, and attention implementation. Both programs must run correctly on the same hardware configuration.

The hardware and software decisions must be made together. Additional compute is valuable only when the programs can use it. Additional storage is valuable only when the mappings create reuse. A locally attractive choice may help prefill while hurting decode, or improve latency while violating area or power limits.

The joint design is a triple: `H` for the shared accelerator, `P_prefill` for the prefill program, and `P_decode` for the decode program. Search for the triple that maximizes `Score(H, P_prefill, P_decode)` subject to:

- functional correctness for both programs;
- one shared, legal hardware configuration;
- legal assembly and memory behavior;
- the published area, power, and latency limits.

The score combines the two workloads, so evaluate changes to each program on the same hardware.

## What you are expected to learn

The goal is not merely to obtain a high score. The assignment is designed to make four ideas concrete:

1. **System layers interact.** Hardware value depends on the program, and program quality depends on the hardware.
2. **Optimization requires evidence.** Familiar GPU rules are hypotheses until they are tested against the specified machine and workload.
3. **Agents need a disciplined experimental loop.** Useful search depends on preserving candidate changes, measurements, failures, and reasons for the next decision.
4. **Co-design is a constrained synthesis problem.** A fast candidate is irrelevant if it is incorrect, illegal, or outside the resource budget.

This is a deliberately bounded teaching problem, not a claim that unrestricted accelerator and compiler design can now be solved automatically. Its purpose is to let you experiment with a more integrated design process: formulate the workload and constraints explicitly, search hardware and software together, and use measured feedback to guide the next decision.

[Read the full assignment statement](/statement/)
