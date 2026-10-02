# Homework Released: Agentic Design Automation of AI Infra HW-SW Co-design

Dear students,

The homework, **Agentic Design Automation of AI Infra HW-SW Co-design**, is now available:

- [Homework portal and access links](https://mlsys.github.io/homework/)
- [Full assignment statement](https://linux-slai.tail6d76d1.ts.net:8443/statement/)

In this homework, you will use your own AI agent to perform **hardware–software co-design**. You will jointly choose:

- one AI hardware accelerator configuration; and
- two assembly programs: one for prompt prefill and one for autoregressive decoding.

The central requirement is to optimize the hardware and software together. A hardware resource is useful only when the programs can exploit it, and a program optimization must be evaluated on the hardware you selected. Both programs must run correctly on the same hardware. Your goal is to improve their combined simulated performance while satisfying the area, power, latency, and correctness requirements.

## Deadlines

### Milestone 1

**October 7, 2026, at 12:00 noon, Beijing time**

Submissions received by this milestone will inform the selection of collaboration proposals before the October 10 college project application deadline. You may continue improving your work afterward.

### Final submission

**October 12, 2026, at 12:00 noon, Beijing time**

The server will stop accepting new submissions at this time. Submissions received before the deadline may finish grading afterward.

## Getting started

1. Read the full assignment statement.
2. Download and extract the starter package.
3. Reproduce the supplied baseline locally.
4. Change one hardware or software design idea at a time, evaluate it on the shared system, and record the result.
5. Check correctness locally before measuring performance.
6. Upload your final ZIP through the submission page.

Your ZIP must contain these paths at its root:

```text
hardware.json
programs/M1_P1.asm
programs/M2_D1.asm
project/iteration-log.md
agent-trace/
```

Include the project files used to generate and evaluate your programs under `project/`. The `agent-trace/` directory must contain the complete trace of the agent sessions used for the homework.

## Important reminders

- Complete the homework independently with your own AI agent.
- Do not request or reuse another participant's solutions, code, designs, feedback, submissions, or agent output.
- The unchanged baseline has a score of **1000**.
- A submission is eligible only if both programs are correct and all published area, power, and latency limits are satisfied.
- Validate locally rather than using the submission server as an experimental loop.
- You may upload at most once every 10 minutes. Grading normally takes 10–30 minutes and may take longer when the queue is busy.
- Use the same display name and real student ID for every submission.
- Save the receipt ID and lookup key returned after uploading. Your highest eligible server score under the same student ID will be retained.

If you believe you have found a problem in the scorer, contact **yiding@slai.edu.cn** or the TA on WeChat.

Please begin early. The assembly programs are substantial, and productive improvement will require repeated correctness checks and measured experiments.
