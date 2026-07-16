# zen-rl

Diagnose failed RL training runs from TensorBoard logs.

No LLM. No API key. No network. Instant.

```bash
pip install zen-rl
zen runs/my_experiment
```

```
LR_TOO_HIGH
Drop learning rate to 3e-4 and add a 500 step warmup, then rerun.
  Evidence: train/approx_kl = 21.73 at step 4096
```

---

## What it does

You point it at a TensorBoard log directory. It tells you what went wrong,
shows you the number that proves it, and tells you what to try next.

Every claim cites a metric, a value, and a step. You can check its work.

## What it detects

| Mode | Signature |
|---|---|
| `LR_TOO_HIGH` | `approx_kl` spikes above 5.0 |
| `POLICY_COLLAPSE` | `entropy_loss` hits ~0 and stops moving |
| `ADVANTAGE_NOT_NORMALIZED` | `\|policy_gradient_loss\|` above 1.0 |
| `DEAD_RUN` | final `ep_rew_mean` under 50 |
| `CAPPED_PERFORMANCE` | final `ep_rew_mean` between 50 and 250 |

When it finds nothing, it says so and lists what it checked. It does not
claim your run is fine — only that these five things didn't fire.

## Root cause, not symptom list

A high learning rate causes entropy collapse, which kills the run. That's one
problem, not three. zen-rl reports the cause and suppresses the consequences.

## Scope

Built and tested against **PPO / Stable-Baselines3 / TensorBoard**.
Other algorithms and loggers may work if the metric names match.

Thresholds were derived from 16 deliberately broken CartPole runs.
They are a starting point, not gospel — different environments will need
different numbers. Issues and PRs welcome, especially with a log attached.

## Why no LLM

The failure modes are deterministic. `approx_kl = 21.7` means the same thing
every time. A model would be slower, cost money, and hallucinate. An
if-statement is the right tool.

## Install

```bash
pip install zen-rl
```

Requires Python 3.11+.

## Usage

```bash
zen path/to/run          # diagnose a run
zen history              # what zen has seen before
```

## Status

Early. 5 failure modes, 24/24 on the internal eval set, zero false positives.
The eval set is synthetic — runs broken on purpose. If you have a real failed
run it gets wrong, that's the most useful bug report you can file.

## License

MIT