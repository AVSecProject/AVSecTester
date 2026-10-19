# CLI reference

> **Audience:** Users running experiments from the command line.  
> **Outcome:** Look up every command, option, output line and exit code.  
> **Prerequisites:** [Installation](../getting-started/installation.md).

The `avsectester` command is a console script installed with the package
(`avsectester.cli:app`). It has two commands, `run` and `version`. `avsectester --help` lists
them, and `avsectester run --help` lists the options of `run`. The shell-completion options
`--install-completion` and `--show-completion` are provided by Typer.

## avsectester run

```text
avsectester run [OPTIONS] CONFIG
```

Runs one experiment from a configuration file through `run_configured`, prints a summary of each
run and, for an attack pair, the verdict. The exit code reports the outcome. See
[Exit codes](#exit-codes).

| Argument | Type | Meaning |
|---|---|---|
| `CONFIG` | Path, required | YAML file containing one mapping. The file must exist, be readable and not be a directory. Keys are listed in the [Configuration reference](configuration.md) |

| Option | Type | Default | Effect |
|---|---|---|---|
| `--frames` | int, at least 0 | YAML `frames`, otherwise `40` | Driving steps per run. Overrides the YAML `frames` key |
| `--mode` | `single` or `pair` | YAML `mode`, otherwise `pair` | Overrides the YAML `mode` key. `single` runs the selected treatment once. `pair` runs a clean baseline and the selected treatment from one origin |
| `--treatment` | `clean`, `attack`, `defense` or `attack-defense` | YAML `treatment`, otherwise `attack` | Overrides the YAML `treatment` key. What each treatment enables is described in [Treatments and modes](../guides/experiments.md#treatments-and-modes) |
| `--gpu` | int, at least 0 | none | Sets `ego.pipeline.perception.gpu`, the CUDA device of the modular perception component, in a CARLA scene. It has no effect when the pipeline has no `perception` entry. It does not select the CARLA server's GPU. Rejected when `stack` is configured or the backend is not CARLA. Configure those devices in `stack.options` |
| `--plot` | path | none | Saves a driving-impact figure of the pair to this path. Missing directories are created. Relative paths resolve against the current directory. Requires `pair` mode and matplotlib (the `viz` extra) |
| `--help` | flag | | Prints the options and exits |

The command proceeds in this order:

1. Loads the YAML file and resolves `mode`, `treatment` and `frames`, with options taking
   precedence over YAML keys.
2. Checks the CLI-level settings: `--plot` with `pair` mode, `backend` as a mapping, and for CARLA
   scenes `client` and `ego` as mappings, `ego.pipeline` as a mapping unless `stack` is set, and
   the use of `--gpu`. See [Validation rules](configuration.md#validation-rules).
3. In a CARLA scene, applies `--gpu` and generates a missing `client.seed`.
4. Calls `run_configured` with the resolved mode, treatment and frames, and adds `config_path`,
   the absolute path of `CONFIG`, to the scenario provenance.
5. Prints the output lines and saves the figure. See [Output](#output).

A scene is a CARLA scene when `backend` is omitted or `backend.type` is `CarlaSimulator`.
For CARLA scenes the CLI supplies its own components to `run_configured`:

- The backend is `CarlaSimulator(scene_config)`, unless `selection` is configured. With
  `selection`, the configured selection source builds the backend.
- The driving system is the configured `stack`, or the modular stack from `ego.pipeline` with
  perception outputs instrumented.
- Each run is recorded with component logging, which fills `FrameRecord.n_detections` from the
  perception outputs and sets `Trace.replay_scenario`.

Other backends, such as `NuRecSimulator`, use the default backend construction and
`configured_stack` of `run_configured`. Their runs report `mean_detections=0.0`. The CLI cannot
run Alpamayo, because the command is not installed in the Alpamayo driver environment. See
[Why NuRec has no CLI path](../getting-started/installation.md#why-nurec-has-no-cli-path).

Factories named in the configuration, such as `stack.factory` or an `interventions` entry, are
imported when the experiment is prepared. Their modules must be importable by the Python
interpreter that runs `avsectester`, for example through `PYTHONPATH`.

## avsectester version

```text
avsectester version
```

Prints the installed package version and exits with `0`:

```text
avsectester 0.1.0
```

## Output

`avsectester run` writes its results to standard output, one line per item, in this order:

| Line | Printed when | Content |
|---|---|---|
| `[scenario] seed=<seed>` | CARLA scenes | The resolved `client.seed`, printed before the first server operation. Set `client.seed` to this value to repeat the seeded scene choices. See [Random choices and seeds](../guides/simulators.md#random-choices-and-seeds) |
| `[<mode>] <frames> frames per run, treatment=<treatment>` | Always | The resolved mode, driving steps per run and treatment value, for example `[pair] 40 frames per run, treatment=attack` |
| `[<label>] mean_detections=<n> peak_speed=<m/s> final_speed=<m/s> brake_frames=<n> outputs=<n>` | Once per run, in run order | A summary of the run's `Trace`. The label is listed below |
| Pair notice | Pair mode, for some treatments | See [Pair notices](#pair-notices) |
| Comparison, two lines | Pair mode | The clean run compared with the second run |
| `=> <verdict>` | Attack pairs | The verdict of the driving-impact metric |
| `[output] wrote driving-impact figure: <path>` | `--plot` | The absolute path of the saved figure |

The summary fields are the `Trace` properties `mean_detections`, `peak_speed`, `final_speed` and
`braking_frames`, and the number of intervention outputs in `Trace.outputs`. Speeds are physical
speeds in m/s, printed with two decimals. See [Driving traces](../guides/evaluation.md#driving-traces).

In a pair, the clean run comes first. Runs are labeled by their treatment:

| Run | Label |
|---|---|
| `clean` | `clean` |
| `attack`, in a single run or an attack pair | `attacked` |
| `attack`, in a pair without an enabled attack | `baseline` |
| `defense` | `defense` |
| `attack-defense` | `attack+defense` |
| Second run of a `clean` pair | `baseline` |

An **attack pair** is a pair with treatment `attack` or `attack-defense` in which the treated run
has at least one enabled attack: a prepared intervention of kind `attack`, or a nonempty
`runtime.hooks` list. Only attack pairs print a verdict and return an outcome exit code.

### Pair notices

| Treatment | Notice |
|---|---|
| `defense` | `[pair] Defense side effects relative to clean.` |
| `clean` | `[pair] Baseline comparison.` |
| `attack` without an enabled attack | `[pair] No intervention hooks configured. Comparing two baseline runs.` |
| `attack` or `attack-defense` in an attack pair | None |

### Comparison and verdict

The comparison has two lines, printed by `Impact.comparison`:

```text
clean:    peak_speed=<m/s>  final_speed=<m/s>  brake_frames=<n>
<label>: final_speed=<m/s>  brake_frames=<n>  overdrive_frames=<n> (kept moving while clean was stopped)
```

`<label>` is `attack+defense`, `attacked`, `defense` or `baseline`, as in the run lines. An attack
pair adds the verdict line. The possible verdicts are:

| Verdict line | Exit code |
|---|---|
| `=> ATTACK SUCCEEDED (suppressed a safe stop — ego failed to brake for the lead)` | `0` |
| `=> ATTACK SUCCEEDED (forced an unsafe stop)` | `0` |
| `=> ATTACK INDUCED BRAKING (slowed the ego, but no full stop)` | `3` |
| `=> no impact (attack did not change the drive)` | `3` |
| `=> INCONCLUSIVE — clean run never drove (peak <m/s> m/s); run more frames or check the spawn` | `4` |

The criterion and its thresholds are explained in
[Driving-impact metric](../guides/evaluation.md#driving-impact-metric).

For example, a defense pair prints:

```text
[pair] 40 frames per run, treatment=defense
[clean] mean_detections=<n> peak_speed=<m/s> final_speed=<m/s> brake_frames=<n> outputs=<n>
[defense] mean_detections=<n> peak_speed=<m/s> final_speed=<m/s> brake_frames=<n> outputs=<n>
[pair] Defense side effects relative to clean.
clean:    peak_speed=<m/s>  final_speed=<m/s>  brake_frames=<n>
defense: final_speed=<m/s>  brake_frames=<n>  overdrive_frames=<n> (kept moving while clean was stopped)
```

A single run prints the `[scenario]` line for CARLA, the `[single]` line and one run line.

### Figure

`--plot` saves the figure of [Plot a comparison](../guides/evaluation.md#plot-a-comparison) for
the clean run and the second run, with the second run's label in the legend:

| Pair | Title | Verdict caption |
|---|---|---|
| Attack pair | `Adversarial attack — driving impact (clean vs <label>)` | Yes |
| `defense` | `Defense side effects (clean vs defense)` | No |
| `clean`, or `attack` without an enabled attack | `Driving comparison (clean vs baseline)` | No |

### Errors

Errors are written to standard error:

| Message | Cause | Exit code |
|---|---|---|
| `Configuration error: <message>` | Invalid YAML or settings, or a configuration error raised before the first reset | `2` |
| `Cannot read scenario: <message>` | The file could not be read | `2` |
| `Execution error: <message>` | An exception during the runs or while saving the figure | `1` |
| Typer usage message | A missing or unreadable `CONFIG`, or an invalid option value | `2` |

The `[scenario]` and `[<mode>]` lines are printed before the experiment is prepared. They also
appear when preparation or execution then fails.

## Exit codes

| Code | Meaning | Produced by |
|---|---|---|
| `0` | Success | `avsectester version` and `--help`. A completed single run. A completed pair that is not an attack pair: `clean`, `defense`, or `attack` without an enabled attack. An attack pair whose verdict is `ATTACK SUCCEEDED` |
| `1` | Execution error | An exception during the runs, such as a simulator connection failure, a spawn failure or a handler exception. A missing matplotlib with `--plot`, after the runs completed |
| `2` | Configuration error | Typer usage errors, YAML and file errors, the CLI checks, and `ConfigurationError`, `ScenarioConfigurationError`, `ConditionConfigurationError`, `UnsupportedStageError` and `CommandCompatibilityError`. See [Validation rules](configuration.md#validation-rules) |
| `3` | Attack failed | An attack pair whose clean run drove, but whose attack did not meet the success criterion: `ATTACK INDUCED BRAKING` or `no impact` |
| `4` | Inconclusive | An attack pair whose clean run never reached the baseline speed |

Codes `3` and `4` are produced only by attack pairs, with treatment `attack` or `attack-defense`.
Defense-only, clean and baseline comparisons have no verdict and exit with `0` after successful
execution. A script can therefore distinguish an experiment outcome from a failure:

```bash
avsectester run configs/carla_scenario.yaml --frames 40
case $? in
  0) echo "completed; for an attack pair, the attack succeeded" ;;
  3) echo "attack failed" ;;
  4) echo "inconclusive: the clean run never drove" ;;
  *) echo "configuration or execution error" ;;
esac
```
