# AVSecTester agent instructions

These instructions are for AI coding agents that use or change AVSecTester. They define what an
agent must not change. The documentation describes how to use the framework.

AVSecTester runs closed-loop security experiments on driving systems under unified settings, so
that attacks and their treatments stay comparable. Start every task from the
[documentation index](docs/README.md).

## Use the framework without changing it

Unless the task explicitly asks you to change AVSecTester itself:

- Do not modify tracked files in this repository.
- Put your code in your own package and connect it through the documented extension points.
- Start an experiment from a supplied configuration. Copy it and edit the copy.
- Keep the supplied settings unless the task requires a change. Report every change that can
  affect results or comparability.
- Use the settings, outputs and comparisons the framework provides. Do not replace them with
  your own code.
- Use only the public API. Do not patch framework code at runtime.
- If the task cannot be completed through the documented extension points, stop and report the
  limitation.
- Experiments can occupy GPUs for hours. Check a reduced configuration first and start full
  runs only when asked.

## Change AVSecTester itself

When the task is a framework change, follow [Contributing](docs/development/contributing.md).
