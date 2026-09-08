# llm_persona_eval

Evaluation for the value-instantiation project: answer a run's held-out `eval.jsonl` with any
model endpoint or a pre-generated answers file, judge every answer against the target's grading
key, and compare two result files before and after a fine-tune.

Moved out of `llm_persona_data_pipeline` (`pipeline/evaluate.py`, `prompts/evaluation.py`,
`tests/test_evaluate_modes.py`) on 8 September 2026. The code is unchanged apart from imports.

## Layout

| Path | What it holds |
|---|---|
| `main.py` | the CLI: `evaluate` and `compare` |
| `persona_eval/evaluate.py` | answer the eval set, judge, summarise, before/after table |
| `persona_eval/prompts.py` | the neutral answer system prompt and the judge prompt |
| `persona_eval/__init__.py` | locates the pipeline checkout and puts it on `sys.path` |
| `tests/` | unit tests, no network |

## Dependency on the data pipeline

This repo does not copy the pipeline's model client, config loader, record types or target
loader; it imports them. The pipeline checkout is found at `$LLM_PERSONA_PIPELINE`, or by
default at `../llm_persona_data_pipeline` next to this repo. Run the eval CLI from a venv that
has the pipeline's requirements installed, with the Fireworks key file in the pipeline repo.

Model roles (`base`, `judge`, and anything you add for an adapted checkpoint) come from the
pipeline config passed with `--config`, relative to the pipeline repo.

## Usage

```bash
# before: run the local base model over eval.jsonl of the latest confucian run and judge it
python main.py evaluate --target confucian --endpoint-role base --label before

# after: judge answers written by llm_persona_training/generate_with_adapter.py
python main.py evaluate --target confucian \
    --answers-file ../llm_persona_training/out/confucian/<run>/answers_adapter.jsonl --label after

# compare any two result files
python main.py compare --target confucian \
    --before ../llm_persona_data_pipeline/runs/confucian/<run>/eval_results_before.jsonl \
    --after  ../llm_persona_data_pipeline/runs/confucian/<run>/eval_results_after.jsonl
```

Results are written next to the export they were scored against:
`../llm_persona_data_pipeline/runs/<target>/<run>/eval_results_<label>.jsonl` and
`before_after.md`. Each result row is `{prompt_id, model, text, endpoint, family_id, case_type,
variant, counterfactual_group_id, judge: {pass, action_summary, principle_notes,
failure_modes_hit, rationale, judge_model}}`. Answers-file rows are `{prompt_id, prompt, model,
text}`.

## Tests

```bash
python -m pytest -q
```
