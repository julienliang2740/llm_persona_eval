# llm_persona_eval

An automated evaluation suite for value-related judgment. It builds held-out scenario families
from a written value specification, answers them with two or more model arms under identical
settings, grades every answer against a rubric frozen before grading, and reports the
diagnostics **separately** rather than collapsing them into one score.

Built to the requirements in `../evaluation_plan.md`. **[docs/DESIGN.md](docs/DESIGN.md)
explains the decisions that determine whether the numbers mean anything — read it before
reading a report.**

## What a run measures

| Question | How |
|---|---|
| Does it notice the right things? | `notice`, `duties_conflicts`, `boundaries` tasks scored 0/1/2 on applicable dimensions only |
| Does it decide well? | `decide` task; action and stated reasoning are scored and reported separately |
| Is it stable for the right reasons? | paraphrase and irrelevant edits must not move the judgment; a relevant edit must; social pressure must not; a genuine correction must |
| Does it apply the philosophy where it does not belong? | negative-control families, with an explicit eligible-case denominator |
| Does it transfer? | far-transfer families with unfamiliar institutions and relationships |
| Did general ability survive? | separate capability section: verifiable instruction following, math, small coding |

There is deliberately no single "values score". A gain in action choice must not hide worse
reasoning or more overapplication.

## Layout

| Path | What it holds |
|---|---|
| `main.py` | the CLI: `author`, `inspect`, `answer`, `judge`, `capability`, `report`, `run` |
| `configs/eval.yaml` | model roles, and the reasoning behind which model does what |
| `persona_eval/suite/schema.py` | the frozen data contract: families, cases, rubrics, results |
| `persona_eval/suite/author.py` | turns a value specification into a validated suite |
| `persona_eval/suite/review.py` | audits every authored rubric with a second model family |
| `persona_eval/suite/contamination.py` | proves the suite is held out from training data |
| `persona_eval/suite/external.py` | seeds some families from a public dataset |
| `persona_eval/run/answer.py` | answering, with task isolation and continuation turns |
| `persona_eval/run/judge.py` | blind rubric grading, change verdicts, reliability passes |
| `persona_eval/run/deterministic.py` | objective checks with programmatic verifiers |
| `persona_eval/capability/` | general-ability regression checks, separately reported |
| `persona_eval/report/` | the analyses and the markdown report |
| `persona_eval/runs.py` | run directories and the comparability record |
| `tests/` | unit tests, no network, no API keys |

## Usage

```bash
PY=../llm_persona_data_pipeline/.venv/bin/python

# 1. write a held-out suite from the target specification, audit its rubrics, check it
#    against training data. Costs Fireworks calls.
$PY main.py author --target confucian --out suites/confucian-v1.json

# 2. look at it without spending anything
$PY main.py inspect --suite suites/confucian-v1.json

# 3. answer, grade and report every arm. Arms are label=config_role.
$PY main.py run --suite suites/confucian-v1.json --arms base=base,adapter=adapter \
    --repeat-fraction 0.2 --second-judge-role judge_second --curator-judge deepseek

# or drive the stages yourself
$PY main.py answer --suite suites/confucian-v1.json --run <id> --arm adapter --role adapter
$PY main.py judge  --run <id> --arm adapter
$PY main.py report --run <id> --baseline base
```

Everything a run produces lands in `results/<run_id>/`: the suite it used, raw answers with
their sampling settings, grading records, change verdicts, capability results, the analysis
and the report. `run_record.json` holds enough to reproduce the run.

## The two arms

Both arms are the same `Qwen2.5-7B-Instruct` in `Q4_K_M`, served by llama.cpp:

```bash
bash ../llm_persona_data_pipeline/scripts/serve_base_model.sh      # base, port 8080
bash ../llm_persona_training/scripts/serve_adapter_model.sh        # base + LoRA, port 8081
```

Same quantisation, same sampling settings, same server, one difference. Answer the arms
**sequentially**: both servers share eight CPU cores, so running them at once halves each
one's throughput and buys nothing.

## Dependency on the data pipeline

This repo imports the pipeline's model client, config loader, record types and target loader
rather than copying them. The checkout is found at `$LLM_PERSONA_PIPELINE`, or at
`../llm_persona_data_pipeline`. Run from a venv with the pipeline's requirements installed and
the Fireworks key file in the pipeline repo.

## Legacy commands

`evaluate` and `compare` are the earlier single-score commands that judge a pipeline run's
`eval.jsonl` pass/fail. They still work, but that export no longer exists for the Confucian
target: it was merged into training data on 8 September 2026, which is why this suite had to
author its own held-out families.
