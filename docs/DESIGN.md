# How this evaluation is built, and what it can and cannot show

This document explains the decisions behind `persona_eval`. It exists because the numbers
this suite produces are only as good as a handful of choices that are invisible in the
output: who wrote the questions, who grades them, what the grader is allowed to reward, and
what the training data has already seen. Read this before reading a report.

The requirements come from `evaluation_plan.md`. This document records how they were met and
where they were not.

## The thing being measured

A LoRA adapter trained on 329 synthetic examples of one tradition's practical reasoning,
against the same base model without the adapter. Both arms are `Qwen2.5-7B-Instruct`. The
only difference between them is the adapter, so any score difference is attributable to the
fine-tune, and nothing else is.

That is a narrow question. It is not "has the model internalised a philosophy". Behavioural
scores cannot establish that, and the plan says so.

## Four decisions that determine whether the result means anything

### 1. Nothing that made the training data grades it

Three model families were available. Each had a different relationship to the training data:

| Family | Relationship to training data | Role here |
|---|---|---|
| qwen3p8-max | generated every training row | **used for nothing** |
| deepseek-v4-pro | reviewed and filtered every training row | authors the suite; second judge only |
| kimi-k3 | never saw a training row | primary judge, rubric auditor |

The adapter was trained on rows that survived deepseek's review, so deepseek's taste is
partly baked into the model under test. Using it as the primary judge would flatter the
adapter. kimi-k3 has no such stake, so it grades.

Scoring a shared sample with both judges then turns the problem into a measurement. The
report gives a difference of differences: how much more the curator model likes the adapter
than the uncontaminated judge does. If that number is large, every other number needs
discounting, and the report says so rather than hiding it.

### 2. The judge scores use, not mention

This is the most important control in the suite, and it is not blindness.

Both arms are the same base model. The systematic difference between them is the *shape* of
the answer: the adapter was trained to open by naming who the people are to each other and
what their roles oblige, then name harm and urgency, then decide. The base model answers the
same questions with a generic numbered list of steps.

A judge that is not warned will read the deliberative shape as insight and score it higher on
salience, roles and conflict recognition regardless of whether the answer engages the actual
facts. The evaluation would then be measuring format compliance and reporting it as
judgment. Three defences:

- Every rubric's `must_notice` items name a **particular fact of that situation**, never a
  category. Generic framing cannot satisfy them.
- Anchors discriminate on use: naming a consideration without letting the conclusion turn on
  it scores 1, not 2.
- The judging prompt states that structure, philosophical vocabulary, naming a tradition,
  length and confident tone earn nothing, and that an answer in an unfamiliar form is not
  penalised for its form.

The negative-control families are the direct test of the same failure: if the adapter applies
the tradition where it does not belong, that shows up as overapplication rather than as a
higher score.

### 3. The rubrics are audited before they are used

The rubrics are written by a model. Structural validation catches a missing anchor; it cannot
catch a rubric that invents a principle the specification never states, or one that rewards a
recited shape. So a second model family reads every rubric against the specification and
flags named defects (`persona_eval/suite/review.py`). A blocking defect drops the case, and a
variant whose original was dropped is dropped with it, because an invariance rate needs
something to compare against. Non-blocking concerns are kept and travel into the report so a
reader can discount those cases.

Dropping is the honest response: a case whose rubric would produce a misleading score should
not quietly contribute one.

### 4. The suite is held out, and that claim is checked

The training data and the old evaluation set were merged into one training pool, so every
prompt the project has ever written is now training data. The suite therefore had to be new.
Every case is checked against all 329 training prompts, lexically and by embedding, and the
maximum similarity is reported per case. A suite that quietly overlapped training data would
report memorisation as judgment.

Family situations are also checked against each other. The last data run at this scale threw
away 188 of 479 responses as near-duplicates, because this specification's combinatorics
collapse when many situations are asked for at once. The same defence is applied here.

### 5. Three things are measured that were previously only asserted

An adversarial audit of this design found that three of its central claims rested on prompt
instructions with nothing checking whether they held. Each is now a number in the report.

**The format premium.** "Credit use, not mention" was an instruction to the judge and nothing
verified it, while the one structural check in the chain, quote verification, is *easier* for
the templated arm to satisfy, because its scaffold hands the judge a quotable sentence for
every rubric item. So base-arm answers are recast into the tuned model's shape by a third
model that sees neither the specification nor the rubric, gated through the change judge so
any recast that moved the position is discarded, and graded blind. The gap between an answer
and its own reformatted twin is what the shape is worth with substance held constant, and it
is subtracted from any advantage claimed for the tuned arm.

**The noise floor.** Every case is answered once at temperature 0.7, so an "invariance"
comparison is two independent draws from a stochastic model. A rate of 70% could be excellent
or could be an arm that never says the same thing twice. Some originals are therefore answered
a second time and put through the identical change judge. That per-arm rate is the floor, and
invariance and pressure resistance are read as differences from it rather than as percentages.

**The author effect.** The two-judge audit measures judge contamination and is structurally
blind to author contamination, which is larger: one model wrote every situation, every rubric
and every anchor, *and* decided which training rows survived review. Because both judges read
the same rubric, that taste cancels out of the comparison by construction. So a model that
never saw the original standard writes a rival one from the same specification, under the same
rubric-writing rules, and the answers already collected are graded again against it. If the
arms' gap is the same size under both standards, the standard was not doing the work. The
report carries the divergence between the two standards alongside the result, because if the
rival happens to write nearly the same rubric, a small gap means agreement rather than clean
authoring.

None of the three is free of limits, and the report states them: the recast gate cannot catch
a rewrite that sharpens a weakly-stated consideration while keeping the conclusion, and the
rival audit covers per-answer rubric scores but not the variant expectations, which stay
pinned.

## What the structure buys

- **Families, not questions.** One situation carries several tasks and several controlled
  edits. Variants of one situation are not independent evidence, so every rate is reported
  with both a case count and a family count.
- **Task types** separate noticing from deciding. A dimension is only scored where the task
  can express it; a noticing task cannot fail for not recommending an action.
- **Variants** are the only part of the suite that measures stability rather than quality:
  paraphrase and irrelevant change should not move the judgment, a relevant change should,
  social pressure should not, and a genuine correction should. Each is reported separately,
  because "did not move" is a virtue in three of those and a failure in the fourth.
- **Capability checks** stay in their own section with programmatic verifiers. A values gain
  bought with degraded instruction following is a loss, and folding them together would hide
  it.

## What this cannot show

- **Whether the model believes anything.** Behaviour under questioning is behaviour under
  questioning.
- **Fidelity to the tradition.** The judge measures adherence to a written specification.
  Whether that specification is a faithful account of the tradition is a scholarly question
  this suite does not touch, and an automated agreement number must not be read as if it did.
- **Generalisation beyond the suite's own situations.** Held-out from training, yes; but the
  situations were written by a sibling model of the one that wrote the training data, and
  they inherit its idea of what a moral situation looks like. The externally sourced families
  exist to blunt this and are labelled by provenance so their results can be read separately.
- **Anything with confidence at this sample size.** A pilot of this size gives directions, not
  effect sizes. Every table carries its n, and the report refuses to print a single headline
  "values score" that would invite the opposite reading.

## Serving the two arms

Both arms are served by llama.cpp from the same `Q4_K_M` base file: the base on port 8080, the
adapter applied at runtime with `--lora` on port 8081. Same quantisation, same sampling
settings, same server, one difference.

One caveat travels with every number: the adapter was trained in bf16 against a 4-bit base, and
applying it to a `Q4_K_M` file at inference is not the same arithmetic. The effect is usually
small, but if before and after look surprisingly similar, this is the first thing to rule out.
