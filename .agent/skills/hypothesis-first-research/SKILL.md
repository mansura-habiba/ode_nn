---
name: hypothesis-first-research
description: Portable stage-gated pipeline for empirical research in any topic or repository. Source-check, register at most four hypotheses, anchor tests in published experiments, run reproducible comparisons, adjudicate and publish with minimal artifacts.
---

# Hypothesis-First Research — Portable Pipeline

## When to use
Apply to **any research**, in this or another repository, or outside GitHub. Treat this as a high-level process skill, not a domain-specific CCNN skill. A project only uses it when the skill is accessible/installed in that environment; never imply it is automatically installed everywhere.

**Mission:** Research question -> primary-source and state-of-the-art audit -> hypothesis -> published experimental protocol -> controlled experiment -> verdict -> publication. The researcher's intended contribution is the starting point, not an assumption the agent may replace with a convenient toy.

**Non-negotiables:** maximum **four** active hypotheses, ideally one primary; no fabricated references, outcomes or runs; no superiority declaration without adequately trained and fairly evaluated comparators; no silent redefinition of the method. Prefer **zero** new files during scoping, **one** canonical research record, **one** research issue for the objective when GitHub is used, and **one** main executable notebook per study. Every new file, issue, experiment and dependency requires an explicit reason.

## The pipeline (checkpoints are mandatory)

| Stage | Owner / adopted method | Exit evidence | Block if missing |
| --- | --- | --- | --- |
| **P0 Recover and scope** | Researcher + pipeline; read repository and checkpoint | Original question, researcher intent, prior decisions, constraints and next action | Clarify material ambiguities before creating |
| **P1 Verify literature and lineage** | Selectively adopt **KalarisLabs literature-review + citation-verification** methods | Verified original model, later variants, relevant published SOTA and genuine gap; exact sources | Do not freeze architecture or claim |
| **P2 Formulate and freeze hypothesis** | Researcher approves <=4 falsifiable hypotheses | Scope, endpoint, meaningful effect, comparator, uncertainty, cost constraints and rejection criteria | Do not run main benchmark |
| **P3 Lock published experimental protocol** | Source experiment + **KalarisLabs experimental-design** principles | Published dataset/splits/baselines/training/metrics/repeats; adaptations labelled; fixed test rules | No scientific run |
| **P4 Reproducible execution** | **Autoresearch-inspired deterministic runner** (only when useful) | Pilot validity; fair full baselines; immutable evaluator; seeds/config/commit/results ledger | No empirical conclusion |
| **P5 Scientific decision** | Compare observed results with frozen criteria | SUPPORTED / REJECTED / INCONCLUSIVE; magnitude, uncertainty, bounded claim and limitations | Additional run only with discriminating reason |
| **P6 Write and reconcile** | Existing paper/notebook structure | Publication-style results, source-backed architecture, citations, reproducibility, file inventory | Never manufacture a success story |

**Gate contract:** Finish and record the stage's evidence in the canonical record; tick the matching checklist in the existing issue; record current branch/commit, accepted and rejected decisions, blocker, and **exactly one next action**. On every resumed session start at P0, recover the last checkpoint and continue at the first unfinished gate. Do not silently reset the question, or open new issues for routine subtasks.

### P0. Recover intent and context
Before proposing research, read the original researcher question, available primary source, current notebooks, GitHub issue, and last checkpoint. Restate what is believed established vs merely proposed. Ask up to **three targeted questions** if answers materially change the proposed contribution, benchmark, comparisons, budget or acceptance criteria; do not repeat questions already answered. If no blocker, proceed. Never optimize for document count or activity.

### P1. Primary-source identity, literature search and verification
Read the **full primary paper sections and relevant figures, equations, supplements and experimental tables**. Distinguish in a compact lineage table: **original published method**, **later user-authored extensions**, and **new unvalidated proposals**; provide exact source references and differences. A shared formula does not mean two methods are identical. Treat user corrections as a signal to reverify source-level evidence, not a reason to invent new claims. Existing README and agent-generated summaries are not primary authority.

Adapt these verified upstream skill patterns:
- [KalarisLabs literature-review](https://github.com/KalarisLabs/research-agent-skills/tree/main/skills/literature-review): scoped search terms, inclusion/exclusion rationale, original method/experiment inspection, evidence table, related and negative results.
- [KalarisLabs citation-verification](https://github.com/KalarisLabs/research-agent-skills/tree/main/skills/citation-verification): resolve DOI/arXiv/proceedings; compare title, author and year; verify each statement against the cited paper, not merely its existence.
- [KalarisLabs experimental-design](https://github.com/KalarisLabs/research-agent-skills/tree/main/skills/experimental-design): replication, controlled comparison, randomization/blocking where applicable, identification of confounders and appropriate independent units.

**Selective adoption only:** Read upstream instructions before using their methods/scripts. Do not copy their dependencies, mandatory figure/PDF outputs, one-file-per-paper routines or whole framework. In ML, prefer directly relevant ICLR/ICML/NeurIPS/TMLR/JMLR protocols when appropriate; other fields require their own highest-relevance published standards. Record source year, peer-review/preprint status, architecture, data, strong baselines, evaluation, reported results, available code, and evidence gaps. Never invent SOTA rankings.

### P2. Register and approve hypotheses (<=4)
Define H1 first; only introduce H2-H4 when a distinct scientific claim needs testing. Each H must include: exact mechanism/claim, domain and published anchor, falsifying observation, metric and minimum meaningful difference, competing mechanism/explanation, adequately trained comparators, uncertainty/replication criterion, preservation/cost constraints, and what the decision changes. Get researcher's agreement on material scientific tradeoffs before freezing. Freeze and timestamp the registration **before** held-out results. Ablations and negative controls are not new hypotheses.

### P3. Lock a published experimental anchor
Identify **the actual published paper section, figure/table or supplement** underlying the experiment. Extract data generation/acquisition, preprocessing, train/validation/test splits, architecture, optimization, baseline tuning budget, compute, independent seeds/replications, metrics, uncertainty and code revision. Write a concise correspondence table: **faithful reproduction / necessary adaptation / genuinely new component**. Confirm the strongest fair alternative, a simple baseline, and negative controls. Lock the evaluator and test data rules. Adequately train baselines on **identical data and fair optimization budget**. A toy may be a debugging pilot but must not substitute for a hypothesis's published benchmark without explicit justification.

### P4. Minimal deterministic experiment harness
The executable reference implementation is **`runner.py` alongside this SKILL.md**, not an imagined future artifact. It uses only Python standard library. Run `python .agent/skills/hypothesis-first-research/runner.py path/to/protocol.json --ledger path/to/results.jsonl` from any repository after registering and reviewing the protocol. The protocol file is a *project's own pre-existing experiment manifest or a single justified new JSON file*; never create manifests for hypothetical studies. Each registered command must invoke the same fixed evaluator by including `{evaluator}`; its checksum is checked before and after every run. Results remain observations, not scientific decisions.

Borrow from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) its **small editable surface, immutable evaluation code, fixed resource rules, concise per-run ledger, and keep/discard reason**. **Reject** its default endless loop, automatic hypothesis mutation, evaluation-set overfitting, fixed five-minute training policy (unless scientifically relevant), and one-metric-wins-science interpretation.

Prefer reusing existing runnable Python/notebook code and existing logging. **Only introduce a minimal Python runner when repeated manual execution would otherwise cause inconsistent evaluation.** It must:
1. Accept a versioned experiment spec: hypothesis ID, dataset and split hashes, baselines, code commit, model/config, seeds, budget, evaluator ID, predeclared metric and threshold.
2. Validate inputs and prevent train/test overlap, missing baseline evidence, evaluation-code mutation, and unregistered protocol changes; fail closed with actionable errors.
3. Execute the **same sealed evaluation** for each registered method; write per-run status `PASS/FAIL/CRASH`, metrics and resource consumption to one machine-readable ledger (TSV or JSONL), with no claim about a scientific verdict.
4. Generate a comparator table from real recorded results, with independent repeats and uncertainty. Distinguish infra pilot from main test; never relabel a pilot as confirmation.
5. Stop after the registered runs. The runner **must not generate new hypotheses, rewrite code autonomously, rerun indefinitely, or declare improvement without the scientific gate**.

**Determinism boundaries:** Explicit seeds, pinned software/hardware details, data and evaluator hashes and stable ordering improve repeatability; GPU kernels and some libraries may remain nondeterministic. Declare known sources of variance rather than promising bitwise-identical outputs. Use actual run evidence only; never claim a Python harness was executed unless execution logs exist. No extra script is required just because the architecture allows one.

### P5. Scientific adjudication, not generic caution
For each H report: **result -> closest published comparator -> effect size -> uncertainty -> alternative explanations/ablations -> verdict -> scope**. Verdicts:
- **SUPPORTED:** all preregistered primary conditions met with meaningful result; state precisely what outperformed what, on which benchmark, under which conditions. Put that specific finding in the paper's contribution.
- **REJECTED:** adequate test contradicts or fails preregistered success criterion; identify what was ruled out.
- **INCONCLUSIVE:** benchmark/baseline/implementation/statistical adequacy prevents judgment; identify exactly what remains missing.

Do not follow a supported finding with a blanket 'trust nothing' disclaimer. Do not claim global proof or state-of-the-art superiority from a narrow toy. Do not hide failed conditions. Additional experiments are allowed **only** when a named unresolved claim predicts a discriminating observation that may change a registered decision; record approval and reuse existing code and notebook. If exploratory findings motivate H2-H4, register with fresh rationale before follow-on confirmatory evaluation.

### P6. Paper-style notebook and evidence-led artifacts
The main notebook should read like an empirical paper, with: **Abstract; Research question and hypotheses; Related work/SOTA; Source basis and proposed method; Published-protocol correspondence; Implementation; Results against published and retrained baselines; Statistical/ablation analysis; Hypothesis verdict; Limits and boundary of validity; Conclusion; References and reproducibility manifest.** Every code/experiment section must have a research motivation and interpretation. Results must be generated by executed code, not placeholder tables.

**After substantive results**, update one architecture Markdown describing the *actually tested* method, one consolidated notebook and the existing manuscript. A scientifically sound negative result may also be documented; no need to fabricate superiority. If a project's format is not notebooks, follow the same argument in its native artifact.

## Durable checkpoint and task contract
Keep **one canonical state section** in an existing research charter or notebook (or create one record only when no suitable artifact exists):
- Original researcher wording, question, lineage with exact source and the approved H1-H4 register.
- Paper/benchmark anchor; locked evaluation protocol, fairness decisions and thresholds.
- Accepted/rejected decisions with reasons, responsible party and date.
- Run ledger pointer and digest: git SHA, data hash, config, seeds, metrics, uncertainty, failures, and scientific verdict.
- Stage status P0-P6, currently unresolved scientific point, **one next discriminating action**.
- Inventory of changed/approved/prohibited artifacts and justification.

For GitHub work, use **one issue per coherent research goal**, update its checklist and dated decision log rather than opening one issue per cell or run. Across sessions, read this issue and the canonical state first. Outside GitHub, reuse the host task tracker or checkpoint, without spawning a new tool ecosystem.

## Creation and change-control budget
Default **no files** for P0-P2; edit existing notebook/record for P3-P5; only then architecture and manuscript. This skill ships a single reusable stdlib Python runner; **reuse it** rather than create a runner per study. Additional scripts require demonstrated necessity. Before every edit, explain what decision it enables and why an existing file cannot serve it. After a change, report exact modified paths and whether code/experiments were actually run. Do not auto-delete unrelated material or introduce dependencies without permission. Researcher-approved changes to the question or protocol must appear in the checkpoint before implementation.

## Response standard
Every progress report states **current question, source anchor, hypothesis, evidence, verdict/status, decision gate, next action, and checkpoint location**. Never conflate completed code with a supported scientific contribution. When evidence does support the claim, clearly say so.
