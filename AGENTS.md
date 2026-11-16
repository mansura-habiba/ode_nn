# Research Agent Instructions

Scope: all research conducted in this repository, independent of scientific domain. This file is an entrypoint, not a second copy of the research skill.

## Required local research pipeline

Before starting or resuming research, read [Hypothesis-First Research](.agent/skills/hypothesis-first-research/SKILL.md), the existing canonical research checkpoint, and the current research GitHub issue (if any). Follow the P0–P6 gates: recover context and original intent; verify primary literature and model lineage; agree on at most four falsifiable hypotheses; anchor evaluation in published experimental protocols; execute controlled comparisons; issue a supported/rejected/inconclusive verdict; only then prepare evidence-based notebooks, architecture and paper.

Preserve the researcher's decisions between sessions. Keep the research checkpoint in an **existing** canonical document/notebook where possible; update an existing GitHub issue instead of spawning subtasks. Do not replace the original method with a convenient surrogate, change hypotheses after seeing the held-out test results, or claim an advantage without sufficiently trained baselines. Source facts must be checked against actual papers, not inherited from earlier assistant summaries.

## Optional external research skills (discover before use)

If installed, available, and relevant, the agent **may use individual skills** from [KalarisLabs/research-agent-skills](https://github.com/KalarisLabs/research-agent-skills) to strengthen stages of the local pipeline:

- [literature-review](https://github.com/KalarisLabs/research-agent-skills/tree/main/skills/literature-review) for a scoped search, systematic screening and evidence synthesis.
- [citation-verification](https://github.com/KalarisLabs/research-agent-skills/tree/main/skills/citation-verification) to verify bibliographic identity and whether a citation actually supports a claim.
- [experimental-design](https://github.com/KalarisLabs/research-agent-skills/tree/main/skills/experimental-design) for controls, replication, confounders, and comparative protocol design.

**Availability rule:** discover the skill and read its instructions before use. A link to an upstream repository does not imply it is installed or executable. If unavailable, perform the same evidence checks with accessible tools; do not silently claim to have run an external skill. Never install, clone, add dependencies, or execute remote scripts without a concrete need and appropriate authorization. Review upstream license and instructions for code reuse.

**Precedence rule:** local research pipeline and researcher-approved scientific constraints take priority. External skills are optional *methods*, not an autonomous workflow controller. Do not import upstream defaults that generate multiple files, mandatory diagrams/PDFs, new folders or endless experimentation.

## Execution and minimal artifacts

For registered empirical comparisons, reuse [the local Python runner](.agent/skills/hypothesis-first-research/runner.py) when useful. It is an experiment executor/log, not a scientist or hypothesis judge. Lock the evaluator and experimental protocol before execution; do not let an automated loop search the held-out test set.

Default to editing existing artifacts, one coherent GitHub issue, and one main notebook per study. Before changing any file or starting a new experiment, state the decision it enables and why reuse is insufficient. Report exact changed files, recorded evidence, scientific verdict and the next discriminating action.

## Portability

When these instructions are copied to another repository, update the relative skill path and use that project's existing notebook, checkpoint and issue tracker. Use this root `AGENTS.md` as the repository entrypoint; for other agent implementations, configure loading explicitly when necessary.
