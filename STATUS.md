# Current task status

## Git upload exclusions (2026-09-27)

- COMPLETE: expanded `.gitignore` to exclude checkpoints, generated results,
  TensorBoard runs, model binaries, datasets/arrays, archives, and local Python
  environments/caches/build outputs. Existing local artifacts are preserved.
- Validation: all 90 existing files at least 10 MiB (3,845.25 MiB total, all
  `.pt` checkpoints) are ignored; no tracked file is at least 10 MiB. Representative
  artifact paths are ignored while source, configs, and research reports remain
  eligible for tracking. `git diff --check` passed.
- Git cannot ignore files by size; 10 MiB is this audit's threshold, while rules
  exclude known artifact paths/types regardless of size. No history rewrite or
  training was performed. Previously tracked Python bytecode is unaffected by
  adding ignore rules.

Date: 2026-09-26

State: COMPLETE — declared ResNet-18 / full BloodMNIST non-IID experiment,
artifact validation and report. Results are mixed; broad robustness is not established.

- User-confirmed design: fixed uncompressed FedAvg control; genuine FedProx
  mu=0.01 and FedProx+FP16; Dirichlet alpha=[0.1,0.3,1.0], one shared seed.
- ResNet-18 defaults, all official samples, 8 rounds, 3 clients, one local epoch,
  batch 32, AdamW lr=0.0001, weight decay=0.01; sequential deterministic CPU.
- Completed: repository inspection, source/config lock, runner and artifact
  validator, partition feasibility and synthetic timing preflight.
- Pre-execution: 44 offline tests passed (24.901 s), and a separate same-family
  technical reviewer approved the locked design before training.
- COMPLETE EXECUTION: `results/non_iid_study/suite_20260926T103613Z`, from
  2026-09-26 10:36:13 UTC to 13:05:02 UTC, worker and supervisor exit code 0.
  All nine matrix cells and the separate FedAvg alpha=0.3 replay completed.
  Replay exactly matches selected tensors, test report and all-round metrics.
  No training was repeated when resuming; the completed suite was audited as-is.
- At 18:09 local, a requested 50-second monitoring wait returned after 688.118 s;
  cause unverified, recorded in suite monitoring_notes.json; no speed claim.
  Temporary Windows system-idle request was released at 13:05:02 UTC, confirmed
  by idle_request.json; no global power configuration or display setting changed.
- Seed 2026 selected before outcomes to avoid singleton BatchNorm batches;
  8 threads selected using synthetic timing only. See locked protocol/preflight.
- Completed: final report generation, PNG visual inspection, and technical
  review. No unfinished step remains for the declared matrix and report.
- Results: FedProx meets the prespecified practical gain criterion only at
  alpha=1.0 (+0.9354 accuracy / +1.1758 macro-F1 percentage points); macro-F1
  falls at alpha=0.1 and 0.3. FedProx+FP16 saves 16.6608% total payload but
  fails the all-alpha quality criterion (alpha=0.1 macro-F1 falls 2.2136 points).
- Final scope: measured improvement is supported at alpha=1.0 only; the
  prespecified consistent-gain and all-alpha communication-quality criteria
  did not pass. Negative results remain prominent; no test-driven retuning.
  Old mixed IID/BloodCNN research artifacts are retired and are not current evidence.
- Prepared while training: `scripts/report_non_iid_study.py` generates audited
  tables, direct codec attribution and PNG/SVG only after matrix/replay completion.
  Six targeted report regression tests cover inconsistent evidence, self-replay,
  unrelated replay references and direct codec attribution. README links the
  revised protocol and measured results.

## Current deliverables and final validation

- `research/non_iid_report.md`: all nine results, per-alpha contrasts, direct
  FedProx+FP16 versus FedProx effects, split audit, limits and 11 ARS fallacy checks.
- `research/non_iid_runs.csv`: exact run metrics, IDs, partitions and paths.
- `research/non_iid_results.png` / `.svg`: scientific figure, visually inspected.
- `research/non_iid_protocol.md`, `research/non_iid_baseline.lock.json`,
  `configs/non_iid_study.yaml`: fixed design and pre-outcome choices.
- `scripts/run_non_iid_study.py`, `scripts/report_non_iid_study.py`,
  `src/fedmedai/non_iid_study.py`: execution, reproducible reporting and audits.
- `python -B -m unittest discover -s tests -v`: all 51 tests passed, 17.052 s
  after the project refresh. Synthetic IID regression tests are not research runs.
- `python -B scripts/report_non_iid_study.py
  results/non_iid_study/suite_20260926T103613Z`: exit 0, full evidence gate passed.
- `git diff --check`: passed. Existing source modifications and current-suite snapshots preserved; retired outputs removed as documented below.
- Final separate-agent technical review APPROVED on 2026-09-26: independently
  checked 9 cells + distinct replay, 240 client-fit records, sample/source/config
  constraints, selected rounds, recomputed metrics, payloads, CSV/report/figure.
  Same model family/shared context; not external or cross-model peer review.

## Project refresh requested after completion

- Current research and communication reports use only the completed ResNet-18
  Dirichlet suite; no retraining or changes to frozen training/source snapshots.
- Communication CLI/config now targets the current matrix; all report entry
  points regenerate both non-IID and communication deliverables from one audit.
- README, implementation plan and repository guidance distinguish completed
  results from optional future work and general-purpose IID API support.
- Cleanup COMPLETE: 21 old runs removed (12 IID + 9 Dirichlet from the retired
  mixed BloodCNN suite), including corresponding checkpoint/TensorBoard data.
  Ten explicitly checked output subtrees removed (including old IID preflight
  logs/config/audits), plus empty parent directories; byte counts and details
  in `research/artifact_cleanup.json`. No actual research IID manifests remain.
- All 10 remaining manifests belong to the current completed Dirichlet suite;
  its index hash, immutable snapshots, frozen source hashes and replay are intact.
- Both CSVs have the same 9 rows: ResNet-18, Dirichlet, 8 rounds, seed 2026.
  The communication report/CSV/figure now contain current results only.
- `configs/config.yaml`, `communication_study.yaml`, `non_iid_study.yaml` all
  resolve exactly to the frozen baseline; covered by a new config regression.
- Initial refresh test run exposed two synthetic legacy-fixture errors after
  repointing communication config. Fixture was made self-contained; all 51
  offline tests pass, with no change to training or artifact-validation code.
- Current report regeneration and communication runner `--help` both exit 0;
  no training was launched during this refresh. `git diff --check` passes.
