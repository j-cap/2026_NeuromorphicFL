# GitHub Actions execution policy

This repository separates lightweight validation from computational research
campaigns. The policy prevents ordinary documentation and paper commits from
rerunning frozen experiment matrices.

## Automatic pull-request checks

Only these workflows run automatically, and only when their scoped paths
change:

| Workflow | Purpose | Timeout |
|---|---|---:|
| IJCNN Evidence Check | Regenerate and validate the checksum-frozen paper bundle and theory contract | 30 min |
| IJCNN Manuscript Check | Compile and inspect the conference manuscript | 30 min |
| Compile LaTeX Report | Compile and inspect the living report | 30 min |

They do not also run on push. This avoids duplicate push and pull-request jobs.
Each workflow has a concurrency group that cancels a superseded run for the
same pull request or branch.

## Manual experiment campaigns

Every experiment, tuning, held-out evaluation, and mechanism-audit workflow is
manual-only through `workflow_dispatch`. This includes the historical Experiment
14/15 and Q1--Q3 workflows, the final-baseline campaigns, and P3, P8, T3, and
T4.

To run one intentionally:

1. Open the repository's **Actions** tab.
2. Select the exact campaign.
3. Choose **Run workflow** and the intended branch.
4. Record the run URL and freeze the resulting evidence before relying on it.

Do not rerun a frozen campaign merely because code, documentation, or the paper
changed. First identify a scientific or reproducibility reason and record it in
the relevant audit or plan.

## Resource controls

- A newer run of the same workflow and branch cancels its predecessor.
- Automatic jobs time out after 30 minutes.
- Manual campaigns time out after 180 minutes unless an existing, stricter
  timeout applies.
- Uploaded artifacts are retained for seven days.
- Submission evidence that must remain available is committed under
  `experiments/results/` or `paper/ijcnn2027/evidence/`; workflow artifacts are
  transport files, not the permanent source of truth.

## Policy for new workflows

New computational campaigns are manual-only by default. A workflow may be
automatic only when it is a bounded validation check, has narrow path filters,
contains no experiment matrix, and completes comfortably within 30 minutes.
