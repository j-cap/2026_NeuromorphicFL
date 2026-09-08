# Manuscript review log

This directory contains immutable, point-in-time reviews of the manuscript.

## Naming convention

Use `NN_YYYY-MM-DD_scope.md`, where:

- `NN` is the next two-digit review number across this directory;
- `YYYY-MM-DD` is the date on which the reviewed snapshot was assessed; and
- `scope` is a short lowercase description, such as `ijcnn_review` or
  `camera_ready_check`.

For example, the review after `01_2026-09-07_ijcnn_review.md` is numbered `02`,
even if both reviews occur on the same date. Do not overwrite or renumber an
earlier review. Each review records the exact Git commit and manuscript path so
that later changes can be compared against the correct snapshot.

