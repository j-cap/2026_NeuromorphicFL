# Experiment 14B reproducibility audit

A late tensor-diagnostic rerun exposed that nominally identical multiclass event runs could diverge when NumPy/BLAS used multiple linear-algebra threads. The event count changed only slightly, but the thresholded hybrid trajectory amplified those floating-point differences into visible final-metric differences.

The report-ready Experiment 14B campaign therefore uses a standardized execution environment:

- Python 3.11
- NumPy 2.4.6
- pandas 3.0.5
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `BLIS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `PYTHONHASHSEED=0`

Under this environment, two independent calls to the selected strong-skew configuration within the full campaign were exactly identical in final training objective, test cross-entropy, accuracy, worst-class accuracy, payload, candidate-event count, and events per message. A second GitHub Actions run on a different runner reproduced the same values exactly:

- test cross-entropy: `0.6167917251586914`
- test accuracy: `0.7934`
- worst-class accuracy: `0.517`
- payload: `34,723,776` bits
- candidate events: `2,170,236`
- messages: `2,404`
- events/message: `902.7603993344426`
- ever-fired fraction: `1.0`

Only the standardized single-thread results should be used as observed Experiment 14B results. Earlier multithreaded audit values are diagnostic/preliminary and are superseded by the deterministic campaign.
