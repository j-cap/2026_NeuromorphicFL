# P17: Fashion-MNIST final-horizon campaign

P17 completes ten matched seeds for the Fashion-MNIST MLP and CNN at the
final horizons selected by P16: 1,500 rounds for the MLP and 1,800 rounds for
the CNN. Each architecture has seven frozen operating points (five
quality-selected and two traffic-matched), for 70 runs per architecture.

The runner is resumable. It skips a run only when both its summary and history
CSV exist. Reissuing the same campaign command therefore continues after an
interruption. Writes are atomic, and the dataset loader is protected by a
portable inter-process lock with stale-lock recovery.

## One-time setup on Windows

Run these commands from an Anaconda Prompt in the repository root:

```bat
git pull origin main
conda env create -f environment-resnet14.yml
conda activate neuromorphicfl-resnet14
python experiments\p17_fmnist_final_horizons.py prepare-data
python experiments\p17_fmnist_final_horizons.py protocol
```

If the environment already exists, replace `conda env create` with:

```bat
conda env update -f environment-resnet14.yml --prune
```

The script adds the repository's `src` and `experiments` directories to
`sys.path`, so no manual `PYTHONPATH` setting is required.

## Run the campaigns

P17 is CPU-based. The GPU is not used. On a 32-vCPU workstation, begin with
eight processes and one numerical-library thread per process:

```bat
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

python experiments\p17_fmnist_final_horizons.py campaign --architecture mlp --workers 8
python experiments\p17_fmnist_final_horizons.py status
python experiments\p17_fmnist_final_horizons.py aggregate --architecture mlp
```

Commit the MLP checkpoint before starting the CNN:

```bat
git add experiments/results/p17_fmnist_final_horizons
git commit -m "Add P17 MLP final-horizon results"
git push origin main
```

Then run the CNN campaign:

```bat
python experiments\p17_fmnist_final_horizons.py campaign --architecture cnn --workers 8
python experiments\p17_fmnist_final_horizons.py status
python experiments\p17_fmnist_final_horizons.py aggregate --architecture cnn
python experiments\p17_fmnist_final_horizons.py finalize
python experiments\p17_fmnist_final_horizons.py verify
```

If CPU and memory utilization remain comfortable, `--workers 12` or
`--workers 16` may be faster. Do not increase numerical-library thread counts
at the same time because nested parallelism usually slows these small runs.

## Resume, split, or rerun

The normal campaign command resumes automatically. To divide work into seed
batches, use inclusive seed indices:

```bat
python experiments\p17_fmnist_final_horizons.py campaign --architecture cnn --workers 8 --start-seed 0 --end-seed 4
python experiments\p17_fmnist_final_horizons.py campaign --architecture cnn --workers 8 --start-seed 5 --end-seed 9
```

Run one point for diagnosis:

```bat
python experiments\p17_fmnist_final_horizons.py point --point-id fmnist_mlp_event_quality --seed-index 1
```

Use `--force` only when a completed result must deliberately be replaced.

## Final commit

After `verify` passes:

```bat
git add experiments/results/p17_fmnist_final_horizons
git commit -m "Complete P17 Fashion-MNIST final-horizon campaign"
git push origin main
```

The expected final verification message is:

```text
P17 verification passed: 140 matched runs, complete histories, finite metrics
```
