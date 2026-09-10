# P14 — Complete Figure 2 with ten matched seeds

## Purpose

Figure 2 currently displays the complete five-method comparison but reports
three-seed estimates. P14 removes the mismatch with the ten-seed Table 1 by
evaluating every frozen Figure 2 operating point on the same ten matched seed
pairs used by P12.

This is an uncertainty extension, not a new hyperparameter search. Methods,
thresholds, Top-K fractions, local training, round counts, partition rules, and
communication accounting remain frozen.

## Compute matrix

Figure 2 contains 21 unique operating points. P12 already supplies ten
seed-level results for 11 of them. P14 runs the remaining ten points:

- Fashion-MNIST MLP: Strom quality, Sign-EF quality, dense quality, and EF-TopK
  traffic-matched;
- Fashion-MNIST CNN: EF-TopK quality, Sign-EF quality, dense quality, and
  EF-TopK traffic-matched;
- CIFAR-10 compact CNN: Sign-EF quality and EF-TopK traffic-matched.

The original Fashion-MNIST campaign retained only three-seed aggregates, not
all seed-level files. P14 therefore reruns all ten matched seeds for these ten
points rather than combining incompatible aggregate and seed-level evidence.
The new campaign has 100 runs. After aggregation, Figure 2 is backed by 210
seed-level runs: 21 points times 10 seeds.

Seed mappings are identical to P12:

- Fashion-MNIST partition seeds: 2500, 2600, ..., 3400; training seed is
  `70000 + partition_seed`.
- CIFAR-10 partition seeds: 3500, 3600, ..., 4400; training seed is
  `80000 + partition_seed`.

## Workstation commands

From the repository root in the prepared environment:

```text
python experiments/p14_figure2_ten_seed.py validate-protocol
python experiments/p12_headline_ten_seed.py prepare-data --dataset fmnist
python experiments/p12_headline_ten_seed.py prepare-data --dataset cifar10
python experiments/p14_figure2_ten_seed.py campaign
python experiments/p14_figure2_ten_seed.py aggregate
```

The campaign is restart-safe: completed point files are skipped. To split the
work into smaller blocks, use `campaign --start-seed 0 --end-seed 4` and then
`campaign --start-seed 5 --end-seed 9`.

Do not use `--force` unless an existing result has been independently shown to
be invalid.

## Acceptance gate

P14 passes when:

1. all 100 new point/seed combinations are present exactly once;
2. all seed and configuration metadata match the frozen protocol;
3. the combined P12+P14 artifact contains 21 points with 10 seeds each;
4. Figure 2 is regenerated exclusively from the combined ten-seed summary;
5. the manuscript caption and discussion are updated from three to ten seeds;
6. no conclusions are conditioned on whether the added seven seeds improve a
   method's mean result.

The resulting estimates may move in either direction. Their purpose is a
consistent uncertainty assessment across the complete comparison figure.
