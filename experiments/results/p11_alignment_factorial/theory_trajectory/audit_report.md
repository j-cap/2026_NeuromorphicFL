# P11 Theory-to-Trajectory Audit

These are **31 selected audit checkpoints per seed**, not all training rounds.
The normalized sums below are **sampled diagnostics**, not full-horizon theorem terms.
No conditional expectation or global smoothness constant is certified.

## Condition-level mean metrics

| regime | local_steps | n_seeds | alignment_positive_pct_mean | alignment_positive_pct_sd | objective_decrease_pct_mean | objective_decrease_pct_sd | aligned_non_descent_pct_mean | aligned_non_descent_pct_sd | conditional_non_descent_pct_mean | conditional_non_descent_pct_sd | normalized_sample_defect_mean | normalized_sample_defect_sd | normalized_sample_event_scale_mean | normalized_sample_event_scale_sd | mean_curvature_remainder_mean | mean_curvature_remainder_sd | positive_remainder_pct_mean | positive_remainder_pct_sd | event_remainder_pearson_mean | event_remainder_pearson_sd | positive_defect_pct_mean | positive_defect_pct_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iid | 1 | 10 | 96.13 | 1.36 | 96.13 | 1.36 | 0 | 0 | 0 | 0 | 0.06309 | 0.002197 | 20.01 | 0.591 | 0.0004129 | 7.228e-05 | 96.77 | 0 | 0.6707 | 0.08819 | 12.26 | 2.04 |
| iid | 5 | 10 | 99.03 | 2.177 | 94.84 | 4.08 | 4.194 | 2.656 | 4.26 | 2.721 | 0.02143 | 0.0006711 | 83.7 | 1.906 | 0.007144 | 0.0006265 | 99.03 | 1.558 | 0.8577 | 0.01963 | 8.065 | 3.802 |
| strong | 1 | 10 | 97.74 | 1.558 | 87.42 | 4.42 | 10.32 | 4.511 | 10.55 | 4.616 | 0.02906 | 0.004295 | 158.1 | 4.333 | 0.0096 | 0.001001 | 99.68 | 1.02 | 0.3883 | 0.1467 | 7.742 | 1.666 |
| strong | 5 | 10 | 99.03 | 1.558 | 69.68 | 9.275 | 29.35 | 8.528 | 29.7 | 8.825 | 0.02208 | 0.003436 | 306.6 | 4.625 | 0.01666 | 0.001058 | 100 | 0 | 0.4514 | 0.03323 | 6.452 | 3.4 |

## Paired seed contrasts (95% Student-t intervals)

| comparison | metric | n | mean_difference | sd_difference | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| strong_minus_iid_E1 | alignment_positive_pct | 10 | 1.613 | 2.281 | -0.01882 | 3.245 |
| strong_minus_iid_E5 | alignment_positive_pct | 10 | 1.421e-15 | 3.041 | -2.176 | 2.176 |
| E5_minus_E1_iid | alignment_positive_pct | 10 | 2.903 | 2.825 | 0.8827 | 4.924 |
| E5_minus_E1_strong | alignment_positive_pct | 10 | 1.29 | 1.666 | 0.09868 | 2.482 |
| strong_minus_iid_E1 | objective_decrease_pct | 10 | -8.71 | 4.038 | -11.6 | -5.821 |
| strong_minus_iid_E5 | objective_decrease_pct | 10 | -25.16 | 9.832 | -32.19 | -18.13 |
| E5_minus_E1_iid | objective_decrease_pct | 10 | -1.29 | 4.612 | -4.59 | 2.009 |
| E5_minus_E1_strong | objective_decrease_pct | 10 | -17.74 | 7.488 | -23.1 | -12.39 |
| strong_minus_iid_E1 | aligned_non_descent_pct | 10 | 10.32 | 4.511 | 7.096 | 13.55 |
| strong_minus_iid_E5 | aligned_non_descent_pct | 10 | 25.16 | 8.841 | 18.84 | 31.49 |
| E5_minus_E1_iid | aligned_non_descent_pct | 10 | 4.194 | 2.656 | 2.294 | 6.093 |
| E5_minus_E1_strong | aligned_non_descent_pct | 10 | 19.03 | 7.967 | 13.33 | 24.73 |
| strong_minus_iid_E1 | normalized_sample_defect | 10 | -0.03403 | 0.004606 | -0.03733 | -0.03074 |
| strong_minus_iid_E5 | normalized_sample_defect | 10 | 0.0006531 | 0.003451 | -0.001816 | 0.003122 |
| E5_minus_E1_iid | normalized_sample_defect | 10 | -0.04166 | 0.002354 | -0.04334 | -0.03998 |
| E5_minus_E1_strong | normalized_sample_defect | 10 | -0.006975 | 0.005659 | -0.01102 | -0.002927 |
| strong_minus_iid_E1 | normalized_sample_event_scale | 10 | 138.1 | 4.522 | 134.8 | 141.3 |
| strong_minus_iid_E5 | normalized_sample_event_scale | 10 | 222.9 | 4.737 | 219.5 | 226.2 |
| E5_minus_E1_iid | normalized_sample_event_scale | 10 | 63.69 | 2.007 | 62.26 | 65.13 |
| E5_minus_E1_strong | normalized_sample_event_scale | 10 | 148.5 | 4.859 | 145 | 152 |
| strong_minus_iid_E1 | mean_curvature_remainder | 10 | 0.009187 | 0.001001 | 0.008471 | 0.009903 |
| strong_minus_iid_E5 | mean_curvature_remainder | 10 | 0.009516 | 0.001283 | 0.008598 | 0.01043 |
| E5_minus_E1_iid | mean_curvature_remainder | 10 | 0.006731 | 0.0005917 | 0.006308 | 0.007154 |
| E5_minus_E1_strong | mean_curvature_remainder | 10 | 0.00706 | 0.001365 | 0.006084 | 0.008036 |

Interpretation: a positive-alignment round can increase the objective;
the realized curvature remainder is an exact algebraic difference.
Its relation to $M q_r^2 N_r$ is descriptive, not an identified
smoothness bound. Seed-paired intervals describe the audited seeds
and should not be interpreted as independent-checkpoint inference.
