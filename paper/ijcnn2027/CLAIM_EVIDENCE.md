# IJCNN claim--evidence matrix

This matrix controls what may enter the conference manuscript. “Ready” means
that suitable evidence exists in the repository; it does not mean that the
material has already been compressed into paper form.

| ID | Intended claim | Primary evidence | Paper placement | Status / boundary |
|---|---|---|---|---|
| C1 | Event-FedAvg is a completely specified multi-step FedAvg communication operator. | `report/sections/11_theory_t1_formal_semantics.tex` and the frozen baseline implementation | Method | Ready; main paper needs compact pseudocode rather than full T1 prose. |
| C2 | The operator is distinct from its closest residual-pulse predecessor through leakage, full reset, and independent trigger/update resolution. | `report/sections/10b_novelty_equivalence_audit.tex`, `report/sections/10f_final_matched_baseline_campaign.tex`, and the traffic-matched Strom control | Introduction, related work, results | Qualified ready; novelty belongs to the complete operator, not its ingredients. |
| C3 | Sparse replay with checkpoint fallback gives exact post-catch-up client synchronization and a complete bidirectional accounting model. | `report/sections/10d_bidirectional_protocol.tex` and T1 communication semantics | Method and experiment protocol | Ready under the evaluated synchronous full-participation protocol. |
| C4 | Event-FedAvg occupies a strong communication--performance operating point on the established Fashion-MNIST MLP and CNN campaigns. | `report/sections/10f_final_matched_baseline_campaign.tex` and its result artifacts | Main frontier figure and result table | Ready; distinguish quality-selected from traffic-matched comparisons and avoid uniform dominance language. |
| C5 | Post-reset encoder state is bounded and the realized event stream has a pathwise evidence-variation budget. | `report/sections/12_theory_t2_encoder_properties.tex` | Theory | Ready; retain theorem statements and move detailed proofs out of the six-page core. |
| C6 | Smoothness yields an exact one-step descent inequality, and aggregate alignment with explicit defect yields conditional finite-horizon stationarity. | `report/sections/13_theory_t3_optimization_interface.tex` | Theory | Ready but conditional; no unconditional convergence claim. |
| C7 | Aggregate Event-FedAvg pulses are usually globally aligned on the frozen strong-non-IID MLP, with weaker late-stage margins. | Exact full-client-gradient baseline in `report/sections/14_theory_t4_defect_schedule.tex` and `report/data/t4_schedule_baseline.csv` | Compact result-table entry or alignment inset | Ready for finite-horizon evidence only. Use this exact audit as authoritative; do not mix it with the earlier 5000-example T3a diagnostic. |
| C8 | Replacing the empirical quantum schedule by an asymptotically admissible exponent does not by itself improve finite-horizon late descent. | `report/sections/14_theory_t4_defect_schedule.tex` and the three T4 schedule CSVs | Limitations or technical report | Ready negative result; not a headline contribution. |
| C9 | The communication--performance result transfers beyond Fashion-MNIST. | P3 second-benchmark campaign | Main frontier figure and result table | Missing and required before broad transfer language is used. |
| C10 | Event-FedAvg reduces measured hardware energy or real-network latency. | None | Excluded | Not claimable in the conference paper. |
| C11 | The current results cover partial participation and asynchronous FedAvg. | Earlier special-case experiments only; not the frozen final claim | Excluded or limitations | Not claimable without a dedicated final-method campaign. |

## Evidence-selection decisions

### Alignment audit

The paper uses the T4 baseline's exact full-client empirical gradients. The
T3a 5000-example reference audit remains useful historical evidence but should
not supply competing headline values.

### Schedule experiment

The admissible-schedule comparison is retained as a limitation: a formal
Robbins--Monro exponent condition is not a practical finite-horizon cure. Its
full table and decomposition remain outside the main paper unless review makes
them necessary.

### Second benchmark

P3 is the only default missing scientific evidence. Its outcome determines the
breadth of C9 and may require narrowing, but not suppressing, the established
Fashion-MNIST result.

## Paper-level evidence gate

Before manuscript freeze, every sentence containing a comparative,
generalization, convergence, synchronization, communication, or novelty claim
must identify one of the claim IDs above. New claim IDs require an explicit
evidence source and boundary.

