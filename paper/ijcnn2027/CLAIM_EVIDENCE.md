# IJCNN claim--evidence matrix

This matrix controls what may enter the conference manuscript. “Ready” means
that suitable evidence exists in the repository; it does not mean that the
material has already been compressed into paper form.

| ID | Intended claim | Primary evidence | Paper placement | Status / boundary |
|---|---|---|---|---|
| C1 | Event-FedAvg is a completely specified multi-step FedAvg communication operator. | Compact transition and round procedure in `paper/ijcnn2027/main.tex`; full semantics in `report/sections/11_theory_t1_formal_semantics.tex`; correspondence audit in `P4_THEORY_AUDIT.md` | Method | Extracted and ready under synchronous full participation. |
| C2 | The targeted literature search did not identify the complete frozen operator; relative to the closest Strom residual-pulse predecessor, its defining differences are leakage, full reset, independent trigger/update resolution, and exact synchronization semantics. | `paper/ijcnn2027/RELATED_WORK.md`, `report/sections/10b_novelty_equivalence_audit.tex`, `report/sections/10f_final_matched_baseline_campaign.tex`, and the traffic-matched Strom control | Introduction, related work, results | Ready for the 2026-09-04 search snapshot and qualified as an absence-search result; refresh at P10. Novelty belongs to the complete operator, not its ingredients. |
| C3 | Sparse replay with checkpoint fallback gives exact post-catch-up client synchronization and a complete bidirectional accounting model. | `report/sections/10d_bidirectional_protocol.tex` and T1 communication semantics | Method and experiment protocol | Ready under the evaluated synchronous full-participation protocol. |
| C4 | Event-FedAvg occupies a strong communication--performance operating point on the established Fashion-MNIST MLP and CNN campaigns. | `paper/ijcnn2027/evidence/fmnist_master_results.csv`, generated from the frozen campaign summaries; audit details in `EVIDENCE_FREEZE.md` | Main frontier figure and result table | Ready; distinguish quality-selected from traffic-matched comparisons and avoid uniform dominance language. |
| C5 | Post-reset encoder state is bounded and the realized event stream has a pathwise evidence-variation budget. | Compact proposition in `paper/ijcnn2027/main.tex`; proof in `report/sections/12_theory_t2_encoder_properties.tex`; randomized checks in `check_theory_contract.py` | Theory | Extracted and ready; detailed first-passage and sign-reliability results remain outside the core. |
| C6 | Smoothness yields an exact one-step descent inequality, and aggregate alignment with explicit defect yields conditional finite-horizon stationarity. | Compact proposition in `paper/ijcnn2027/main.tex`, full proof in `report/sections/13_theory_t3_optimization_interface.tex`, and audit in `P4_THEORY_AUDIT.md` | Theory | Extracted and explicitly conditional; no unconditional or experiment-covered asymptotic convergence claim. |
| C7 | Aggregate Event-FedAvg pulses are usually globally aligned on the frozen strong-non-IID MLP, with weaker late-stage margins. | Compact audit paragraph in `paper/ijcnn2027/main.tex`; exact full-client-gradient baseline in `report/sections/14_theory_t4_defect_schedule.tex` | Theory/results bridge | Extracted as finite-trajectory evidence only; the earlier 5000-example T3a diagnostic is not used for headline values. |
| C8 | Replacing the empirical quantum schedule by an asymptotically admissible exponent does not by itself improve finite-horizon late descent. | `report/sections/14_theory_t4_defect_schedule.tex` and the three T4 schedule CSVs | Limitations or technical report | Ready negative result; not a headline contribution. |
| C9 | The communication--performance result transfers from Fashion-MNIST to CIFAR-10 with a compact conventional CNN. | `paper/ijcnn2027/evidence/cifar10_master_results.csv`, generated from the frozen P3 campaign; protocol and interpretation in `P3_PROTOCOL.md` and `P3_DECISION.md` | Main frontier figure and result table | Ready; three held-out seeds support the tested operating-point claim, not statistical significance or broad dataset generalization. |
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

P3 closed with a pass. Event-FedAvg is nondominated on the frozen CIFAR-10
compact-CNN campaign and has the best mean test CE and accuracy among the
quality-selected methods while using the least total traffic. C9 is limited to
this concrete transfer; worst-class accuracy and three-seed uncertainty remain
visible qualifications.

## Paper-level evidence gate

Before manuscript freeze, every sentence containing a comparative,
generalization, convergence, synchronization, communication, or novelty claim
must identify one of the claim IDs above. New claim IDs require an explicit
evidence source and boundary.
