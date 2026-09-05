# P4 publishable-theory extraction and audit

## Outcome

P4 is closed. The six-page manuscript now contains the exact full-participation
Event-FedAvg transition, a compact round procedure, two unconditional encoder
properties, the deterministic one-step descent inequality, the conditional
finite-horizon stationarity bound, and the exact-gradient alignment audit.

No new theorem is introduced here. The extraction preserves the statements
proved in T1--T3 and the empirical boundary established in T4.

## What remains in the main paper

1. Weighted multi-step FedAvg delta and evidence conversion
   $a_i^r=p_i\delta_i^r/\eta_r$.
2. Leaky pre-trigger state, one threshold test, signed pulse, and full reset.
3. Algebraic server aggregation $C^r=\sum_i c_i^r$ and independently scheduled
   update $w^{r+1}=w^r+q_rC^r$.
4. Exact replay/checkpoint synchronization semantics.
5. Post-reset state invariance and the pathwise coordinate-event budget.
6. Aggregate alignment identity, event-energy bound, one-step descent
   inequality, and conditional finite-horizon stationarity.
7. One concise exact-gradient audit of the alignment assumption.

Detailed sign-reliability derivations, first-passage times, the T4 defect
decomposition, the schedule experiment, observer-effect diagnostics, and the
PL specialization remain in the living report.

## Independent consistency audit

| Item | Check | Outcome |
|---|---|---|
| Client weighting | $p_i$ enters the evidence state once; the nonlinear pulse is not weighted again at the server. | Consistent with implementation. |
| Sign convention | Local SGD makes $\delta_i^r=-\eta_r\sum_e g_{i,e}^r$; adding a pulse with the delta sign is descent-aligned when it opposes the global gradient. | The descent term is correctly $-q_rA_r$. |
| Round indexing | $w^r,z_i^r$ are start-of-round/post-reset states; local work sees $w^r$ and emitted pulses produce $w^{r+1},z_i^{r+1}$. | No same-round causal leak. |
| Threshold/reset | The implementation tests `abs(state) >= threshold` once and assigns zero on fired coordinates. | Matches full reset; no threshold subtraction or repeated firing. |
| Event energy | Coordinatewise Cauchy--Schwarz gives $\|\sum_i c_i^r\|_2^2\leq M\sum_i\|c_i^r\|_2^2=MN_r$. | Correct before or after pulse cancellation. |
| Event budget | Full reset creates disjoint reset-to-event blocks; all leak products lie in $[0,1]$. | Pathwise bound is valid without stochastic assumptions. |
| Conditioning | $\mathcal F_r$ is fixed before local round randomness, so $w^r$ and $\nabla F(w^r)$ are measurable while $A_r,N_r$ may remain random. | Conditional expectation and telescoping are valid. |
| Initial model | The finite-horizon display uses deterministic $w^1$, as in the implementation. | No missing expectation on the initial objective. |
| Quantum schedule | The finite-horizon result only requires $q_r>0$. Weighted asymptotic stationarity additionally needs $\sum q_r=\infty$, $\sum q_r^2<\infty$, and $\sum q_r\beta_r<\infty$. | Empirical exponents 0.1 and 0.3 are not presented as asymptotically admissible. |
| Synchronization | Sparse replay reconstructs the same algebraic sum as the server; checkpoint fallback sends the resulting model. | Communication accounting does not alter the learning transition. |

`check_theory_contract.py` independently stress-tests the state invariant,
pathwise event budget, aggregate alignment identity, and event-energy bound on
random instances. It also guards the exact implementation statements that
realize evidence scaling, leakage, thresholding, signed aggregation, and full
reset.

## Empirical assumption audit retained in the paper

The authoritative audit is T4's exact full-client empirical-gradient baseline,
not T3a's 5000-example reference gradient. Across three held-out partitions and
31 independently replayed rounds per partition, the weighted trajectory
alignment ratio is $35.42\pm1.35$ and the objective decreases on
$75.27\pm7.45\%$ of snapshots. All 93 emitted-state signs agree with the
current local-delta signs. Heterogeneity is the principal adverse signed term,
but cancellation makes its componentwise absolute upper bound uninformative.

These observations support plausibility of the alignment interface on the
tested trajectory. They do not prove the conditional expectation, a uniform
positive alignment constant, or unconditional convergence.
