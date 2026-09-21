# IJCNN 2027 draft review 02

## Review metadata

- **Review date:** 2026-09-21
- **Reviewed commit:** `80ba352d398396abf1014b9dd9b0b67129408fb8`
- **Manuscript:** `paper/ijcnn2027/main.tex`
- **Review mode:** simulated anonymous IJCNN review plus submission audit
- **Venue basis:** [IJCNN 2027 Call for Papers](https://ijcnn.org/2027/authors/call-for-papers) and [Initial Author Instructions](https://ijcnn.org/2027/authors/initial-author-instructions)

The official instructions place the work directly within federated,
communication-efficient, and event-driven learning. They require IEEE
two-column Letter format, double-blind submission, and at most six pages without
an extra-page charge. Up to four additional pages are permitted at USD 100 per
page. The instructions also require disclosure and citation of AI-generated
text.

## Overall assessment

**Simulated recommendation: weak accept, subject to two important corrections.**

The paper has advanced substantially since Review 01. It now presents a
complete and unusually careful communication protocol, a ten-seed comparison
over four model--dataset combinations, a meaningful ResNet-14 result, a
conditional theorem with an explicit assumption boundary, and an empirical
study intended to connect the theorem to observed trajectories. The principal
result is easy to understand: Event-FedAvg obtains favorable accuracy--traffic
operating points while retaining exact round-to-round synchronization.

The paper is now credible as an IJCNN contribution. Its strongest qualities are
protocol completeness, bidirectional accounting, matched-seed evaluation, and
honest separation of encoder guarantees from optimization assumptions. The
ResNet-14 campaign removes the earlier concern that all evidence comes from
very small models.

Two issues currently prevent a confident accept recommendation. First, the
theorem-facing diagnostics introduced in the theory and discussed in the
results are not displayed in Table II, despite the text saying that the table
reports their numerical values. Second, the novelty claim still relies on a
combination of known ingredients without directly ablating full reset or
persistent temporal accumulation. These are repairable without redesigning the
paper.

## Simulated scorecard

| IJCNN criterion | Assessment | Rationale |
|---|---|---|
| Relevance | **Strong** | Direct fit to federated learning, communication-efficient learning, and event-driven neural processing. |
| Technical quality | **Good** | The protocol, pathwise bounds, decomposition, and conditional theorem are coherent and carefully scoped. |
| Novelty | **Borderline to good** | The combined operator is distinctive, but two defining choices remain experimentally unisolated. |
| Presentation | **Good** | The flow is substantially clearer and shorter. The revision colors and final page count still require submission cleanup. |
| Reproducibility | **Strong** | One-command validation passes, frozen settings are stated, and ten-seed artifacts are retained. |
| Experimental validation | **Good** | Four settings, ResNet-14, five method families, paired seeds, trajectories, and an alignment factorial form a credible package. |
| Theory--experiment connection | **Borderline** | The intended bridge is valuable, but its central defect and curvature diagnostics are absent from the displayed table. |
| Overall | **Weak accept** | Acceptance case is credible once the internal table/text mismatch and novelty ablation gap are closed or explicitly narrowed. |

## Main strengths

1. **The contribution is now visible at paper scale.** The abstract, Figure 1,
   Algorithm 1, and method section communicate the same operator: leaky
   parameter-wise accumulation, thresholded signed events, independent server
   quantum, aggregation, and exact synchronization.
2. **The communication comparison is unusually complete.** Uploads and
   downlinks, packet headers, addresses, requests, checkpoint fallback, and
   initial synchronization are all counted under one rule.
3. **The primary evidence is consistent.** Figure 2 and Table I now use ten
   matched seeds. Dense references appear for every benchmark, and all five
   method families appear in the broad comparison.
4. **The ResNet-14 experiment materially strengthens the paper.** It shows the
   mechanism on a larger conventional architecture and includes full
   communication--accuracy trajectories rather than only endpoints.
5. **The theory is honest about what is and is not proved.** State boundedness
   and the event budget are encoder properties. Optimization progress requires
   an explicit conditional alignment assumption. The manuscript does not call
   the empirical audit a convergence certificate.
6. **The negative findings improve credibility.** Leakage has no measurable
   isolated effect at the selected operating point, Strom is not labeled
   generally unstable, and positive first-order alignment is not equated with
   realized descent.
7. **The repository contract passes.** The complete reproduction command
   validates 11 evidence products, seven visual products, 40 alignment runs,
   1,240 snapshots, 13 manuscript claims, the theory contract, 75 frozen
   checksums, and the workflow policy.

## Major issues

### M1. The theorem diagnostics are discussed but not reported

**Severity: submission blocker, straightforward to fix.**

Equation (15) defines the sampled defect contribution
`\widehat{\mathcal D}_{\mathcal A}` and event-curvature factor
`\widehat{\mathcal V}_{\mathcal A}`. The theory text says that the zero-defect
frequency is also reported. The results section then states that Table II gives
the numerical values. However, Table II contains only
`\widehat\kappa_{\mathcal A}`, positive-alignment and descent frequencies,
`\overline B_{\mathcal A}`, `\overline L_{\mathcal A}`, and the curvature
remainder. It does not show the zero-defect frequency,
`\widehat{\mathcal D}_{\mathcal A}`, or
`\widehat{\mathcal V}_{\mathcal A}`.

The frozen analysis already contains the missing numbers:

| Setting | Zero-defect frequency [\%] | `D_hat` | `V_hat` |
|---|---:|---:|---:|
| IID, E=1 | 87.7 | 0.063 | 20.0 |
| IID, E=5 | 91.9 | 0.021 | 83.7 |
| Strong non-IID, E=1 | 92.3 | 0.029 | 158.1 |
| Strong non-IID, E=5 | 93.5 | 0.022 | 306.6 |

Add these three columns to the generated table, or replace less central columns
with them. The current table caption should also say **audited checkpoints**,
not “replayed rounds,” because replay is already a distinct downlink term in
the method.

### M2. Two defining operator choices remain unablated

**Severity: main novelty risk.**

The paper differentiates Event-FedAvg from Strom and error-feedback methods by
three choices: leakage, full reset, and separation of threshold from server
quantum. The experiments show that leakage is neutral at the selected point and
that one coupled threshold/quantum choice performs badly. They do not isolate:

1. full reset versus subtractive or residual-conserving reset, and
2. persistent cross-round state versus a memoryless per-round trigger.

Without these comparisons, a skeptical reviewer can interpret the gain as
primarily a tuned sign/coordinate server optimizer with a decaying step size.
A compact CIFAR-10 CNN ablation with ten matched seeds is enough. Report final
accuracy and total traffic for the frozen operator, residual reset, memoryless
trigger, and a fairly tuned coupled sibling. If compute or space prevents this,
narrow the novelty language so that the contribution is explicitly the complete
protocol and independent quantum, not empirical validation of every encoder
ingredient.

### M3. The dense-baseline gap needs one sentence of defensive context

**Severity: likely reviewer question.**

Event-FedAvg exceeds dense FedAvg by 5.66 percentage points on ResNet-14 while
using much less traffic. This is attractive, but also surprising for a method
presented primarily as a communication layer. Dense FedAvg uses a selected
constant server gain, while Event-FedAvg receives a decaying quantum schedule
and coordinate-wise sign updates. A reviewer may ask whether the accuracy gain
comes from communication sparsity or a better optimization schedule.

The paper should state directly that Event-FedAvg changes both communication
and the induced server update and therefore does not isolate communication as
the cause of the accuracy improvement. If available, report that the dense
gain sweep and 1,800-round audit were performed and that the chosen dense point
was the best development setting. A scheduled dense learning-rate baseline
would be stronger, but it is not essential if the claim remains an operating-
point comparison rather than superiority caused by event communication.

### M4. The communication saving is protocol-dependent

**Severity: moderate, previously unresolved.**

Conservative bidirectional unicast is clearly defined and fair under the chosen
protocol. The numerical savings nevertheless depend on separate downlinks,
64-bit headers, and ordered replay rather than broadcast or sparse aggregate
downlink. Since no retraining is needed, a small accounting sensitivity for
logical broadcast and sparse aggregate transmission would strengthen the
systems claim. At minimum, state that the ranking has not been established
under alternative networking models.

### M5. The novelty boundary became too compressed

**Severity: moderate acceptance risk.**

The shortened related-work section names the right families but no longer
compares the closest event-triggered FL methods structurally. Because the
ingredients are individually established, the novelty argument needs a compact
comparison of trigger granularity, persistent state, reset semantics, applied
update magnitude, and downlink synchronization. One dense paragraph or a small
comparison table would be more persuasive than a longer literature list.
Sparse Ternary Compression is cited but not evaluated. Briefly explain why the
chosen EF-TopK and Sign-EF baselines cover the relevant residual/sparsity
mechanisms, or add STC if an implementation is already available.

## Theory assessment

The pathwise event-budget proposition appears sound under the stated sampled
round-end threshold semantics. The Cauchy--Schwarz bound correctly converts
client-parameter event counts into an aggregate-norm upper bound. The exact
alignment decomposition is algebraically transparent and useful for diagnosis.

The finite-horizon theorem is also internally coherent. Its conclusion is
conditional on expected aggregate alignment with defect `beta_r`. Because the
defect can absorb unfavorable behavior, this is not an unconditional
convergence theorem for Event-FedAvg. The manuscript states this boundary
clearly enough in the theorem discussion and limitations.

The theorem title “Conditional finite-horizon stationarity” is slightly
stronger than the actual deliverable. “Conditional finite-horizon
gradient-norm bound” would align exactly with the abstract and avoid an
overreading. The choice `kappa_0=8` is a descriptive, post-campaign audit value
chosen below previously observed ratios. The paper correctly avoids presenting
it as a predeclared or independently estimated theorem constant. Preserve that
qualification.

## Experimental assessment

The current evidence is sufficient for an IJCNN mechanism paper. ResNet-14,
ten matched seeds, worst-class accuracy, five comparison families, and complete
traffic accounting are meaningful improvements over Review 01. The remaining
scope limitation is ten-client synchronous full participation on two image
datasets. Partial participation would be especially informative because client
state persistence is central, but it is now future-work material rather than a
condition for acceptance.

Table I is useful but dense. The “selection rule” entries repeat within each
benchmark, and Sign-EF appears only for ResNet-14 in the primary table. This is
acceptable because Figure 2 is the broad five-family comparison, provided the
text makes that division explicit. The very poor Strom ResNet result should
remain accompanied by the current careful statement that it characterizes this
synchronous adaptation rather than the original asynchronous method generally.

The alignment factorial is valuable because it connects heterogeneity and
local depth to the decomposition. Its role should remain diagnostic. Once M1
is fixed, it will form a credible theory--experiment bridge.

## Presentation and submission audit

1. **Revision colors must be disabled before submission.** The source contains
   87 uses of blue, violet, red, or green revision macros. Keep the macros for
   review, but add a single final-mode switch that renders all revision commands
   in black.
2. **Page count must be verified in a complete IEEE environment.** This local
   environment lacks `IEEEtran.cls`, so the compressed draft could not be
   rendered here. The official base limit is six pages. Seven pages are allowed
   but incur the stated extra-page charge.
3. **The AI disclosure is incomplete under the published IJCNN instructions.**
   The acknowledgment discloses ChatGPT and Codex, but the instructions also
   require a citation to the AI system in sections containing AI-generated
   text. Add a suitable citation and precise disclosure before submission.
4. **Double-blind status is otherwise appropriate.** The author block is
   anonymous, and the acknowledgment does not identify the authors.
5. **Figure 1 remains information-dense.** Reducing its width saves prominence
   but also reduces text size. Confirm all equations and labels at 100% print
   scale. If anything is below roughly 8 pt, shorten secondary text rather than
   shrinking the figure further.
6. **Terminology is mostly consistent.** Replace “31 replayed rounds” in the
   Table II caption with “31 independently audited checkpoints.”

## Comparison with Review 01

| Review 01 concern | Current status |
|---|---|
| Reproduction manifest failed | **Resolved** |
| Frozen Event-FedAvg settings missing | **Resolved** |
| Figure used three seeds while Table I used ten | **Resolved** |
| Dense headline comparisons absent from Table I | **Resolved** |
| Validation limited to small models | **Largely resolved by ResNet-14** |
| Conditional theorem easy to overread | **Substantially improved** |
| Full-reset and persistent-state ablations absent | **Open** |
| Communication-accounting sensitivity absent | **Open** |
| Related-work novelty boundary too thin | **Still open after compression** |
| Figure 1 final-size legibility | **Needs final PDF check** |

## Prioritized action plan

### P0 — before the next review

1. Put zero-defect frequency, `D_hat`, and `V_hat` into Table II and fix its
   “replayed rounds” terminology.
2. Verify the page count and final-size figure readability in Overleaf or a
   complete IEEE TeX environment.
3. Add a final black-text mode and complete the IJCNN AI disclosure/citation.
4. Add the full-reset and memoryless ablations, or narrow the novelty claim
   explicitly.

### P1 — highest expected effect on acceptance

5. Add one defensive sentence explaining that Event-FedAvg changes the induced
   server optimizer as well as communication, and document the dense sweep.
6. Strengthen the structural comparison with the closest event-triggered and
   residual methods.
7. Add a no-retraining communication-accounting sensitivity if space permits.

## P0/P1 implementation status

Updated on 2026-09-21 after the review was written:

| Item | Status | Resolution |
|---|---|---|
| P0.1 theorem diagnostics | **Completed** | Table II now reports zero-defect frequency, `D_hat`, and `V_hat`, and uses “independently audited checkpoints.” |
| P0.2 page and Figure 1 audit | **Completed** | The IEEE Letter build is eight pages including references. Figure 1 was inspected at final PDF scale and remains readable. |
| P0.3 submission color and AI disclosure | **Completed in source** | One switch renders every revision macro in black. The acknowledgment now cites the named AI systems and states their role and the authors' responsibility. Keep revision mode on for the next author review and disable it for submission. |
| P0.4 missing operator ablations | **Claim narrowed** | The manuscript now attributes results to the complete operator and explicitly does not attribute an isolated gain to full reset or persistent accumulation. A reproducible P15 campaign was added for a later empirical test. |
| P1.5 dense-baseline context | **Completed** | The ResNet result is framed as an operating-point comparison, the optimizer difference is explicit, and the development gain audit is documented. |
| P1.6 structural novelty boundary | **Completed** | Related work now distinguishes trigger granularity, retained state, reset semantics, update magnitude, and synchronization, and explains the coverage supplied by EF-TopK and Sign-EF. |
| P1.7 accounting sensitivity | **Completed** | Logical-broadcast and conservative-unicast Event-to-dense ratios are reported for all four benchmarks from the recorded streams without retraining. |

All newly added prose is marked with the dark-orange `\actionrev{}` macro in
the review build. The empirical claim checker now validates the accounting
sensitivity directly against P14 and the ten ResNet run files.

### P2 — final polish

8. Rename the theorem to “Conditional finite-horizon gradient-norm bound.”
9. Inspect Figure 1 at final printed size and remove microcopy if necessary.
10. Remove revision colors and run the full anonymity, PDF, font, reference,
    and page-limit checks from a clean checkout.

## Bottom line

The paper has moved from the borderline/weak-reject state of Review 01 to a
credible weak accept. The empirical package is now broad enough for IJCNN, and
the theory is appropriately scoped. The most important immediate repair is not
a new large experiment. It is completing the displayed theory--experiment
connection in Table II. The remaining scientific decision is whether to run
the two small novelty ablations or deliberately narrow the claim to what the
current evidence already establishes.
