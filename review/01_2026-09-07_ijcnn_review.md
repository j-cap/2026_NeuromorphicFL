# IJCNN 2027 draft review 01

## Review metadata

- **Review date:** 2026-09-07
- **Reviewed commit:** `71bdc95e2922842f151243c9a1d4cb0c72bc830a`
- **Manuscript:** `paper/ijcnn2027/main.tex`
- **Review mode:** simulated anonymous conference review plus submission audit
- **Venue basis:** [IJCNN 2027 Call for Papers](https://ijcnn.org/2027/authors/call-for-papers) and [IJCNN 2027 Reviewer Guidelines](https://ijcnn.org/2027/reviewer-guidelines)

The official call requires at most six pages in IEEE two-column conference
format. The reviewer guidance asks reviewers to judge relevance, technical
quality, novelty, presentation and replication detail, and the overall strength
of the contribution and evaluation. It also states that submissions are
double-blind and receive three independent reviews.

## Overall assessment

**Simulated recommendation: borderline, leaning weak reject in its current
form.** The paper is unusually careful about protocol completeness, evidence
boundaries, and negative qualifications. Event-FedAvg is well within IJCNN's
federated, communication-efficient, and event-driven scope. The method is easy
to understand, the update semantics are explicit, and the empirical operating
points are promising.

The main acceptance risk is not correctness of the reported numbers. It is
whether the contribution is sufficiently novel and broadly validated for a
full IJCNN paper. The closest-predecessor boundary is narrow, the isolated leak
has no observed effect, full reset is not directly ablated, and the principal
experiments use small conventional networks, ten clients, synchronous full
participation, and two image datasets. A reviewer can reasonably conclude that
the paper establishes a useful finite-horizon operating point but not yet a
general communication method.

There is also one objective repository blocker: the advertised reproduction
entry point currently fails because the checksum manifest is stale.

## Simulated scorecard

| IJCNN criterion | Assessment | Rationale |
|---|---|---|
| Relevance | **Strong** | Directly fits federated/distributed learning, communication-efficient learning, and event-driven processing. |
| Technical quality | **Good, with a central qualification** | The operator, accounting, identities, and conditional descent bound are coherent. The optimization result depends on an unproved alignment condition. |
| Novelty | **Borderline to good** | The complete combination is plausibly new, but its ingredients are established and the experiments do not isolate all distinguishing choices. |
| Presentation | **Good** | Clear structure, strong limitations section, and much improved terminology. Figure 1 remains dense at final scale. |
| Replication detail | **Borderline** | The repository is unusually well documented, but essential Event-FedAvg values are absent from the paper and the frozen checksum manifest currently fails. |
| Experimental validation | **Borderline** | Paired seeds and honest qualifications are strengths; task, model, client, and systems breadth remain limited. |
| Overall | **Borderline / weak reject** | Promising paper with a credible path to acceptance after the priority issues below are closed. |

## Strengths

1. **The communication protocol is complete.** The paper accounts for both
   uplink and downlink, defines exact synchronization, includes metadata and
   address costs, and avoids the common mistake of presenting an uplink-only
   compressor as a complete FL protocol.
2. **The method is specified precisely.** The distinction among retained
   evidence, threshold, full reset, signed event, aggregate pulse, and model
   quantum is clear. Algorithm 1 provides a useful executable summary.
3. **The evidence discipline is strong.** Local learning is matched across
   methods, selection is development-only, headline estimates use paired
   seeds, and the added seven-seed analysis is disclosed rather than blended
   into an apparently prospective experiment.
4. **The manuscript reports adverse evidence.** It states that leakage at
   `rho = 0.999` has no measurable isolated effect, that the worst-class
   advantage on CIFAR-10 is unresolved, and that positive first-order alignment
   need not produce an objective decrease.
5. **The theory is scoped honestly.** The pathwise encoder claims are separated
   from the assumption required to connect pulses to optimization. The paper
   explicitly rejects unconditional and asymptotic convergence claims.
6. **The current PDF is formally close to submission-ready.** It compiles to
   exactly six US Letter pages, has no unresolved references or overfull boxes,
   and passes the repository's anonymity, embedded-font, PDF-version, and
   attachment checks.

## Major issues

### M1. The frozen reproduction contract currently fails

**Severity: submission blocker, easy to repair.**

Running `python paper/ijcnn2027/reproduce.py` validates the generated evidence,
26 manuscript claims, 40 alignment runs with 1,240 snapshots, the theory
contract, and the workflow policy. It then fails the checksum-manifest check.
Five tracked files have hashes that differ from the committed manifest:

- `paper/ijcnn2027/generated/main_results_table.tex`
- `paper/ijcnn2027/figures/event_fedavg_method.pdf`
- `paper/ijcnn2027/figures/communication_frontier.pdf`
- `paper/ijcnn2027/main.tex`
- `paper/ijcnn2027/build_visuals.py`

This appears to be freeze-record drift after deliberate recent edits, not a
numerical inconsistency. Nevertheless, the repository currently promises a
one-command validation that does not pass. Review and intentionally update the
manifest, then rerun the full reproduction and compile checks from a clean
checkout.

### M2. Essential experimental settings are missing from the paper

**Severity: major under the replication-detail criterion.**

The manuscript gives local learning rates, round counts, local steps, `r_0`,
and the quantum exponents, but it does not state the selected Event-FedAvg
threshold, leak, and initial quantum for every benchmark. Those values are only
recoverable from repository documentation. The frozen settings are currently:

| Benchmark | `rho` | threshold | initial quantum `q_0` | exponent `alpha` |
|---|---:|---:|---:|---:|
| Fashion-MNIST MLP | 0.999 | 0.025 | 0.005 | 0.1 |
| Fashion-MNIST compact CNN | 0.999 | 0.025 | 0.010 | 0.3 |
| CIFAR-10 compact CNN | 0.999 | 0.025 | 0.005 | 0.2 |

Add a compact configuration table or sentence containing these values and the
selected comparator configurations. A paper should permit reconstruction of
its central settings without requiring access to a repository.

### M3. The empirical evidence is narrow relative to the general claim

**Severity: major acceptance risk.**

The tested models contain roughly 14.5k--25.8k parameters, the federation has
ten clients, all clients participate synchronously, and the datasets are
Fashion-MNIST and CIFAR-10. This is sufficient for a mechanism study but weak
evidence for a broadly useful high-dimensional FL communication layer. The
paper itself acknowledges this limitation, but acknowledgment does not replace
validation.

At least one stronger stress test would materially improve the paper: a modern
residual network on CIFAR-100 or another larger task, more clients, or partial
participation. The most informative design would vary client count and
participation because persistent client-side state is central to the method.
If compute is constrained, a focused scaling study is more valuable than
another small architecture on Fashion-MNIST.

There is also a mismatch between the two central result displays. Figure 2
shows all five method families but uses three seeds; Table I uses ten seeds but
only reports Event-FedAvg and selected Strom/EF-TopK points. The dense FedAvg
number emphasized in the abstract is absent from Table I, and Sign-EF is absent
from the ten-seed table. Either produce the full figure from ten-seed results or
make the three-seed status less central. At minimum, add the tuned dense
CIFAR-10 row to Table I so the abstract's main comparison is directly visible.

### M4. The experiments do not fully isolate the claimed novelty

**Severity: major novelty risk.**

The paper distinguishes Event-FedAvg from Strom by leakage, full reset, and an
independent model quantum. The current mechanism audit shows that removing the
small leak has no measurable effect. It tests one coupled
`q_r = threshold = 0.025` sibling, but it does not isolate full reset from
residual subtraction, nor persistent integration from a memoryless
round-by-round trigger. Consequently, a skeptical reviewer can attribute the
result mainly to a tuned server step size rather than to the proposed
event-encoder design.

Add a compact, matched ablation containing:

1. full reset versus subtractive/residual-conserving reset;
2. persistent state versus a memoryless threshold rule; and
3. independent threshold/quantum versus a fairly tuned coupled sibling.

Report both accuracy and total traffic. If the ablations are negative, narrow
the contribution to the ingredients that are actually supported. The current
honest result on leakage is an asset and should remain.

### M5. The theoretical result is correct in scope but easy to overread

**Severity: important framing issue.**

Propositions 1 and 2 are useful deterministic characterizations. Proposition 3
is a conditional weighted gradient-norm bound whose key alignment assumption,
Eq. (13), is not derived from primitive properties of the algorithm. Since
`beta_r` can absorb silence, stale evidence, sign errors, drift, and
heterogeneity, Eq. (14) does not by itself establish convergence of
Event-FedAvg. The manuscript says this later, but the abstract's phrase
"conditional finite-horizon stationarity result" may still sound stronger
than what is proved.

Prefer "a conditional finite-horizon gradient-norm bound" in the abstract and
contribution summary. State immediately before or after Eq. (13) that this is
the unresolved learning condition, not an encoder property. The empirical
alignment ratio is a diagnostic analogue and should not be presented as
validation of Eq. (13). The paper currently respects that boundary in the
results section; preserve it.

The description of `widehat{kappa_A}` as "dimensionless" should also be
checked. Under ordinary physical/unit bookkeeping, its units depend on those
assigned to model parameters and objective gradients. Calling it a normalized
trajectory ratio is safer and sufficient.

### M6. Communication conclusions depend on one accounting design

**Severity: important.**

Conservative bidirectional unicast is a defensible primary metric, and the
paper explains it well. However, the headline savings depend on a 64-bit packet
header, separate unicast downlinks, and replay of the ordered client packet
stream. Other deployments may broadcast once or transmit a sparse aggregate
of `C^r` instead of replaying every client packet. The current result is
therefore a comparison under one explicit protocol, not a protocol-independent
compression ratio.

Add a small sensitivity analysis for at least logical broadcast versus unicast
and, if exact arithmetic permits it, aggregated sparse downlink versus ordered
replay. A short table of traffic ratios is enough; no retraining is required.
Explain whether the method ordering changes. Keep the conservative unicast
metric as the primary measure.

## Presentation and layout review

### Figure 1

The figure now has a clear five-stage flow, correct notation, and a serif style
that is visually close to the manuscript. At its actual `0.94 textwidth` scale,
however, the smallest 17--18 point labels in the 1,255-point-wide source become
approximately 6.5--7 point text. They are readable when zoomed but borderline
for a printed proceedings page. The source uses Liberation Serif while the
compiled IEEE body uses a Times-compatible Nimbus Roman face, so the match is
close rather than exact.

Do not make the figure taller again. Instead, remove secondary microcopy,
shorten the downlink box, and raise the minimum final text size toward 8 points.
The equations and the five stage titles should receive priority. The figure is
large but no longer disproportionately dominant; density, not total width, is
the remaining issue.

### Figure 2

The restored five-method view is much more informative than a selected-method
subset. The axes, symbols, and error bars are legible. The main concern is
evidential rather than graphical: readers must notice that its error bars use
three seeds while Table I uses ten. A ten-seed version would remove that
cognitive mismatch.

### Tables

The bold best values in Table I now stand out clearly, and the caption explains
that lower traffic is better. Table II is technically useful but extremely
dense and uses script-size text. Consider moving one of its two subtables to a
compact sentence or reporting only the components needed to support the stated
heterogeneity/depth conclusion.

### PDF and LaTeX

- The compiled manuscript is exactly six pages; any additions must replace or
  compress existing content.
- The output PDF passes the current repository compliance checker.
- There are no overfull boxes or unresolved citations/references. The log has
  underfull-box warnings, which are cosmetic.
- The Figure 1 input is PDF 1.7 while pdfTeX declares a maximum included version
  of 1.5. The output remains PDF 1.5 and renders correctly, but exporting the
  figure as PDF 1.5 would remove the warning.
- The detailed 2027 author instructions are not yet represented by a stable
  venue-specific checklist in the repository. Recheck anonymity, page size,
  AI-disclosure language, and any PDF eXpress requirement when IJCNN publishes
  the final instructions.

## Section-level comments

### Title and abstract

The title is accurate and avoids claiming that the trained network is spiking.
The abstract is evidence-rich but crowded. Retain the main CIFAR-10 operating
point and the limitation sentence; reduce the audit inventory and temper
"stationarity result" as noted above.

### Introduction and related work

The italicized design question is effective. The definitions of coordinate,
client encoder, model quantum, ordered replay, dense-checkpoint fallback, LIF,
and SNN now address the earlier clarity concerns.

The related-work section is too short for a novelty claim resting on a precise
combination of known components. STC is discussed but not evaluated even though
it is a natural bidirectional FL-compression comparator. Explain why it is not
included or add it. Expand the comparison with the closest event-triggered FL
method at the level of trigger granularity, retained state, reset rule,
downlink, and convergence assumptions. The current sentence that a targeted
search found no exact predecessor is appropriately qualified and should remain
qualified.

### Method

This is the strongest section. The role of `p_i`, division by `eta_r`, complete
reset, coincident pulses, address length, and packet header is explicit.
Algorithm 1 is prominent and useful. Add the frozen numerical configurations
near the method or experimental protocol.

### Theory

The Cauchy--Schwarz step and the derivation route to Eqs. (12) and (14) are now
specified. Reusing `A_r` is explicitly declared. `F_inf` and the purpose of the
random index `J_R` are explained. The remaining issue is not local clarity but
the strength of the conditional assumption and how prominently that limitation
is signaled.

### Experiments and results

The protocol explains matched local learning, unicast accounting, marker
semantics, strong non-IID data, and the data underlying both tables. Sources
are attached to compared algorithm families in the text and Table I. The
results section is substantially clearer than the earlier draft.

The main remaining work is to make the validation commensurate with the
generality of the contribution, reconcile three-seed and ten-seed displays,
show the dense headline comparison in the table, and isolate the operator's
distinctive components.

### Limitations and conclusion

The limitations section is strong and should not be weakened. The conclusion
is sharper than the earlier version, but it repeats several qualifications and
experimental details. Use some of that space for a crisp final claim:
Event-FedAvg establishes an exact, stateful sparse communication operating
point under the evaluated synchronous protocol; generalization to larger and
partially participating federations remains open.

## Prioritized revision plan

### P0 - close before the next review

1. Repair the reproducibility manifest and make the documented clean-checkout
   validation pass.
2. Put every frozen Event-FedAvg and selected comparator configuration in the
   manuscript.
3. Add the tuned dense CIFAR-10 row to Table I and resolve the three-seed versus
   ten-seed Figure 2 mismatch.
4. Add direct full-reset, persistent-state, and fairly tuned quantum-coupling
   ablations, or narrow the novelty claim to match the evidence.

### P1 - highest expected effect on acceptance

5. Add one meaningful scale/participation stress test.
6. Reframe Proposition 3 as a conditional gradient-norm bound everywhere.
7. Add a communication-accounting sensitivity table.
8. Strengthen the closest-work comparison and address the missing STC
   experiment.

### P2 - final presentation pass

9. Simplify Figure 1 until all final-size labels are approximately 8 points or
   larger without increasing its footprint.
10. Reduce Table II's density, export Figure 1 as PDF 1.5, and clean the
    remaining underfull-box warnings if this does not consume substantive
    space.
11. Recheck the final IJCNN 2027 author instructions immediately before
    submission.

## Gate for review 02

Review 02 should be performed on a new immutable commit after all P0 items are
resolved. Its first checks should be the clean-checkout reproduction command,
the six-page compiled PDF, the revised novelty ablations, and consistency of
all seed counts across the abstract, Figure 2, Table I, and results text.
