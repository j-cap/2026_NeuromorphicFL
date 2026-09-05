# P7 simulated peer review

## Overall verdict

The paper has a coherent conference contribution and no identified mathematical
correctness failure. The method is completely specified, the closest prior
operator is treated honestly, the evidence is reproducible, and bidirectional
communication accounting is unusually careful.

The present draft is nevertheless **not yet ready to freeze**. One comparator-
fairness issue threatens the strongest empirical wording, and several important
clarifications would materially improve the theory and IJCNN case. P7 is
therefore complete as a review exercise, but its gate is **revisions required**.
P8 should be narrow and directly tied to the findings below. The review does
not justify reopening broad mechanism discovery or running T5.

Severity means:

- **Blocking:** threatens the central claim or likely produces a major-review
  objection if left unresolved;
- **Important:** should be corrected for a strong submission but does not
  invalidate the central result;
- **Optional:** improves presentation or breadth without being necessary for
  correctness.

## Review 1: federated-learning method and evaluation

### FL-B1 -- Independent update-scale fairness

**Severity: blocking for the current dense-FedAvg headline.**

Event-FedAvg selects a trigger threshold and an independent server-model
quantum. Dense FedAvg directly applies the averaged local delta. The earlier Q3
audit fairly selected the shared local learning rate from
`{0.01, 0.02, 0.05, 0.1}`, and the final campaigns use method-matched local
steps and minibatch streams. However, dense FedAvg and the residual baselines
were not given a comparable server update-scale degree of freedom. The report
already acknowledges that Event-FedAvg's predictive advantage should not be
interpreted as compression intrinsically improving generalization.

This matters because the abstract currently emphasizes that CIFAR-10
Event-FedAvg improves mean accuracy by 5.63 points over dense FedAvg. A reviewer
can reasonably attribute part of that gap to global step-size selection rather
than event communication. The matched-traffic comparison with Strom remains
useful, but it does not by itself protect the dense comparison.

**Required P8 resolution:** use the development partition to tune a small,
predeclared server-gain grid for dense FedAvg, then evaluate only the selected
point on the existing three held-out seeds. Apply the same principle to any
baseline whose server-side scaling is used in a headline comparison. If this
run is not performed, remove the accuracy-improvement-over-dense statement from
the abstract and treat dense FedAvg only as a communication reference.

### FL-I1 -- “Traffic matched” is stronger than the protocol

**Severity: important.**

The protocol selects the baseline point nearest to Event-FedAvg in log traffic
on the development partition. Held-out traffic is not constrained to be equal:
for CIFAR-10, the nearest Strom point uses 10.1% more traffic and the nearest
EF-TopK point uses 32.9% less. “Traffic-matched” can therefore be read as an
exact-budget comparison that was not performed.

**Required P8 resolution:** use “development-selected nearest-traffic control”
throughout the paper, captions, and generated table. Reserve “matched” for the
shared training protocol and communication-accounting rules.

### FL-I2 -- Worst-class evidence is too easy to miss

**Severity: important.**

Worst-class accuracy is a declared metric and is particularly relevant under
strong label skew. The text honestly reports the CIFAR-10 qualification, but
the headline table omits the metric. Consequently, a reader scanning the
visual argument sees only mean accuracy and may miss that dense FedAvg and
quality-selected EF-TopK have slightly higher CIFAR-10 worst-class means.

**Required P8 resolution:** add a compact worst-class column to the main table,
or add an equally visible table note containing the selected rows. Preserve the
three-seed uncertainty and do not claim uniform class-wise superiority.

### FL-I3 -- Scope of the held-out partitions

**Severity: important.**

The held-out seeds resample examples and training randomness under one fixed
strong-label-skew construction. They do not test multiple heterogeneity
families such as different Dirichlet concentrations. “Independent held-out
partitions” is technically defensible, but can imply broader heterogeneity
coverage than the campaign provides.

**Required P8 resolution:** state that the three seeds are independent
realizations of the fixed strong-skew template. Additional heterogeneity is a
journal-extension task unless the revised claims expand beyond this template.

### FL-I4 -- Three-seed uncertainty limits ranking claims

**Severity: important but already substantially mitigated.**

The draft reports mean plus sample standard deviation and avoids significance
claims. This is appropriate. However, the CIFAR-10 mean difference between
Event-FedAvg and quality-selected Strom is only 0.38 percentage points, smaller
than the reported dispersion. “Best mean among the tested settings” is valid;
“better method” or statistical superiority is not.

**Required P8 resolution:** retain operating-point and mean language. Do not
add a significance claim. Extra seeds are only necessary if P8 elects to make
a stronger ranking claim.

### FL-O1 -- Communication-model sensitivity

**Severity: optional.**

Conservative unicast is clearly defined and applied symmetrically. A real
system may provide multicast or charge requests differently. The frozen CSVs
also contain uplink and logical-broadcast totals, which makes a sensitivity
statement possible without new experiments.

**Optional P8 improvement:** state in one sentence whether the qualitative
frontier conclusion survives the stored broadcast accounting. Do not claim
network latency or energy.

## Review 2: theory and algorithm correspondence

### TH-C1 -- Correctness and implementation correspondence

**Verdict: pass; no blocking defect found.**

The weighting and sign conventions are consistent: the client weight enters
the evidence state once, local SGD deltas have the descent sign, and pulses are
not weighted a second time at the server. Round indexing prevents same-round
causal leakage. The implementation performs one threshold test and a true full
reset. The event-energy inequality
`||sum_i c_i||^2 <= M sum_i ||c_i||^2 = M N_r` and the pathwise event budget
are valid. The smoothness inequality has the correct sign, and telescoping under
the stated conditional alignment assumption yields the displayed finite-
horizon bound. The manuscript also correctly avoids applying its asymptotic
interpretation to the empirical quantum exponents.

### TH-I1 -- The empirical alignment ratio is undefined in the paper

**Severity: important.**

The theory section reports a `$q_r$-weighted trajectory alignment ratio' of
35.42, but never defines the statistic. A reader cannot determine whether it is
a minimum, mean of per-round ratios, or ratio of weighted sums, nor connect it
to the constant in the conditional assumption. The report denotes the value by
`hat{kappa}_R`, but that definition did not survive manuscript extraction.

**Required P8 resolution:** give the one-line formula immediately before its
value, including the rounds and weighting convention. Explicitly call it a
finite-trajectory statistic, not an estimate proving a uniform conditional
constant.

### TH-I2 -- The conditional theorem is an interface, not a mechanism proof

**Severity: important but correctly bounded.**

The alignment assumption contains the central optimization difficulty:
silence, stale state, local drift, sign errors, and heterogeneity are absorbed
into the defect. The theorem is therefore useful as an exact interface and
diagnostic decomposition, but does not establish convergence of the coupled
method from primitive assumptions.

**Required P8 resolution:** preserve the current limitation language and avoid
promoting the result to a convergence theorem for Event-FedAvg. T5 is not
required for the conference paper because the manuscript already states this
boundary accurately.

### TH-O1 -- Proof-sketch density

**Severity: optional.**

The pathwise budget proof and stationarity proof are concise but sufficient for
a conference paper. If P8 frees space, a sentence making the disjoint reset-to-
event blocks explicit would improve readability; no additional theorem is
needed.

## Review 3: IJCNN and neuromorphic relevance

### NN-I1 -- The distinctive ingredients are not isolated empirically

**Severity: important and close to blocking for a skeptical neuromorphic
reviewer.**

The LIF analogy is substantive at the operator level: persistent leaky state,
first passage, a signed event, and reset all affect communication. The paper
also avoids claiming that the trained model is spiking or that hardware energy
was measured. However, the final setting uses `rho=0.999`, and the closest
Strom comparison changes leakage, reset semantics, and model-quantum coupling
simultaneously. The results therefore demonstrate the value of the complete
operating point, not why leakage, full reset, or independent resolution matters
individually.

**Required P8 resolution:** run a compact, development-first, one-factor audit
of the frozen operator. At minimum compare (i) frozen Event-FedAvg, (ii)
`rho=1` with full reset and independent quantum, and (iii) a reset or quantum-
coupling sibling. Promote only predeclared informative variants to the existing
held-out seeds. Report the result compactly; a negative or negligible leakage
effect is acceptable but must narrow the neuromorphic interpretation.

### NN-I2 -- Novelty wording must reflect the qualified search result

**Severity: important.**

The P2 audit correctly concludes only that a targeted, dated search identified
no exact predecessor. The manuscript instead says that cited method families
“do not combine” the frozen ingredients, which reads closer to a universal
absence claim.

**Required P8 resolution:** write that the reviewed methods did not reveal the
complete operator and retain Strom and LENA as close predecessors. Continue to
claim the complete construction, not priority for temporal accumulation,
thresholding, sign events, full clearing, or event-triggered FL.

### NN-O1 -- Conventional networks are acceptable but should remain explicit

**Severity: optional; current handling is good.**

IJCNN relevance does not require the predictive model itself to be spiking.
The paper clearly states that LIF dynamics form a communication layer around
conventional MLP/CNN training. Keep this distinction and do not introduce
neuromorphic-hardware language without measurements.

## Presentation review

### PR-O1 -- Float order

**Severity: optional.**

The complete draft compiles cleanly in five pages, and all visual elements are
introduced before use. Because both principal results elements are double-
column floats, the rendered conclusion appears before the figure and table on
the following page. This is legal but weakens the reading sequence.

**Optional P8 improvement:** use a float barrier or modest source reordering so
the conclusion follows the visual evidence. Recheck that this does not create a
sixth-page overflow or damage the reference layout.

## Required P8 task list

1. **Close FL-B1:** tune and evaluate a development-selected dense server gain,
   or remove accuracy-superiority-over-dense from the headline.
2. **Close NN-I1:** perform one compact operator-factor audit, development
   first and held-out only for frozen informative variants.
3. **Revise terminology and theory exposition:** close FL-I1, FL-I3, TH-I1,
   and NN-I2 with manuscript-only edits.
4. **Expose the class-wise qualification:** close FL-I2 in the main visual
   argument.
5. **Preserve uncertainty boundaries:** retain the bounded wording required by
   FL-I4 and TH-I2.

Optional communication sensitivity and float-order cleanup may be included if
they do not displace central content.

## P7 gate decision

The three reviews are complete, but the gate is **not yet passed** because
FL-B1 remains blocking. The current draft is best characterized as a promising
borderline conference submission that should become materially stronger after
a narrow P8. No finding requires T5, asynchronous experiments, hardware tests,
larger models, or a new benchmark.
