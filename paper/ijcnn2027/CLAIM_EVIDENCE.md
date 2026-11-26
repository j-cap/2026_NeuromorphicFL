# Current manuscript claim–evidence map — 2026-11-26

This map supersedes the historical P12/P13/P19 headline selection. Those outputs
remain archived and checked by their original builders.

| Claim | Evidence | Scope |
|---|---|---|
| Complete signed-event operator and exact synchronization | Main equations/algorithm; frozen event codec and runner sources; panel receipts; `check_theory_contract.py` | Synchronous server-based FL; separate scoped partial-participation extension |
| Bounded post-reset state and pathwise event count | Main proposition; randomized algebra checks | No assumption that weak evidence eventually fires |
| Serialized event-upload upper bound | Explicit-index candidate in EVS1; 128-bit header and at most seven padding bits per nonempty client packet | `135 P + (ceil(log2 d)+1) N`; requests/synchronization excluded from this component bound |
| Conditional finite-horizon gradient result | Main theorem and exact alignment identity; P11 diagnostics and theory-to-trajectory note | Alignment is assumed, not proved by sampled trajectories; no final-iterate or asymptotic guarantee |
| Event has the least final unicast traffic at both fixed panels | `calibrated_core_endpoints.csv`; source `main_runs.csv` / `confirmation_runs.csv` | Five selected operating points, ten matched seeds per model |
| Lower cost to all declared accuracy targets | `calibrated_core_targets.csv`; all 24 paired contrasts; source histories | First sampled per-run crossing; unadjusted paired log-ratio intervals; one Dense 55% miss stays censored |
| 50% STC savings: compact 57.8%, ResNet 23.4% | Generated target table and target effects CSV | All ten pairs reach; Event uses fewer bits in all pairs |
| 50% EF-TopK savings: compact 95.8%, ResNet 65.4% | Same sources | Cost to a predictive target, not a universal convergence-rate claim |
| Best compact-CNN mean final accuracy | Generated endpoint table | Ström paired accuracy interval spans zero; EF-TopK has lowest mean CE |
| ResNet Event/Ström quality and cost advantage | Endpoint effects: +3.308 pp, 30.735% saving | Adapted residual encoder in common FedAvg host |
| ResNet final STC/EF quality tradeoffs | Endpoint effects: Event–STC -0.724 pp [-1.611,0.163]; Event–EF -0.943 [-1.575,-0.311] | Neither equivalence to STC nor predictive dominance of EF is established |
| Every family uses architecture-specific calibration | Frozen development selections/protocols | Common accumulated ResNet budget, different historical compact-CNN rules; bounded searches only |
| Competitive sparse STC points remain possible | ResNet development study, p=.05/gain4 and p=.01/gain4 | Development only; not substituted into confirmation and not a method-wide frontier claim |
| Persistence helps at an earlier audited operating point | P15 ten-seed 120-round operator audit | Full reset's accuracy advantage unresolved; no isolated leak benefit claimed |
| Earlier partial participation retains savings | Frozen P20 50-client ten-seed runs and supporting figure | Original replay/checkpoint catch-up costs; no newly calibrated STC/EF ranking transfer |

`build_calibrated_core.py --check` validates the frozen input hashes, reconstructs
100 endpoints and first crossings, and matches the original summary/target CSVs.
`check_manuscript_claims.py` checks headline numbers, target coverage and necessary
qualifications against those regenerated tables.
