# Calibrated participation pilot: analysis and confirmation decision

Results commit: 563f4a534e5ae950bc0fd9596b1b39bd374fffd7.
Driver/source commit: 95ee681043ffe1087b1f7f917464b2c64bf7f4cc.
Evidence directory: experiments/results/calibrated_participation_v1.

## Decision

**Proceed unchanged to the planned ten-seed, 1,800-round confirmation.**
All fifteen pilot fits are complete and finite, source/configuration identities
match, and the communication/reconstruction checks pass. The pilot gate concerns
protocol correctness and complete evidence; no ranking or Event-win condition
was used. The existing confirmation command needs no code change.

**Event has the lowest final communication in every fraction, with important
quality and target-specific qualifications.** At 20% participation it has the
highest validation accuracy and lowest validation CE in this seed. At 10%, Ström,
STC and EF-TopK finish at higher accuracy; the strongest comparison is Ström:
2.10 percentage points higher accuracy and 15.4 points higher worst-class
accuracy. Event uses 8.69% less final traffic than Ström.

The pilot is one development seed at 300 rounds, evaluated on held-out training
validation data. It does not establish test performance, ten-seed robustness,
equivalence, universal target-cost superiority or faster convergence.
No partial-participation-specific tuning has been performed. These are the
full-participation-selected configurations transferred to a fifty-client host.

## Audited protocol and completeness

- Five methods x three active-client counts (50, 10, 5), all fifteen complete;
  zero numerical failures and zero missing fits.
- Compact CIFAR-10 CNN, 20,570 parameters, fifty equally weighted clients;
  fixed 45,000/5,000 development training/validation split. Each client has
  900 training examples: 495 from its dominant class and 45 from each other.
- Development partition seed 13000, SGD seed 113000, split seed 23000 and
  selection seed 213000; initialization seed 7777. The nested selection
  schedules and partitions are shared across methods.
- Five local SGD steps, batch 32, learning rate .05 and L2 coefficient .0005;
  every method keeps its frozen calibration from the completed compact-CNN core.
- The 1,800-round configurations are stopped at round 300. Each fit has
  300 raw round records and six saved evaluations (1, 60, 120, 180, 240, 300).
  There is no test-set parsing/evaluation in this pilot and no development
  checkpoint transfer into confirmation.

The independent audit validates 65 source hashes and all receipt/file
identities. It checks 4,500 raw round records and 90 sampled model fingerprints,
and reconstructs every online catch-up cost and saved all-client terminal cost
from the client schedules, packet lengths and synchronization cursors. Actual
frame padding, recipient copies, requests, initialization and cumulative
components balance exactly. All fifteen runners recorded successful exact
client reconstruction, including fifty-client checks at each saved endpoint.

The seven original numerical CSV products and completion/receipt audit are
reproduced in a temporary copy. Original campaign files remain unchanged.
The local progress-display field latest_checkpoint_round is excluded from this
comparison: ignored progress JSON is not committed, so a clean checkout shows
zero even for a complete receipt. This is a display artifact, not lost training.
No dataset images, model arrays or wire journals are loaded or retrained by
the independent audit; model reconstruction evidence is the source-pinned
runner assertions/flags and aligned fingerprints.

## Final validation quality and conservative communication

All values below are individual development fits at round 300.
Traffic includes final exact deployment to all fifty registered clients.
One Gbit is 10^9 bits. There are no confidence intervals or seed SDs for n=1.

| Participation | Method | Accuracy, % | Validation CE | Worst class, % | Total unicast, Gbit |
|---:|---|---:|---:|---:|---:|
| 100% | Event | 51.34 | 1.358730 | 27.0 | **0.248873** |
| 100% | Ström | 48.72 | 1.391125 | 27.4 | 0.299716 |
| 100% | STC | 51.70 | 1.350349 | 31.4 | 0.665390 |
| 100% | EF-TopK | 47.94 | 1.435579 | 27.8 | 7.930635 |
| 100% | Dense | 46.72 | 1.524281 | 17.2 | 19.782515 |
| 20% | Event | 48.94 | 1.424922 | 17.2 | **0.281779** |
| 20% | Ström | 48.40 | 1.426389 | 11.6 | 0.302639 |
| 20% | STC | 48.24 | 1.444359 | 12.8 | 0.411365 |
| 20% | EF-TopK | 46.56 | 1.456300 | 21.8 | 2.174304 |
| 20% | Dense | 47.56 | 1.499691 | 23.6 | 5.597656 |
| 10% | Event | 45.94 | 1.501216 | 9.6 | **0.290650** |
| 10% | Ström | 48.04 | 1.458875 | 25.0 | 0.318322 |
| 10% | STC | 46.66 | 1.478712 | 16.4 | 0.373732 |
| 10% | EF-TopK | 46.22 | 1.483442 | 22.8 | 1.086587 |
| 10% | Dense | 35.98 | 1.868165 | 3.2 | 2.938060 |

| Reference | Event final saving at 100%, % | At 20%, % | At 10%, % |
|---|---:|---:|---:|
| Ström | 16.96 | 6.89 | 8.69 |
| STC | 62.60 | 31.50 | 22.23 |
| EF-TopK | 96.86 | 87.04 | 73.25 |
| Dense | 98.74 | 94.97 | 90.11 |

These are fixed-budget cost differences, not matched-accuracy savings.
At 100%, STC has slightly higher accuracy and lower CE. At 20%, Event's
advantage in mean accuracy/CE does not extend to the largest worst-class
accuracy. At 10%, Event trails Ström by 2.10 accuracy points and 15.4 worst-class
points, and its CE is higher by .04234. The ten-seed run must show whether these
differences persist at the predeclared final horizon; longer training is not
assumed to remove them.

## Communication to the declared validation targets

The first saved crossing is used, without interpolation or a stability
requirement. Terminal costs mean hypothetical deployment IF stopped at that
crossing. These target costs are not computed from a mean curve.

| Participation | Method | First saved 45% round | Total cost to 45%, Mbit |
|---:|---|---:|---:|
| 100% | Event | 180 | 147.09 |
| 100% | Ström | 180 | 178.38 |
| 100% | STC | 120 | 285.52 |
| 100% | EF-TopK | 240 | 6,270.33 |
| 100% | Dense | 300 | 19,782.52 |
| 20% | Event | 240 | 225.03 |
| 20% | Ström | 240 | 241.14 |
| 20% | STC | 120 | 184.39 |
| 20% | EF-TopK | 240 | 1,734.61 |
| 20% | Dense | 300 | 5,597.66 |
| 10% | Event | 300 | 290.65 |
| 10% | Ström | 240 | 256.91 |
| 10% | STC | 300 | 373.73 |
| 10% | EF-TopK | 300 | 1,086.59 |
| 10% | Dense | Censored | Censored |

At 20%, Event uses **22.04% more traffic than STC to 45%**, despite its lower
round-300 total and higher final accuracy. At 10%, Event uses **13.13% more
than Ström to 45%** and crosses one saved interval later. Thus there is no
all-target communication advantage.

At 100%, only Event and STC reach 50% within this pilot: Event at round 300
with 248.87 Mbit; STC at round 240 with 538.10 Mbit. Event saves 53.75% despite
crossing later. No method reaches 50% at 20% or 10% participation, and no
method/fraction reaches 55%. Unreached contrasts remain censored and have
no imputed cost ratio.
The 1,800-round confirmation will retain all three predeclared targets.

## Why fewer selected clients do not imply less Event traffic

| Event participation | Initial distribution, Mbit | Current-round downlink, Mbit | Online catch-up, Mbit | Terminal catch-up incl. requests, Mbit | Operational total, Gbit |
|---:|---:|---:|---:|---:|---:|
| 100% | 32.915 | 204.264 | 0 | 0 | 0.248873 |
| 20% | 32.915 | 48.052 | 188.974 | 3.214 | 0.278564 |
| 10% | 32.915 | 25.191 | 217.229 | 7.737 | 0.282913 |

Online catch-up is 67.06% of Event's total conservative traffic at 20% and
74.74% at 10%. The terminal deployment surcharge is only 1.14% and 2.66%.
Consequently, the increased traffic relative to full participation is already
present in operational accounting; it is not created by the final audit sync.
Clients that return must obtain missed server updates or a checkpoint, so
reducing current recipients does not make their eventual synchronization free.

Event uses dense catch-up checkpoints in 1 of 2,413 non-current return syncs
at 20%, and 16 of 1,368 at 10%; the rest use replay. Dense checkpoints are
uncommon for Event in this pilot. The actual mix/costs, including all
comparators, are exported by the independent audit.

At the same 300 rounds, local SGD work is 75,000 steps at 100%, 15,000 at 20%
and 7,500 at 10%, per method. Therefore, across-fraction fixed-round quality
differences are not compute-matched. Methods share the same work within a
fraction. No additional compute-matched training is introduced at this stage.

## Runtime planning

The recorded final-attempt durations sum to 1.2023 elapsed worker-hours:
about 638--659 seconds per 100% job, 135--145 per 20% job, and 71--79 per 10%
job. These are parallel experiment durations, not deployment latency, CPU-time
measurements or a method-speed ranking. The receipts do not establish whether
an attempt resumed an earlier checkpoint.

The full matrix has ten times the seeds and six times the rounds. If all pilot
attempts were fresh and comparable throughput is sustained, the simple 60x
projection is about 72 summed elapsed worker-hours, or an idealized nine hours
at eight continuously busy workers. Longer test evaluation, load balance and
machine conditions add uncertainty; this is a planning estimate.

## Next local command

The existing driver already prepares the full matrix. Keep settings, seeds and
targets unchanged and start fresh confirmation fits; no retuning is needed to
pass this pilot gate:

    cd C:\Users\WeberJ\2026_NeuromorphicFL
    conda activate neuromorphicfl-resnet14
    git pull --ff-only
    python experiments/calibrated_participation.py confirm --workers auto
    python experiments/calibrated_participation.py status
    git add experiments/results/calibrated_participation_v1
    git diff --cached --check
    git commit -m "Add ten-seed calibrated participation confirmation"
    git push origin main

The command performs the analysis automatically. It verifies all pilot receipts,
writes its confirmation freeze before official test parsing, and resumes
unfinished confirmation jobs when repeated. It does not reuse pilot models.
The current result commit contains no confirmation fits.

Do not replace the manuscript's supporting participation result with this
single-seed validation pilot as a confirmed result. Integrate the completed
confirmation with both accuracy/CE and the target-specific exceptions visible.

## Read-only recheck

The independent audit is outside the frozen training-source inventory and
adds only analysis_pilot_independent_* products. It leaves every original
campaign file and training source untouched:

    python experiments/calibrated_participation_pilot_audit.py --check
