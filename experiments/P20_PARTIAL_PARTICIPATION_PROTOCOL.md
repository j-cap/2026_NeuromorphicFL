# P20 — Larger-client and partial-participation pilot

## Question

Does Event-FedAvg retain its end-to-end communication advantage when the
registered client population grows and only a subset participates in each
round?

The primary comparison holds the number of active clients and the local
computation per round fixed.  It changes the population from ten clients with
full participation to fifty clients with ten selected uniformly without
replacement per round.  A fifty-of-fifty control separates the effect of
splitting the data over more clients from the effect of client sampling.

## Frozen one-seed pilot

| Scenario | Registered clients | Active per round | Purpose |
|---|---:|---:|---|
| `m10_k10` | 10 | 10 | Existing matched P19 reference |
| `m50_k50` | 50 | 50 | Client-count scaling control |
| `m50_k10` | 50 | 10 | Primary 20% participation test |
| `m50_k05` | 50 | 5 | Optional 10% stress test |

The default pilot runs Event-FedAvg, quality-selected EF-TopK, and dense FedAvg
for the first three scenarios.  The three `m10_k10` results are imported from
the frozen P19 seed-zero campaign after checking the seed and horizon.  Thus,
the default command computes six new runs.  The stress test is deliberately a
separate decision.

All cells use the compact CIFAR-10 CNN, partition seed 3500, training seed
83500, 1,800 rounds, five local steps, batch size 32, and the frozen P19 method
parameters.  No method is retuned for partial participation.

## Fifty-client partition

The 50,000 training samples are divided equally, giving 1,000 examples per
client.  Five clients are assigned to each dominant class.  Every client has
550 examples of its dominant class and 50 examples from each other class.
Consequently, every client has the same weight and every global class still has
exactly 5,000 examples.

## Sampling and state semantics

For round `r`, the server selects a fixed-size set `S_r` uniformly without
replacement.  One deterministic selection schedule is shared by all methods.
Only selected clients perform local SGD.

The client contribution is corrected before compression using

`p_i / pi_i`,

where `p_i` is the data weight and `pi_i` is the inclusion probability.  With
equal clients and `K` selected from `M`, this reduces to `1 / K`.

An Event-FedAvg client that returns after a gap of `g` global rounds first ages
its stored encoder state by `rho ** g`, then adds the new weighted evidence and
applies the unchanged threshold and reset operator.  EF-TopK residuals persist
while a client is inactive.

## Exact synchronization and traffic

The initial dense model is charged once for every registered client.  Before a
selected client computes, it catches up on updates missed since its previous
synchronization.  The server sends either the complete ordered packet replay or
a dense model checkpoint, whichever uses fewer packetized bits.  After the
server update, every selected client receives that round's exact update under
the same replay-versus-checkpoint rule.  Inactive clients receive nothing during
that round.

The operational traffic total includes selected-client requests, uplinks,
packet headers, replay/checkpoint data, and the initial model.  A second,
stricter total also charges a terminal catch-up for every stale client so all
registered clients exactly match the final server model.  This stricter value
is reported as `unicast_hybrid_total_bits` and should be used for the primary
comparison.

## Run on the workstation

From an Anaconda Prompt in the repository root:

```bat
conda activate neuromorphicfl-resnet14
git pull
python experiments\p20_cifar10_partial_participation.py smoke
python experiments\p20_cifar10_partial_participation.py point --point-id m50_k10_event
python experiments\p20_cifar10_partial_participation.py pilot --workers 4
python experiments\p20_cifar10_partial_participation.py status
python experiments\p20_cifar10_partial_participation.py verify
python experiments\p20_cifar10_partial_participation.py analyze
```

The point command is a useful timing check.  The campaign is resumable and
skips complete outputs, so the subsequent pilot command does not repeat that
run.  If memory pressure or workstation responsiveness is a concern, use two
workers.  Do not run the stress cell until the pilot has been inspected.

Optional stress test:

```bat
python experiments\p20_cifar10_partial_participation.py stress --workers 3
python experiments\p20_cifar10_partial_participation.py verify --include-stress
python experiments\p20_cifar10_partial_participation.py analyze --include-stress
```

## Matched ten-seed extension

After the pilot gate passes, extend only the primary `m50_k10` scenario.  The
campaign runs Event-FedAvg, quality-selected EF-TopK, and dense FedAvg with ten
matched partition, training, and client-selection seeds.  Seed zero reuses the
completed pilot, leaving 27 new runs.  The campaign is resumable.

```bat
python experiments\p20_cifar10_partial_participation.py ten-seed --workers 4
python experiments\p20_cifar10_partial_participation.py status
python experiments\p20_cifar10_partial_participation.py verify-ten-seed
python experiments\p20_cifar10_partial_participation.py finalize-ten-seed
```

Use `--start-seed` and `--end-seed` to split the work into batches.  Complete
point/seed pairs are skipped unless `--force` is supplied.  A single failed
pair can be resumed explicitly with `point --point-id ... --seed-index ...`.

## Saved evidence

The output directory is
`experiments/results/p20_cifar10_partial_participation/`.  It contains one
summary and learning history per run, per-client activity diagnostics for
Event-FedAvg, the exact client-selection schedule for each scenario, a machine
readable protocol, a cross-method comparison, and a pilot-gate report.

The Event-FedAvg activity files record participation gaps, events on return,
catch-up mode and cost, current-round synchronization mode and cost, and uplink
events.  These fields distinguish a genuine sparse-participation benefit from a
result that merely shifts most traffic into dense catch-up checkpoints.

## Pilot decision rule

The `m50_k10` Event-FedAvg cell is a candidate for a matched ten-seed extension
only if it is stable, finishes within one test-accuracy point of its dense cell,
uses less exact-final-sync traffic than dense, and fewer than 90% of nonempty
synchronizations use dense checkpoints.  This screen is a planning rule rather
than a statistical conclusion.  The learning curves and return-burst
diagnostics must also be inspected before expanding the campaign.

