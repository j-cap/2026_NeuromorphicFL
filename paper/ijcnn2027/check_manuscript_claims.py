"""Check the current calibrated-panel narrative against regenerated evidence.

Historical evidence remains independently checked by its original builders;
it must not be substituted for the new 1800-round five-method headline.
"""
from pathlib import Path
import csv
import statistics

PAPER = Path(__file__).resolve().parent


def read(name):
    with (PAPER / 'evidence' / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    source = ' '.join((PAPER / 'main.tex').read_text(encoding='utf-8').split())
    endpoints = read('calibrated_core_endpoints.csv')
    targets = read('calibrated_core_targets.csv')
    effects = read('calibrated_core_endpoint_effects.csv')
    for panel in ('compact_cnn', 'resnet14'):
        event = next(x for x in endpoints if x['panel'] == panel and x['method'] == 'event')
        others = [x for x in endpoints if x['panel'] == panel and x['method'] != 'event']
        assert all(float(event['unicast_total_bits_mean']) < float(x['unicast_total_bits_mean']) for x in others)
        for row in targets:
            if row['panel'] != panel:
                continue
            assert float(row['ratio_ci95_high']) < 1
            assert int(row['event_reached']) == 10
            assert int(row['common_hits']) == (9 if panel == 'compact_cnn' and row['reference'] == 'dense' and float(row['target_percent']) == 55 else 10)
        for method in ('stc', 'ef_topk', 'strom', 'dense'):
            row = next(x for x in targets if x['panel'] == panel and x['reference'] == method and float(x['target_percent']) == 50)
            assert int(row['cost_wins']) == 10
            value = f"{float(row['saving_percent']):.1f}"
            assert value + r'\%' in source, ('missing 50% cost saving', panel, method, value)
    for method in ('event', 'strom', 'stc', 'ef_topk'):
        row = next(x for x in endpoints if x['panel'] == 'resnet14' and x['method'] == method)
        value = f"${100*float(row['final_test_accuracy_mean']):.2f}\\pm{100*float(row['final_test_accuracy_std']):.2f}\\%$"
        assert value in source, ('missing ResNet predictive endpoint', method, value)
        cost = f"{float(row['unicast_total_bits_mean'])/1e9:.3f}"
        assert cost + '~Gbit' in source, ('missing endpoint cost', method, cost)
    for method in ('stc', 'ef_topk'):
        row = next(x for x in effects if x['panel'] == 'resnet14' and x['reference'] == method and x['metric'] == 'accuracy_difference_pp')
        for field in ('estimate', 'ci95_low', 'ci95_high'):
            assert f"{float(row[field]):.2f}" in source
    required = (
        '14,400 unique training rounds', 'Historical budgets and selection rules',
        '95\\% Student-$t$ intervals on the log scale',
        'does not establish equivalence or noninferiority',
        'not a new untouched dataset', 'sparse STC development point',
        '128-bit header', 'U_R\\leq135P_R', 'byte-equivalent',
        'The STC interval', 'Broader datasets and model families remain',
        'without interpolation', 'not compute-matched',
    )
    for fragment in required:
        assert fragment in source, ('missing claim qualification', fragment)
    for stale in ('STC is not numerically evaluated', '38.66', '3,000 rounds', 'principal four-benchmark', 'nondominated among the tested mean'):
        assert stale not in source, ('stale headline', stale)
    # Supporting participation results keep their original frozen accounting.
    for filename in ("ten_seed_runs.csv", "stress_ten_seed_runs.csv"):
        path = PAPER.parents[1] / "experiments/results/p20_cifar10_partial_participation" / filename
        with path.open(newline="", encoding="utf-8") as stream:
            runs = list(csv.DictReader(stream))
        event = [r for r in runs if r["method"] == "event"]
        accuracy = [100*float(r["final_test_accuracy"]) for r in event]
        endpoint = f"${statistics.mean(accuracy):.2f}\\pm{statistics.stdev(accuracy):.2f}\\%$"
        assert endpoint in source, ("P20 supporting accuracy", endpoint)
        cost = statistics.mean(float(r["unicast_hybrid_total_bits"]) for r in event)/1e9
        assert f"{cost:.2f}" in source, ("P20 supporting cost", cost)
    # Preserve the reviewed finite-trajectory/conditional-expectation boundary.
    for fragment in ('This normalized trajectory ratio compares realized first-order descent',
                     'not the theoretical', '11.64', '[-12.79,-10.49]', '[-0.01,0.85]',
                     'cannot establish the conditional expectation', '29.4\\%'):
        assert fragment in source, ('mechanism/theory qualification drift', fragment)
    print('Validated calibrated cost/quality claims, all 24 target contrasts, historical audit boundaries, and tuning qualifications')


if __name__ == '__main__':
    main()
