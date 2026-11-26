"""Check the current calibrated-panel narrative against regenerated evidence.

Historical evidence remains independently checked by its original builders.
Current headline: the closed CIFAR and Fashion-MNIST three-architecture panel.
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
    mlp_endpoints = [r for r in read('three_architecture_endpoints.csv') if r['panel'] == 'fmnist_mlp']
    mlp_targets = [r for r in read('three_architecture_targets.csv') if r['panel'] == 'fmnist_mlp']
    mlp_effects = [r for r in read('three_architecture_endpoint_effects.csv') if r['panel'] == 'fmnist_mlp']
    coverage = [r for r in read('three_architecture_coverage.csv') if r['panel'] == 'fmnist_mlp']
    event = next(r for r in mlp_endpoints if r['method'] == 'event')
    for row in mlp_endpoints:
        assert int(row['n']) == 10 and int(row['rounds']) == 1500
        if row['method'] != 'event':
            assert float(event['unicast_total_bits_mean']) < float(row['unicast_total_bits_mean'])
    for row in mlp_effects:
        if row['metric'] == 'unicast_ratio':
            assert int(row['paired_cost_wins']) == 10 and float(row['ci95_high']) < 1
        if row['metric'] == 'accuracy_difference_pp':
            assert float(row['ci95_low']) < 0 < float(row['ci95_high'])
    for method in ('event', 'stc', 'strom'):
        row = next(r for r in mlp_endpoints if r['method'] == method)
        endpoint = "$"+f"{100*float(row['final_test_accuracy_mean']):.2f}\\pm{100*float(row['final_test_accuracy_std']):.2f}\\%$"
        assert endpoint in source, ('missing MLP endpoint', method, endpoint)
        assert f"{float(row['unicast_total_bits_mean'])/1e9:.3f}~Gbit" in source
    for row in mlp_targets:
        if int(row['target_percent']) in (80, 85):
            assert int(row['common_hits']) == 10
    strom85 = next(r for r in mlp_targets if r['reference'] == 'strom' and int(r['target_percent']) == 85)
    stc85 = next(r for r in mlp_targets if r['reference'] == 'stc' and int(r['target_percent']) == 85)
    assert float(strom85['ratio_ci95_low']) > 1 and int(strom85['later']) == 9
    assert float(stc85['ratio_ci95_high']) < 1 and int(stc85['cost_wins']) == 10
    for method, hits, common in (('strom', 8, 7), ('stc', 9, 9), ('ef_topk', 7, 6), ('dense', 9, 8)):
        r = next(r for r in coverage if r['method'] == method and int(r['target_percent']) == 87)
        p = next(r for r in mlp_targets if r['reference'] == method and int(r['target_percent']) == 87)
        assert int(r['reached']) == hits and int(p['common_hits']) == common
    for fragment in (
        'three architectures on two datasets', '150 method--seed outcomes',
        '12,000 per MLP family', '54,000/6,000', 'full reset',
        '17.6\\% [11.9,22.8]', '10.0\\% [0.7,20.3] more traffic',
        '37.6\\% [37.2,37.9]', '14.5\\% [14.2,14.8]',
        'All four MLP paired accuracy intervals span zero',
        'Only three Event runs finish at or above 87\\%',
        'common hits are 7/9/6/8', 'gain three', '1,500 rounds',
    ):
        assert fragment in source, ('missing MLP qualification', fragment)
    assert 'figures/three_architecture_trajectories.pdf' in source
    assert 'figures/three_architecture_target_ratios.pdf' in source
    assert 'generated/three_architecture_results' in source
    assert 'generated/three_architecture_targets' in source
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
    print('Validated three-architecture cost/quality claims, all 36 target contrasts, MLP Ström exception/censoring, historical audit boundaries and tuning qualifications')


if __name__ == '__main__':
    main()
