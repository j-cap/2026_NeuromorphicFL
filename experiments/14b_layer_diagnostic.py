from pathlib import Path
from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation, run_federated_method

ROOT = Path(__file__).resolve().parents[1]
fed = make_multiclass_federation(root=ROOT / 'data' / 'fashion-mnist', regime='strong', seed=2400)
r = run_federated_method(
    federation=fed,
    method='events',
    n_ticks=650,
    gamma=1.0,
    threshold=0.025,
    jump0=0.0035,
    eval_stride=100,
)
keys = [
    'final_test_ce','final_test_accuracy','final_worst_class_accuracy','payload_bits',
    'candidate_events','messages','events_per_message','ever_fired_fraction',
    'W1_events_per_param','W1_never_fired','b1_events_per_param','b1_never_fired',
    'W2_events_per_param','W2_never_fired','b2_events_per_param','b2_never_fired',
    'W3_events_per_param','W3_never_fired','b3_events_per_param','b3_never_fired',
]
for k in keys:
    print(f'{k}={r[k]}')
