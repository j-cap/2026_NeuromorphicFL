#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT_NAME="neuromorphicfl-resnet14"
ACTION="${1:-help}"

run_python() {
  conda run --no-capture-output -n "${ENVIRONMENT_NAME}" \
    env PYTHONPATH=src python experiments/p13_cifar10_resnet14_torch.py "$@"
}

case "${ACTION}" in
  setup)
    conda env update --name "${ENVIRONMENT_NAME}" \
      --file environment-resnet14.yml --prune
    conda run --no-capture-output -n "${ENVIRONMENT_NAME}" python -c \
      'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
    ;;
  smoke)
    run_python --device cuda smoke --rounds 2
    ;;
  dense-audit)
    run_python --device cuda dense-audit --rounds "${2:-900}"
    ;;
  dense-horizon)
    run_python --device cuda dense-horizon --rounds "${2:-3600}"
    ;;
  development)
    run_python --device cuda development-campaign
    ;;
  test)
    run_python --device cuda test-campaign
    run_python --device cuda verify
    ;;
  full)
    run_python --device cuda full-campaign
    run_python --device cuda verify
    ;;
  final)
    run_python --device cuda --torch-threads "${3:-8}" final-campaign --workers "${2:-2}"
    run_python --device cpu final-summary
    run_python --device cpu verify
    ;;
  final-dry-run)
    run_python --device cuda --torch-threads "${3:-8}" final-campaign --workers "${2:-2}" --dry-run
    ;;
  verify)
    run_python --device cpu verify
    ;;
  status)
    run_python --device cpu status
    ;;
  *)
    echo "Usage: $0 {setup|smoke|dense-audit [rounds]|dense-horizon [rounds]|development|test|full|final [workers] [threads-per-worker]|final-dry-run [workers] [threads-per-worker]|verify|status}"
    exit 2
    ;;
esac
