"""Audit the six exported held-out checkpoints, without inference."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'evaluation'))
from metrics import load_predictions, load_ground_truth, compute_all_metrics


def main():
    report = json.loads(Path(__file__).with_name('summary.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(ROOT / report['dataset']) == report['dataset_sha256']
    assert sha(ROOT / 'evaluation/metrics.py') == report['metrics_sha256']
    for name, digest in report['artifact_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in report['uncompressed_log_sha256'].items():
        assert hashlib.sha256(gzip.decompress((ROOT / name).read_bytes())).hexdigest() == digest, name
    gt = load_ground_truth(ROOT / report['dataset'])
    for iteration, entry in report['checkpoints'].items():
        raw = load_predictions(ROOT / entry['predictions'])
        chosen = {}
        for row in raw:
            k = row['index']
            if k not in chosen or chosen[k].get('vulnerable') is None or row.get('vulnerable') is not None:
                chosen[k] = row
        assert len(raw) == entry['raw_records']
        assert set(chosen) == set(range(870))
        assert all(r['uuid'] == gt[k]['uuid'] and r['label'] == gt[k]['target'] for k,r in chosen.items())
        assert compute_all_metrics(list(chosen.values()), ground_truth=gt) == entry['metrics']
        print(f"iteration {iteration}: {entry['metrics']['pairwise_accuracy']:.2f}% pairwise; localization F1 {entry['metrics']['loc_f1']:.2f}")
    print('PASS: hashes, 870 unique samples / 435 pairs per checkpoint, and all metrics match.')


if __name__ == '__main__':
    main()
