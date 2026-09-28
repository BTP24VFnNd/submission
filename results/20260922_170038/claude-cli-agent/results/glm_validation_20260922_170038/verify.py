"""Verify the saved validation run without inference or credentials."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / 'meta-vul-code/meta-vul/outputs/20260922_170038'
sys.path.insert(0, str(ROOT / 'meta-vul-code/meta-vul'))
from runner.eval_prompt import load_ground_truth
from metrics import compute_all_metrics, load_predictions


def records(text):
    decoder, pos = json.JSONDecoder(), 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            break
        row, pos = decoder.raw_decode(text, pos)
        yield row


def main():
    provenance = json.loads((REPORT / 'provenance.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    dataset = ROOT / provenance['dataset']
    assert sha(dataset) == provenance['dataset_sha256']
    assert sha(ROOT / 'evaluation/metrics.py') == provenance['metrics_sha256']
    for name, digest in provenance['artifact_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in provenance['uncompressed_log_sha256'].items():
        assert hashlib.sha256(gzip.decompress((ROOT / name).read_bytes())).hexdigest() == digest, name
    # The input has 202 paired rows, not 404 flat rows. Use the optimizer's
    # loader so CWE and localization metrics align with the prediction indices.
    gt = load_ground_truth(dataset)
    pairs = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    summary = json.loads((SESSION / 'summary.json').read_text())
    assert summary['history'] == json.loads((SESSION / 'history.json').read_text())
    assert (SESSION / 'best_prompt.txt').read_bytes() == (SESSION / 'iter_004.txt').read_bytes()
    assert len(summary['history']) == 9 and len(gt) == 404
    for entry in summary['history']:
        i = entry['iteration']
        folder = SESSION / f'iter_{i:03d}'
        preds = load_predictions(folder / 'predictions.jsonl')
        assert len(preds) == 404 and {r['index'] for r in preds} == set(range(404)), i
        assert all(r['label'] == gt[r['index']]['target'] and r['uuid'] == pairs[r['index']//2]['uuid'][r['index']%2] for r in preds), i
        logs = list(records(gzip.decompress((folder / 'logs.jsonl.gz').read_bytes()).decode()))
        assert {r['index'] for r in logs} == set(range(404)), i
        assert {r['model'] for r in logs} == {'z-ai/glm-5.3'}, i
        metrics = compute_all_metrics(preds, ground_truth=gt)
        assert all(metrics[k] == v for k, v in entry['metrics'].items() if k != 'session'), i
        assert json.loads((folder / 'metrics.json').read_text()) == entry['metrics'], i
        assert metrics['total_pairs'] == 202, i
        print(f"iter {i}: {metrics['pairwise_accuracy']:.2f}% pairwise; {metrics['parse_failures']} unparseable")
    print('PASS: hashes, coverage, UUIDs, labels, models and recorded metrics match.')


if __name__ == '__main__':
    main()
