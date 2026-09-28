#!/usr/bin/env python3
"""
Compute evaluation metrics from predictions.jsonl files.

Metrics: precision, recall, F1, accuracy, pairwise accuracy, per-CWE breakdown.
Supports single file or directory of multiple strategies.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file (supports pretty-printed records)."""
    rows = []
    buf = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            buf += line
            try:
                obj = json.loads(buf)
                rows.append(obj)
                buf = ""
            except json.JSONDecodeError:
                continue
    return rows


def binary_metrics(labels: List[int], preds: List[bool]) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for label, pred in zip(labels, preds):
        if label == 1 and pred:
            tp += 1
        elif label == 0 and pred:
            fp += 1
        elif label == 1 and not pred:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def pairwise_accuracy(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute pairwise accuracy over consecutive index pairs: (0,1), (2,3), etc.
    Each pair should share the same CVE and have one vulnerable (label=1) and one patched (label=0).
    Pairs where either sample has a parse failure are excluded from the total.
    """
    by_index = {r.get("index"): r for r in preds}
    if not by_index:
        return {"pairwise_accuracy": 0.0, "correct_pairs": 0, "total_pairs": 0}

    max_idx = max(by_index.keys())
    correct = 0
    total = 0
    for i in range(0, max_idx + 1, 2):
        a = by_index.get(i)
        b = by_index.get(i + 1)
        if not a or not b:
            continue
        vuln = a if a.get("label") == 1 else b if b.get("label") == 1 else None
        patch = a if a.get("label") == 0 else b if b.get("label") == 0 else None
        if not vuln or not patch:
            continue
        vuln_pred = vuln.get("vulnerable")
        patch_pred = patch.get("vulnerable")
        total += 1
        if vuln_pred is True and patch_pred is False:
            correct += 1

    pairwise_acc = (correct / total * 100) if total > 0 else 0.0
    return {
        "pairwise_accuracy": round(pairwise_acc, 2),
        "correct_pairs": correct,
        "total_pairs": total,
    }


def _get_gt_cwes(gt: Dict[str, Any]) -> set:
    """Extract ground truth CWEs as an upper-cased set."""
    raw = gt.get("cwe")
    if not raw:
        return set()
    if isinstance(raw, str):
        raw = [raw]
    return {c.strip().upper() for c in raw}


def _get_pred_cwes(r: Dict[str, Any]) -> set:
    """Extract predicted CWEs as an upper-cased set."""
    verdict = r.get("verdict_json") or {}
    pred_cwes = (verdict.get("verdict") or {}).get("cwe") or []
    if isinstance(pred_cwes, str):
        pred_cwes = [pred_cwes]
    if not isinstance(pred_cwes, (list, tuple, set)):
        return set()
    return {c.strip().upper() for c in pred_cwes if isinstance(c, str)}


def _get_top1_cwe(r: Dict[str, Any]) -> str:
    """First predicted CWE, upper-cased, or "" when none is given."""
    verdict = r.get("verdict_json") or {}
    pred_cwes = (verdict.get("verdict") or {}).get("cwe") or []
    if isinstance(pred_cwes, str):
        pred_cwes = [pred_cwes]
    if not isinstance(pred_cwes, (list, tuple)) or not pred_cwes or not isinstance(pred_cwes[0], str):
        return ""
    return pred_cwes[0].strip().upper()


def _snippet_text(value: Any) -> str:
    """Evidence snippet as text. Some models return a list of lines instead of a
    string; join those, and ignore any other type."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(v) for v in value if v is not None)
    return ""


def _get_pred_lines(r: Dict[str, Any], func_source: str = "") -> set:
    """Extract predicted vulnerable lines by matching evidence snippets against
    the function source. Snippet matching is the primary method since models
    often report correct code but wrong line numbers. Reported line numbers are
    used only as a fallback when no snippets are available."""
    verdict = r.get("verdict_json") or {}
    evidence_list = verdict.get("evidence", [])
    if not isinstance(evidence_list, list):
        return set()
    snippets = []
    line_numbers = set()
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        snippet = _snippet_text(ev.get("snippet"))
        if snippet:
            snippets.append(snippet)
        loc = ev.get("location", {})
        if isinstance(loc, dict):
            start = loc.get("start_line")
            end = loc.get("end_line")
            if isinstance(start, int) and isinstance(end, int):
                line_numbers.update(range(start, end + 1))
            elif isinstance(start, int):
                line_numbers.add(start)
    # Primary: match snippets against function source
    if snippets and func_source:
        matched = _snippet_to_lines(snippets, func_source)
        if matched:
            return matched
    # Fallback: use reported line numbers
    return line_numbers


def _snippet_to_lines(snippets: List[str], func_source: str) -> set:
    """Match evidence snippet lines to function source lines.
    Only matches lines with >= 10 non-trivial characters to avoid false positives."""
    func_lines = func_source.splitlines()
    matched = set()
    for snippet in snippets:
        for sp in snippet.splitlines():
            cleaned = re.sub(r'\.\.\.|/\*.*?\*/|//.*', '', sp).strip()
            if len(cleaned) < 10:
                continue
            for line_no, func_line in enumerate(func_lines, 1):
                fl = func_line.strip()
                if len(fl) < 10:
                    continue
                if cleaned == fl or cleaned in fl or fl in cleaned:
                    matched.add(line_no)
    return matched


def cwe_metrics(preds: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    CWE identification accuracy over true positives (label=1 and predicted vulnerable=true).
    A sample is correct if the ground truth CWE appears in the predicted CWE list.
    Denominator = TP samples with a known ground truth CWE.
    """
    gt_by_index = {i: row for i, row in enumerate(ground_truth)}
    total = 0
    correct = 0
    for r in preds:
        if r.get("label") != 1 or r.get("vulnerable") is not True:
            continue
        idx = r.get("index")
        gt = gt_by_index.get(idx)
        if not gt:
            continue
        gt_cwes = _get_gt_cwes(gt)
        if not gt_cwes:
            continue
        total += 1
        pred_cwes = _get_pred_cwes(r)
        if gt_cwes & pred_cwes:
            correct += 1
    acc = (correct / total * 100) if total > 0 else 0.0
    return {
        "cwe_accuracy": round(acc, 2),
        "cwe_correct": correct,
        "cwe_total": total,
    }


def localization_metrics(preds: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Line-level localization precision, recall, and F1 over true positives
    (label=1 and predicted vulnerable=true) that have known vuln_lines.
    Denominator = TP samples with ground truth vuln lines.
    """
    gt_by_index = {i: row for i, row in enumerate(ground_truth)}
    total_samples = 0
    samples_with_predictions = 0
    precision_sum = 0.0
    recall_sum = 0.0
    hit_count = 0  # binary: any overlap at all

    for r in preds:
        if r.get("label") != 1 or r.get("vulnerable") is not True:
            continue
        idx = r.get("index")
        gt = gt_by_index.get(idx)
        if not gt or not gt.get("vuln_lines"):
            continue
        gt_lines = set(gt["vuln_lines"])
        total_samples += 1

        func_source = gt.get("func", "")
        pred_lines = _get_pred_lines(r, func_source=func_source)
        if not pred_lines:
            # predicted vulnerable but no evidence lines
            continue

        samples_with_predictions += 1
        overlap = gt_lines & pred_lines
        p = len(overlap) / len(pred_lines)
        rec = len(overlap) / len(gt_lines)
        precision_sum += p
        recall_sum += rec
        if overlap:
            hit_count += 1

    # Macro-average precision/recall over samples that have predictions
    avg_precision = (precision_sum / samples_with_predictions * 100) if samples_with_predictions > 0 else 0.0
    # Macro-average recall over ALL vulnerable samples with GT lines
    avg_recall = (recall_sum / total_samples * 100) if total_samples > 0 else 0.0
    f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0.0

    hit_rate = (hit_count / total_samples * 100) if total_samples > 0 else 0.0

    return {
        "loc_precision": round(avg_precision, 2),
        "loc_recall": round(avg_recall, 2),
        "loc_f1": round(f1, 2),
        "loc_hit_rate": round(hit_rate, 2),
        "loc_samples_with_preds": samples_with_predictions,
        "loc_total": total_samples,
    }


def _snippet_matches_content(snippets: List[str], gt_contents: List[str]) -> bool:
    """Check if any evidence snippet line matches any ground truth vulnerable
    line content. Same containment logic as _snippet_to_lines but compares
    directly against GT content strings instead of the full function source."""
    if not snippets or not gt_contents:
        return False
    gt_cleaned = [re.sub(r'\.\.\.|/\*.*?\*/|//.*', '', c).strip() for c in gt_contents]
    gt_cleaned = [c for c in gt_cleaned if len(c) >= 10]
    if not gt_cleaned:
        return False
    for snippet in snippets:
        for sp in snippet.splitlines():
            cleaned = re.sub(r'\.\.\.|/\*.*?\*/|//.*', '', sp).strip()
            if len(cleaned) < 10:
                continue
            for gc in gt_cleaned:
                if cleaned == gc or cleaned in gc or gc in cleaned:
                    return True
    return False


def joint_cwe_loc_metrics(preds: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Joint metric: among TPs, how many get BOTH the CWE correct AND localization
    correct (snippet hit)? Denominator = TP samples that have both a known CWE
    and known vuln_lines.
    """
    gt_by_index = {i: row for i, row in enumerate(ground_truth)}
    total = 0
    cwe_only = 0
    loc_only = 0
    both = 0
    neither = 0
    for r in preds:
        if r.get("label") != 1 or r.get("vulnerable") is not True:
            continue
        idx = r.get("index")
        gt = gt_by_index.get(idx)
        if not gt:
            continue
        gt_cwes = _get_gt_cwes(gt)
        if not gt_cwes:
            continue
        total += 1
        cwe_hit = bool(gt_cwes & _get_pred_cwes(r))
        gt_vuln_contents = gt.get("vuln_line_contents", [])
        verdict = r.get("verdict_json") or {}
        evidence_list = verdict.get("evidence", [])
        snippets = []
        if isinstance(evidence_list, list):
            for ev in evidence_list:
                if isinstance(ev, dict) and _snippet_text(ev.get("snippet")):
                    snippets.append(_snippet_text(ev["snippet"]))
        loc_hit = _snippet_matches_content(snippets, gt_vuln_contents)
        if cwe_hit and loc_hit:
            both += 1
        elif cwe_hit:
            cwe_only += 1
        elif loc_hit:
            loc_only += 1
        else:
            neither += 1
    both_rate = (both / total * 100) if total > 0 else 0.0
    return {
        "joint_total": total,
        "joint_both": both,
        "joint_both_rate": round(both_rate, 2),
        "joint_cwe_only": cwe_only,
        "joint_loc_only": loc_only,
        "joint_neither": neither,
    }


def joint_top1_metrics(preds: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Refined joint metric: among TPs, how many get the FIRST predicted CWE right
    AND hit at least one ground truth vulnerable line (same line matching as
    loc_hit_rate)? Scoring only the first CWE stops long CWE lists from earning
    credit. Denominator = TP samples that have both a known CWE and vuln_lines.

    Setup. A harness examines a set of code functions. Each function i has:
      - a true label y_i: 1 if the function is vulnerable, 0 if it is the
        patched (safe) version;
      - a prediction y^_i: 1 if the harness says "vulnerable", 0 otherwise.
    For vulnerable functions we also know two facts from the dataset:
      - C_i, the set of correct CWE categories (usually one, sometimes several);
      - L_i, the set of line numbers where the vulnerability is.
    When the harness flags a function, it also returns:
      - a list of CWE guesses (c^_i(1), c^_i(2), ...), where c^_i(1) is the
        first guess;
      - a set of lines it points to as evidence, L^_i.

    Definition 1 (the samples we score). Let T be the set of functions that are
    vulnerable, are flagged by the harness, and have both a known CWE and known
    vulnerable lines:
        T = { i : y_i = 1, y^_i = 1, C_i != {}, L_i != {} }
    These are the true positives we can check.

    Definition 2 (correct category). Function i has the correct category if the
    first CWE guess is one of the true CWEs:
        cat_i = 1 if c^_i(1) in C_i, else 0

    Definition 3 (correct location). Function i has a correct location if the
    harness points to at least one true vulnerable line:
        loc_i = 1 if L^_i ∩ L_i != {}, else 0

    Definition 4 (joint score). Function i is fully correct only if both
    conditions hold. Since each is 0 or 1, their product is 1 exactly when both
    are 1:
        joint_i = cat_i * loc_i

    Definition 5 (the metric). The joint metric is the percentage of fully
    correct functions in T:
        joint_top1_rate = 100 / |T| * sum over i in T of joint_i
    """
    gt_by_index = {i: row for i, row in enumerate(ground_truth)}
    total = 0
    both = 0
    for r in preds:
        if r.get("label") != 1 or r.get("vulnerable") is not True:
            continue
        gt = gt_by_index.get(r.get("index"))
        if not gt or not gt.get("vuln_lines"):
            continue
        gt_cwes = _get_gt_cwes(gt)
        if not gt_cwes:
            continue
        total += 1
        cwe_hit = _get_top1_cwe(r) in gt_cwes
        loc_hit = bool(set(gt["vuln_lines"]) & _get_pred_lines(r, func_source=gt.get("func", "")))
        if cwe_hit and loc_hit:
            both += 1
    rate = (both / total * 100) if total > 0 else 0.0
    return {
        "joint_top1_total": total,
        "joint_top1_both": both,
        "joint_top1_rate": round(rate, 2),
    }


def compute_all_metrics(preds: List[Dict[str, Any]], ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compute all metrics for predictions."""
    valid = [(r["label"], r["vulnerable"]) for r in preds
             if r["label"] in (0, 1) and r["vulnerable"] is not None]

    if not valid:
        return {"error": "No valid predictions with labels found"}

    labels, pred_flags = zip(*valid)
    metrics = binary_metrics(list(labels), list(pred_flags))

    # Report sample counts clearly
    n = len(valid)
    total = len(preds)
    metrics["n"] = n  # Main sample count
    metrics["total"] = total
    metrics["parse_failures"] = sum(1 for r in preds if r["vulnerable"] is None)
    metrics["policies"] = sorted(set(r["policy"] for r in preds))

    # Add pairwise accuracy
    pairwise = pairwise_accuracy(preds)
    metrics.update(pairwise)

    # Add CWE and localization metrics if ground truth available
    if ground_truth:
        metrics.update(cwe_metrics(preds, ground_truth))
        metrics.update(localization_metrics(preds, ground_truth))
        metrics.update(joint_cwe_loc_metrics(preds, ground_truth))
        metrics.update(joint_top1_metrics(preds, ground_truth))

    return metrics


def compute_intersection_metrics(
    all_preds: Dict[str, List[Dict[str, Any]]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Find samples where ALL policies correctly predict vulnerable (TP intersection).
    Compute per-policy CWE and localization metrics on only those shared samples.
    """
    # Build index -> pred mapping per strategy
    by_strategy = {}
    for strategy, preds in all_preds.items():
        by_strategy[strategy] = {r.get("index"): r for r in preds}

    strategies = sorted(by_strategy.keys())

    # Find indices that are TP in ALL strategies
    all_indices = set()
    for s in strategies:
        s_indices = {idx for idx, r in by_strategy[s].items()
                     if r.get("label") == 1 and r.get("vulnerable") is True}
        if not all_indices:
            all_indices = s_indices
        else:
            all_indices &= s_indices

    gt_by_index = {i: row for i, row in enumerate(ground_truth)}
    intersection_size = len(all_indices)

    result = {"intersection_tp_count": intersection_size, "strategies": strategies}

    # Compute per-policy CWE and localization on intersection only
    for strategy in strategies:
        filtered = [by_strategy[strategy][idx] for idx in sorted(all_indices)
                    if idx in by_strategy[strategy]]
        result[strategy] = {
            **cwe_metrics(filtered, ground_truth),
            **localization_metrics(filtered, ground_truth),
            **joint_cwe_loc_metrics(filtered, ground_truth),
        }

    return result


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    """Load ground truth JSONL (one record per line)."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics from predictions")
    parser.add_argument("--predictions", default="", help="Path to single predictions.jsonl file")
    parser.add_argument("--outputs-dir", default="", help="Path to outputs directory with multiple strategies")
    parser.add_argument("--ground-truth", default="", help="Path to ground truth JSONL with cwe and vuln_lines fields")
    parser.add_argument("--output", default="", help="Path to write metrics JSON (default: stdout)")
    args = parser.parse_args()

    if not args.predictions and not args.outputs_dir:
        print("Error: provide either --predictions or --outputs-dir", file=sys.stderr)
        sys.exit(1)

    gt = None
    if args.ground_truth:
        gt = load_ground_truth(Path(args.ground_truth))

    result = {}

    if args.predictions:
        # Single file mode
        preds = load_predictions(Path(args.predictions))
        result = compute_all_metrics(preds, ground_truth=gt)
    else:
        # Directory mode - analyze all strategies
        outputs_dir = Path(args.outputs_dir)
        all_preds = {}  # strategy -> list of preds
        for strategy_dir in sorted(outputs_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            pred_file = strategy_dir / "predictions.jsonl"
            if not pred_file.exists():
                continue
            strategy = strategy_dir.name
            preds = load_predictions(pred_file)
            all_preds[strategy] = preds
            result[strategy] = compute_all_metrics(preds, ground_truth=gt)

        # Add comparison if multiple strategies
        if len(result) > 1:
            accs = {s: r.get("accuracy", 0) for s, r in result.items() if isinstance(r, dict)}
            pairwise_accs = {s: r.get("pairwise_accuracy", 0) for s, r in result.items() if isinstance(r, dict)}
            result["comparison"] = {
                "best_accuracy": max(accs.values()) if accs else 0,
                "best_pairwise_accuracy": max(pairwise_accs.values()) if pairwise_accs else 0,
                "all_accuracies": accs,
                "all_pairwise_accuracies": pairwise_accs,
            }

        # Intersection metrics: samples where ALL policies predict TP
        if gt and len(all_preds) > 1:
            result["intersection"] = compute_intersection_metrics(all_preds, gt)

    out = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(out)
        print(f"Wrote metrics to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
