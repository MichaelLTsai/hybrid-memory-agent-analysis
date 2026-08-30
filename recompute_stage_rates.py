#!/usr/bin/env python
"""以「階段別失敗率」重算 P1 / P4 / P5。

新定義（2026-08-25 起）：分母為全部題目，分子為歸入該階段且**答錯**的題目。

    P1 = |SUMMARY|   / N      該記的沒進記憶庫（NO_WRITE 併入此類）
    P4 = |RETRIEVAL| / N      存了卻沒撈到
    P5 = |REASONING| / N      料在眼前仍答錯

三者相加即為可歸因的錯誤率。皆為越低越好，與 MemFail 官方的
summary_error / retr_error / reason_error 採同一定義。

規則：
  · 階段失敗但答對的題目不計入分子（模型靠參數知識或猜的答對，
    記憶系統確實失敗，但依使用者裁定不列為失敗）。
  · 拒答題（P5b）不屬於任何階段，另計。
  · 裁判判不出來的題目（verdict 為 UNKNOWN / NO_DUMP）排除於分母。

    ./venv_memos/bin/python recompute_stage_rates.py
    ./venv_memos/bin/python recompute_stage_rates.py --json out.json
"""

import argparse
import glob
import json
import os
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))

# verdict → 階段。NO_WRITE 依使用者裁定併入 SUMMARY。
STAGE = {
    "SUMMARY": "P1",
    "NO_WRITE": "P1",
    "RETRIEVAL": "P4",
    "REASONING": "P5",
}
# 不屬於任何階段，但仍算在分母裡
NEUTRAL = {"OK"}
# 拒答題，另計，不進分母
ABSTAIN = {"P5b_FAIL", "P5b_OK"}
# 判不出來，排除
EXCLUDE = {"UNKNOWN", "NO_DUMP", None}

DATASETS = [
    ("LongMemEval", "longmemeval_experiment/results/*/*_lme_probe_detail.jsonl"),
    ("LoCoMo", "locomo_experiment/results/*/*_locomo_probe_detail.jsonl"),
    # probe_halumem_unified.py, not the retired probe_halumem.py: only the unified
    # probe carries P1 and the SUMMARY / RETRIEVAL vocabulary.
    ("HaluMem", "halumem_experiment/results/*/*_probe_unified.jsonl"),
]


def run_name_of(path):
    return os.path.basename(os.path.dirname(path))


def score_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    stage_fail = Counter()      # 階段 → 答錯且歸入該階段
    stage_but_right = Counter() # 階段失敗但答對（不計入分子，僅記錄）
    n = 0
    abstain = excluded = 0
    verdicts = Counter()

    for r in rows:
        v = r.get("verdict")
        verdicts[v] += 1
        if v in EXCLUDE:
            excluded += 1
            continue
        if v in ABSTAIN:
            abstain += 1
            continue
        n += 1
        st = STAGE.get(v)
        if st is None:
            continue                      # OK 或未知終點，只進分母
        if r.get("is_correct") is True:
            stage_but_right[st] += 1      # 依裁定不計為失敗
        else:
            stage_fail[st] += 1

    out = {
        "run": run_name_of(path),
        "n": n,
        "abstain_n": abstain,
        "excluded_n": excluded,
        "verdicts": dict(verdicts),
        "stage_but_correct": dict(stage_but_right),
    }
    for st in ("P1", "P4", "P5"):
        out[f"{st}_fail_n"] = stage_fail[st]
        out[f"{st}_fail"] = round(stage_fail[st] / n, 4) if n else None
    attributable = sum(stage_fail.values())
    out["attributable_error_rate"] = round(attributable / n, 4) if n else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="把結果寫成 JSON")
    args = ap.parse_args()

    results = {}
    for ds, pattern in DATASETS:
        paths = sorted(glob.glob(os.path.join(BASE, pattern)))
        rows = [score_file(p) for p in paths]
        results[ds] = rows

        print(f"\n{'=' * 96}\n{ds}  （{len(rows)} 個 run）\n{'=' * 96}")
        print(f"{'run':38s}{'N':>5s}{'P1↓':>8s}{'P4↓':>8s}{'P5↓':>8s}"
              f"{'錯誤率':>9s}{'失敗但答對':>11s}{'排除':>6s}")
        for r in rows:
            sbc = sum(r["stage_but_correct"].values())
            def f(x):
                return f"{x:.3f}" if isinstance(x, float) else "  -  "
            print(f"{r['run'][:37]:38s}{r['n']:>5d}"
                  f"{f(r['P1_fail']):>8s}{f(r['P4_fail']):>8s}{f(r['P5_fail']):>8s}"
                  f"{f(r['attributable_error_rate']):>9s}{sbc:>11d}{r['excluded_n']:>6d}")

    # NOT_RETRIEVED 是已退役的 probe_halumem.py 的用詞，它不做 P1，讀到就會讓
    # P1 與 P4 靜默算成 0。正常情況下不該再出現，出現代表讀到了舊檔。
    hal = results.get("HaluMem") or []
    if hal and any("NOT_RETRIEVED" in r["verdicts"] for r in hal):
        print("\n警告：HaluMem 出現 NOT_RETRIEVED verdict，那是已退役的 probe_halumem.py 的輸出。")
        print("      該檔不含 P1，上表的 P1 與 P4 會是 0。請改跑 probe_halumem_unified.py。")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n已寫入 {args.json}")


if __name__ == "__main__":
    main()
