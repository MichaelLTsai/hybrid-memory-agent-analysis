"""
LoCoMo memory extraction evaluation: the extraction-stage metrics LoCoMo does not ship.

LoCoMo's official evaluation covers only the read side (qa_accuracy, token_f1,
retrieval recall@k) and has no metric that uses the 2,541 `observation`
annotations it provides. This script treats those annotations as golden memories
and measures extraction quality:

  integrity (recall)    was the fact that should have been recorded recorded
                        (a miss is a SUMMARY FAILURE)
  accuracy  (precision) is what was recorded correct (fabrication)
  extraction_f1         harmonic mean of the two

Adjudication reuses HaluMem's two official judge prompts, imported unmodified
from halumem_experiment, so the resulting figures share a definition with
HaluMem's memory_integrity / memory_accuracy / memory_extraction_f1 and are
directly comparable across datasets.

One thing is specific to LoCoMo: it is a **two-party** dialogue
(speaker_a / speaker_b) and observations are annotated per speaker. Recording a
fact correctly but attributing it to the wrong person is mechanically speaker
confusion rather than loss during compression, so it is counted separately as
speaker_confusion and never folded into the integrity failures. HaluMem has a
single user and cannot measure this dimension at all.

Scoping:
  Each observation carries its source turn (such as "D1:9"), and any memory
  carrying a dia_id can be traced back to its session. Comparison is therefore
  scoped to **the same session**, aligning with HaluMem, where the dataset gives
  the session directly. Backends without turn provenance (letta, graphiti) fall
  back to whole-store comparison and are marked scope="global"; those figures are
  not directly comparable to session-scoped ones.

Nothing is guessed: a judge failure returns None, is excluded from the
denominator, and its cause is recorded in judge_failures.

Usage:
  python extraction_locomo.py --run mem0-default
  python extraction_locomo.py --run mem0-default --max-obs 200   # sample to control cost
"""

import os
import re
import sys
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

from llms import llm_request_for_json
# Reuse HaluMem's official judge prompts unmodified, keeping the definition
# identical across datasets
from eval_tools import (
    EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY,
    EVALUATION_PROMPT_FOR_MEMORY_ACCURACY,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "locomo10.json")

# Specific to LoCoMo: the fact is right but attributed to the wrong speaker
SPEAKER_ATTRIBUTION_PROMPT = """You are auditing an AI memory system built from a conversation between two people: **{speaker_a}** and **{speaker_b}**.

# Extracted Memories
{memories}

# Expected Memory Point (about {owner})
{expected_memory_point}

The system FAILED to capture this memory point as stated. Decide which case applies:

- "missing"      : the underlying fact does not appear in the Extracted Memories at all.
- "wrong_owner"  : the same underlying fact IS present, but attributed to the OTHER person
                   (or to an unnamed/incorrect subject) instead of {owner}.

Return strictly this JSON:
```json
{{"verdict": "missing" | "wrong_owner"}}
```
"""

# Judge-failure counters (thread-safe)
_FAILS = Counter()
_LOCK = threading.Lock()
_WARNED = set()


def _note_failure(kind: str, exc: Exception) -> None:
    with _LOCK:
        _FAILS[kind] += 1
        first = kind not in _WARNED
        _WARNED.add(kind)
    if first:
        print(f"[extraction_locomo] {kind} judge failure (further ones of this kind are not printed): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


# ── Parsing helpers ───────────────────────────────────────────────────────────

_DIA_RE = re.compile(r"^D(\d+):")
_SESS_RE = re.compile(r"^session_(\d+)")


def session_of_dia(dia_id) -> int | None:
    """'D1:9' → 1"""
    if not isinstance(dia_id, str):
        return None
    m = _DIA_RE.match(dia_id.strip())
    return int(m.group(1)) if m else None


def _flat_ids(x) -> list[str]:
    """An observation's source field is sometimes a nested list; flatten and pull out D*:* ids."""
    if isinstance(x, str):
        return [x] if x.startswith("D") else []
    out = []
    if isinstance(x, (list, tuple)):
        for i in x:
            out += _flat_ids(i)
    return out


def build_golden(sample_raw: dict) -> dict[int, list[dict]]:
    """Build session -> [facts that should be recorded] from locomo10.json observations."""
    out = defaultdict(list)
    for key, per_speaker in (sample_raw.get("observation") or {}).items():
        m = _SESS_RE.match(key)
        if not m:
            continue
        sid = int(m.group(1))
        for owner, items in per_speaker.items():
            for it in items:
                if isinstance(it, (list, tuple)) and it:
                    text = it[0]
                    srcs = _flat_ids(it[1:])
                else:
                    text, srcs = it, []
                if isinstance(text, str) and text.strip():
                    out[sid].append({"text": text.strip(), "owner": owner,
                                     "source_dia_ids": srcs, "session": sid})
    return out


def build_dialogue(sample_raw: dict) -> dict[int, str]:
    """session -> full dialogue text (needed by the accuracy judge)."""
    out = {}
    conv = sample_raw.get("conversation") or {}
    for key, turns in conv.items():
        m = _SESS_RE.match(key)
        if not m or not isinstance(turns, list):
            continue
        out[int(m.group(1))] = "\n".join(
            f'{t.get("speaker")}: {t.get("text","")}' for t in turns)
    return out


def group_memories(memory_dump: list) -> tuple[dict[int, list[str]], str]:
    """
    Group memories by session. Returns (groups, scope).

    scope="session" means turn provenance is available; "global" means the whole
    store cannot be split by session.
    """
    by_sess = defaultdict(list)
    allm, has_prov = [], False
    for m in memory_dump or []:
        text = (m.get("text") or "").strip() if isinstance(m, dict) else str(m).strip()
        if not text:
            continue
        allm.append(text)
        sid = session_of_dia(m.get("dia_id")) if isinstance(m, dict) else None
        if sid is not None:
            has_prov = True
            by_sess[sid].append(text)
    if has_prov:
        return by_sess, "session"
    return {None: allm}, "global"


def _mem_block(mems: list[str]) -> str:
    return "\n".join(f"- {m}" for m in mems) if mems else "(none)"


# ── LLM judges ────────────────────────────────────────────────────────────────

def _score_from(result: dict, keys=("score", "accuracy_score")):
    """
    Extract the 0/1/2 score. The two HaluMem prompts use different field names:
      integrity → "score"        accuracy → "accuracy_score"
    and the value is a string ("2"). Returns None when unadjudicable, for the caller to handle.
    """
    v = None
    for k in keys:
        if k in result:
            v = result[k]
            break
    if v is None:
        raise ValueError(f"no score field among {keys}; got: {list(result)}")
    if isinstance(v, bool):
        raise ValueError(f"unexpected score type: {v!r}")
    try:
        s = int(str(v).strip())
    except Exception as e:
        raise ValueError(f"unparsable score: {v!r}") from e
    if s not in (0, 1, 2):
        raise ValueError(f"score out of range: {s}")
    return s


def judge_integrity(mems: list[str], expected: str):
    """Was the fact that should have been recorded recorded. 0/1/2, or None if unadjudicable."""
    prompt = EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY.format(
        memories=_mem_block(mems), expected_memory_point=expected)
    try:
        return _score_from(llm_request_for_json(prompt))
    except Exception as e:
        _note_failure("integrity", e)
        return None


def judge_accuracy(dialogue: str, golden: list[str], candidate: str):
    """Is this extracted memory correct. 0/1/2, or None if unadjudicable."""
    prompt = EVALUATION_PROMPT_FOR_MEMORY_ACCURACY.format(
        dialogue=dialogue or "(unavailable)",
        golden_memories=_mem_block(golden),
        candidate_memory=candidate)
    try:
        return _score_from(llm_request_for_json(prompt))
    except Exception as e:
        _note_failure("accuracy", e)
        return None


def judge_speaker(mems: list[str], expected: str, owner: str, spk_a: str, spk_b: str):
    """On an integrity failure, ask again whether it vanished entirely or went to the wrong speaker. None if unadjudicable."""
    prompt = SPEAKER_ATTRIBUTION_PROMPT.format(
        speaker_a=spk_a, speaker_b=spk_b, owner=owner,
        memories=_mem_block(mems), expected_memory_point=expected)
    try:
        v = str(llm_request_for_json(prompt).get("verdict", "")).strip().lower()
        if v not in ("missing", "wrong_owner"):
            raise ValueError(f"unexpected verdict: {v!r}")
        return v
    except Exception as e:
        _note_failure("speaker", e)
        return None


# ── Main flow ─────────────────────────────────────────────────────────────────

def evaluate_sample(sample: dict, raw: dict, max_workers: int = 8,
                    max_obs: int | None = None, max_mem: int | None = None) -> dict:
    """Compute integrity / accuracy / f1 / speaker_confusion for one conversation."""
    spk_a = sample.get("speaker_a") or raw.get("conversation", {}).get("speaker_a", "A")
    spk_b = sample.get("speaker_b") or raw.get("conversation", {}).get("speaker_b", "B")

    golden = build_golden(raw)
    dialogue = build_dialogue(raw)
    by_sess, scope = group_memories(sample.get("memory_dump"))

    def mems_for(sid):
        return by_sess.get(sid if scope == "session" else None, [])

    # ── integrity: iterate over the facts that should have been recorded ───
    obs_all = [o for sid in sorted(golden) for o in golden[sid]]
    if max_obs and len(obs_all) > max_obs:
        step = len(obs_all) / max_obs
        obs_all = [obs_all[int(i * step)] for i in range(max_obs)]

    integ_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(judge_integrity, mems_for(o["session"]), o["text"]): o
                for o in obs_all}
        for fut in as_completed(futs):
            o = futs[fut]
            try:
                score = fut.result()
            except Exception as e:
                _note_failure("integrity_worker", e); score = None
            integ_records.append({**o, "score": score})

    # Ask again about failures: vanished entirely, or attributed to the wrong speaker
    failed = [r for r in integ_records if r["score"] is not None and r["score"] < 2]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(judge_speaker, mems_for(r["session"]), r["text"],
                          r["owner"], spk_a, spk_b): r for r in failed}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                r["attribution"] = fut.result()
            except Exception as e:
                _note_failure("speaker_worker", e); r["attribution"] = None

    # ── accuracy: iterate over the memories the system extracted ───────────
    mem_items = []
    for sid, mems in by_sess.items():
        for m in mems:
            mem_items.append({"session": sid, "text": m})
    if max_mem and len(mem_items) > max_mem:
        step = len(mem_items) / max_mem
        mem_items = [mem_items[int(i * step)] for i in range(max_mem)]

    acc_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for it in mem_items:
            sid = it["session"]
            gold_txt = ([g["text"] for g in golden.get(sid, [])] if scope == "session"
                        else [g["text"] for sl in golden.values() for g in sl])
            dlg = (dialogue.get(sid) if scope == "session"
                   else "\n\n".join(dialogue[k] for k in sorted(dialogue)))
            futs[ex.submit(judge_accuracy, dlg, gold_txt, it["text"])] = it
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                score = fut.result()
            except Exception as e:
                _note_failure("accuracy_worker", e); score = None
            acc_records.append({**it, "score": score})

    return _aggregate(sample.get("sample_id"), scope, integ_records, acc_records)


def _aggregate(sample_id, scope, integ, acc) -> dict:
    iv = [r for r in integ if r["score"] is not None]
    av = [r for r in acc if r["score"] is not None]

    # Speaker confusion is counted separately and never as an integrity miss
    wrong_owner = sum(1 for r in iv if r.get("attribution") == "wrong_owner")
    strict_hit = sum(1 for r in iv if r["score"] == 2)

    recall = strict_hit / len(iv) if iv else None
    precision = (sum(1 for r in av if r["score"] == 2) / len(av)) if av else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else 0.0)

    return {
        "sample_id": sample_id,
        "scope": scope,
        "integrity_recall": recall,
        "integrity_num": len(iv),
        "integrity_unknown": len(integ) - len(iv),
        "integrity_score_dist": dict(Counter(r["score"] for r in iv)),
        "accuracy_precision": precision,
        "accuracy_num": len(av),
        "accuracy_unknown": len(acc) - len(av),
        "accuracy_score_dist": dict(Counter(r["score"] for r in av)),
        "extraction_f1": f1,
        # Specific to LoCoMo: the fact is present but attributed to the wrong speaker
        "speaker_confusion_num": wrong_owner,
        "speaker_confusion_ratio": wrong_owner / len(iv) if iv else None,
        "_integrity_records": integ,
        "_accuracy_records": acc,
    }


def run(run_name: str, frame: str | None = None, data_path: str = DATA_PATH,
        max_workers: int = 8, max_obs: int | None = None, max_mem: int | None = None,
        write: bool = True) -> dict:
    frame = frame or run_name.split("-")[0]
    result_dir = os.path.join(BASE_DIR, "results", run_name)
    results_file = os.path.join(result_dir, f"{frame}_locomo_results.jsonl")
    scores_file = os.path.join(result_dir, f"{frame}_locomo_scores.json")
    detail_file = os.path.join(result_dir, f"{frame}_locomo_extraction_detail.jsonl")

    with open(results_file, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    with open(data_path, encoding="utf-8") as f:
        raws = {s["sample_id"]: s for s in json.load(f)}

    with _LOCK:
        _FAILS.clear(); _WARNED.clear()

    usable = [s for s in samples if s.get("memory_dump")]
    if not usable:
        summary = {
            "skipped": "This run has no memory_dump: the backend did not export a "
                       "full store, or the eval script predates the dump feature. "
                       "Extraction metrics cannot be computed.",
            "samples_total": len(samples),
        }
        print(f"⚠️  {summary['skipped']}")
        if write:
            _merge_scores(scores_file, summary)
        return summary

    print(f"Extraction evaluation: {len(usable)}/{len(samples)} conversations have a memory_dump")
    per_sample, details = [], []
    for s in usable:
        raw = raws.get(s.get("sample_id"))
        if not raw:
            print(f"  skipping {s.get('sample_id')} (no matching source data in locomo10.json)")
            continue
        print(f"  → {s['sample_id']} …", flush=True)
        r = evaluate_sample(s, raw, max_workers=max_workers,
                            max_obs=max_obs, max_mem=max_mem)
        details.append({"sample_id": r["sample_id"],
                        "integrity": r.pop("_integrity_records"),
                        "accuracy": r.pop("_accuracy_records")})
        per_sample.append(r)
        print(f"     recall={r['integrity_recall']} precision={r['accuracy_precision']} "
              f"f1={r['extraction_f1']:.4f} scope={r['scope']}")

    summary = _pool(per_sample, len(samples))
    if write:
        _merge_scores(scores_file, summary)
        with open(detail_file, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\nPer-item records -> {detail_file}")
    print("\n=== LoCoMo extraction ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"},
                     indent=2, ensure_ascii=False))
    return summary


def _pool(per_sample: list[dict], n_total: int) -> dict:
    """Micro-average: denominated over all adjudicated samples, not averaged per conversation."""
    ih = inum = iunk = 0
    ah = anum = aunk = 0
    wrong = 0
    for r in per_sample:
        if r["integrity_recall"] is not None:
            ih += round(r["integrity_recall"] * r["integrity_num"])
        inum += r["integrity_num"]; iunk += r["integrity_unknown"]
        if r["accuracy_precision"] is not None:
            ah += round(r["accuracy_precision"] * r["accuracy_num"])
        anum += r["accuracy_num"]; aunk += r["accuracy_unknown"]
        wrong += r["speaker_confusion_num"]

    recall = ih / inum if inum else None
    precision = ah / anum if anum else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else 0.0)
    scopes = sorted({r["scope"] for r in per_sample})

    return {
        "memory_integrity_recall": recall,
        "memory_accuracy_precision": precision,
        "memory_extraction_f1": f1,
        "integrity_num": inum,
        "accuracy_num": anum,
        "integrity_unknown": iunk,
        "accuracy_unknown": aunk,
        # Unadjudicable share: the three metrics above already exclude these, and
        # without it their reliability cannot be assessed
        "unknown_ratio": (iunk + aunk) / (inum + anum + iunk + aunk)
                         if (inum + anum + iunk + aunk) else 0,
        "judge_failures": dict(_FAILS),
        "speaker_confusion_num": wrong,
        "speaker_confusion_ratio": wrong / inum if inum else None,
        "scope": scopes[0] if len(scopes) == 1 else scopes,
        "scope_note": ("session-scoped (turn provenance available), aligned with HaluMem and comparable"
                       if scopes == ["session"] else
                       "includes global scope (backend has no turn provenance); not directly comparable to session-scoped"),
        "conversations_evaluated": len(per_sample),
        "conversations_total": n_total,
        "judge_note": "Reuses HaluMem's official EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY / _ACCURACY; "
                      "strict = score 2, and score 1 (partial coverage) counts as a failure",
        "per_sample": per_sample,
    }


def _merge_scores(scores_file: str, block: dict) -> None:
    scores = {}
    if os.path.exists(scores_file):
        with open(scores_file, encoding="utf-8") as f:
            scores = json.load(f)
    scores["extraction"] = block
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"✅ wrote the \"extraction\" block of {scores_file}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory name under results/, e.g. mem0-default")
    ap.add_argument("--frame", default=None, help="filename prefix; defaults to the first segment of the run name")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-obs", type=int, default=None,
                    help="max observations judged per conversation (evenly sampled, to control cost)")
    ap.add_argument("--max-mem", type=int, default=None,
                    help="max system memories judged per conversation (evenly sampled, to control cost)")
    args = ap.parse_args()
    run(args.run, frame=args.frame, max_workers=args.max_workers,
        max_obs=args.max_obs, max_mem=args.max_mem)
