"""
Phase-aware LLM cost / latency meter.

At import time this patches openai's Completions.create at the class level, so
every OpenAI client in the process (Mem0, LightMem, A-MEM, llms.py) feeds the
same counter automatically, with no per-adapter changes to call sites.

It measures four things, and does so **per phase**:

    ingest  memory-write phase (the extraction/update the backend performs as
            each turn or session is fed in)
    qa      answering phase (retrieval + generation)
    other   everything else (judge, probes, warm-up); never counted in the two above

The phase split is the point. The earlier version kept a single total and could
not answer questions like "how many LLM calls does writing one session take".
It also matters because run.py performs extraction and evaluation in the same
process, so without bucketing the judge's tokens would leak into the totals.

Each bucket records:
    calls              number of LLM calls
    prompt/completion  token split (the earlier version had only a total)
    llm_seconds        summed pure API wall time (overlaps under concurrency, so
                       it is not the same as elapsed time)
    units              how many units this phase processed (sessions, questions)
    unit_seconds       per-unit end-to-end durations, kept as a list for medians

Usage from an adapter:

    import token_tracker as _tk

    _tk.reset()                                  # start of a new run
    with _tk.unit("ingest"):                     # one turn or one session
        mem.add(...)
    with _tk.unit("qa"):                         # one question
        mems = mem.search(q); llm_request(...)
    _tk.save(save_path, frame)

`unit()` does two things at once: it switches phase and times that unit
end-to-end. To switch phase without counting a unit, use
`with _tk.phase("qa"):`. Both are thread-local and safe to nest.

Scripts that run the judge or the probes need do nothing: unmarked calls default
to the "other" bucket.
"""
import json
import os
import statistics
import threading
import time
from contextlib import contextmanager

from openai.resources.chat.completions import Completions as _Completions

PHASES = ("ingest", "qa", "other")


def _new_bucket():
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "llm_seconds": 0.0,
            "units": 0, "unit_seconds": [], "errors": 0}


class _Meter:
    def __init__(self):
        self._lock = threading.Lock()
        self._b = {p: _new_bucket() for p in PHASES}
        self._local = threading.local()      # thread-local phase stack
        self._default = "other"              # bucket for unmarked calls
        self._dstack = []                    # global default stack (for threads the backend spawns)

    # ── Phase resolution ──────────────────────────────────────────────────
    @property
    def _stack(self):
        s = getattr(self._local, "stack", None)
        if s is None:
            s = self._local.stack = []
        return s

    def current_phase(self) -> str:
        """Thread-local marking wins; an unmarked worker thread falls back to the global default.

        ThreadPoolExecutor workers do not inherit the main thread's context, so
        when an adapter marks a whole section with set_default_phase() on the
        main thread and then hands work to a thread pool, the workers are still
        attributed correctly.
        """
        st = self._stack
        return st[-1] if st else self._default

    def set_default_phase(self, phase: str):
        assert phase in PHASES, phase
        self._default = phase

    def push_default(self, phase: str):
        """Also push the global default to this phase.

        This is necessary, not defensive. Mem0's add() spawns its own
        ThreadPoolExecutor for extraction, and those workers do not inherit the
        main thread's thread-local marking, so the entire extraction cost would
        land in "other". A LoCoMo smoke run measured exactly that: ingest 0 calls,
        other 90 calls. An adapter's ingest and qa phases are sequential, so the
        global default cannot cause interference, and threads that mark themselves
        still take precedence.
        """
        with self._lock:
            self._dstack.append(phase)
            self._default = phase

    def pop_default(self):
        with self._lock:
            if self._dstack:
                self._dstack.pop()
            self._default = self._dstack[-1] if self._dstack else "other"

    # ── Recording ─────────────────────────────────────────────────────────
    def record(self, prompt_tokens=0, completion_tokens=0, total_tokens=None,
               seconds=0.0, phase=None, calls=1):
        p = phase or self.current_phase()
        if p not in PHASES:
            p = "other"
        tot = total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
        with self._lock:
            b = self._b[p]
            b["calls"] += calls
            b["prompt_tokens"] += prompt_tokens or 0
            b["completion_tokens"] += completion_tokens or 0
            b["total_tokens"] += tot or 0
            b["llm_seconds"] += seconds or 0.0

    def record_error(self, phase=None):
        p = phase or self.current_phase()
        with self._lock:
            self._b[p if p in PHASES else "other"]["errors"] += 1

    def record_unit(self, phase: str, seconds: float):
        with self._lock:
            b = self._b[phase if phase in PHASES else "other"]
            b["units"] += 1
            b["unit_seconds"].append(round(seconds, 4))

    # ── Legacy API (used by 20+ files, cannot be removed) ─────────────────
    def add(self, tokens: int):
        self.record(total_tokens=tokens, prompt_tokens=0, completion_tokens=0)

    @property
    def total(self) -> int:
        with self._lock:
            return sum(b["total_tokens"] for b in self._b.values())

    def reset(self):
        with self._lock:
            self._b = {p: _new_bucket() for p in PHASES}
            self._dstack = []
        self._default = "other"

    # ── Export ────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        with self._lock:
            b = {p: dict(v) for p, v in self._b.items()}

        out = {"total_tokens": sum(v["total_tokens"] for v in b.values())}
        for p in PHASES:
            v = b[p]
            secs = v.pop("unit_seconds")
            n = v["units"]
            d = dict(v)
            # Per-unit averages: how many calls and tokens one session costs.
            d["calls_per_unit"]  = round(v["calls"] / n, 3) if n else None
            d["tokens_per_unit"] = round(v["total_tokens"] / n, 1) if n else None
            d["prompt_tokens_per_unit"] = round(v["prompt_tokens"] / n, 1) if n else None
            d["completion_tokens_per_unit"] = round(v["completion_tokens"] / n, 1) if n else None
            # Latency: mean and median (the tail is heavy, so both are needed)
            d["sec_per_unit_mean"]   = round(statistics.mean(secs), 3) if secs else None
            d["sec_per_unit_median"] = round(statistics.median(secs), 3) if secs else None
            d["sec_per_unit_p90"]    = (round(sorted(secs)[int(len(secs) * 0.9)], 3)
                                        if len(secs) >= 10 else None)
            d["llm_seconds"] = round(v["llm_seconds"], 2)
            out[p] = d
        return out


meter = _Meter()
tracker = meter          # legacy alias (eval_*.py refers to _tk.tracker)


# ── Public phase markers ──────────────────────────────────────────────────

@contextmanager
def phase(name: str):
    """Switch phase only; do not count a unit."""
    name = name if name in PHASES else "other"
    st = meter._stack
    st.append(name)
    meter.push_default(name)        # worker threads the backend spawns land here too
    try:
        yield
    finally:
        st.pop()
        meter.pop_default()


@contextmanager
def unit(name: str):
    """Switch phase and time this block as one unit (a turn/session, or one QA)."""
    name = name if name in PHASES else "other"
    st = meter._stack
    st.append(name)
    meter.push_default(name)
    t0 = time.time()
    try:
        yield
    finally:
        st.pop()
        meter.pop_default()
        meter.record_unit(name, time.time() - t0)


def set_default_phase(name: str):
    """For thread pools: the main thread marks a section and unmarked workers land here."""
    meter.set_default_phase(name)


def record_external(prompt_tokens=0, completion_tokens=0, total_tokens=None,
                    seconds=0.0, phase=None, calls=1):
    """For server-side LLMs (Letta): report manually from response.usage."""
    meter.record(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                 total_tokens=total_tokens, seconds=seconds, phase=phase, calls=calls)


def reset():
    meter.reset()


# ── Patch OpenAI Completions.create (applied once at import) ─────────────
_orig_create = _Completions.create
_guard = threading.local()      # prevents litellm -> openai double counting


def _usage_of(response):
    u = getattr(response, "usage", None)
    if not u:
        return None
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    return pt, ct, (getattr(u, "total_tokens", None) or (pt + ct))


def _patched_create(self, *args, **kwargs):
    # litellm's openai provider routes through here; if the outer layer already
    # counted the call, do not count it again
    if getattr(_guard, "in_litellm", False):
        return _orig_create(self, *args, **kwargs)
    t0 = time.time()
    try:
        response = _orig_create(self, *args, **kwargs)
    except Exception:
        meter.record_error()
        raise
    dt = time.time() - t0
    u = _usage_of(response)
    if u:
        meter.record(prompt_tokens=u[0], completion_tokens=u[1], total_tokens=u[2], seconds=dt)
    else:
        # Record calls and timing even when usage is absent, otherwise
        # calls_per_unit is underestimated
        meter.record(seconds=dt)
    return response


_Completions.create = _patched_create


# ── litellm (A-MEM's Ollama backend) ─────────────────────────────────────

def patch_litellm():
    """Called from eval_amem.py once litellm is importable."""
    try:
        import litellm as _litellm
    except ImportError:
        return                      # litellm not installed; this backend does not need it
    if getattr(_litellm, "_tk_patched", False):
        return
    _orig_completion = _litellm.completion

    def _tracked_completion(*args, **kwargs):
        _guard.in_litellm = True    # stand down the inner openai patch
        t0 = time.time()
        try:
            response = _orig_completion(*args, **kwargs)
        except Exception:
            meter.record_error()
            raise
        finally:
            _guard.in_litellm = False
        u = _usage_of(response)
        if u:
            meter.record(prompt_tokens=u[0], completion_tokens=u[1], total_tokens=u[2],
                         seconds=time.time() - t0)
        else:
            meter.record(seconds=time.time() - t0)
        return response

    _litellm.completion = _tracked_completion
    _litellm._tk_patched = True


# ── Saving ────────────────────────────────────────────────────────────────

def save(result_dir: str, frame: str, extra: dict | None = None):
    """Write {frame}_token_usage.json."""
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"{frame}_token_usage.json")
    d = meter.to_dict()
    if extra:
        d.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

    ing, qa = d["ingest"], d["qa"]
    print(f"✅ Cost/latency → {path}")
    print(f"   ingest : {ing['units'] or 0} units | {ing['calls']} calls "
          f"({ing['calls_per_unit']}/unit) | {ing['total_tokens']:,} tok "
          f"({ing['tokens_per_unit']}/unit) | {ing['sec_per_unit_median']}s/unit (median)")
    print(f"   qa     : {qa['units'] or 0} questions | {qa['calls']} calls "
          f"({qa['calls_per_unit']}/question) | {qa['total_tokens']:,} tok "
          f"({qa['tokens_per_unit']}/question) | {qa['sec_per_unit_median']}s/question (median)")
    if d["other"]["calls"]:
        print(f"   other  : {d['other']['calls']} calls / {d['other']['total_tokens']:,} tok "
              f"(judge, warm-up, etc.; not counted in the two buckets above)")
    return d
