"""The noise floor a verdict is judged against, and the verdict.

Moved out of `report.py`, which re-exports every name here.
"""

from __future__ import annotations

import compare
import regression
from regression import SIGMA, noise_band
from report_data import UPLOAD_ROWS, Run


def noise_dataset_mismatch(runs: list) -> str | None:
    """Whether the floor on disk describes these rows' dataset.

    Run-to-run spread is a property of a corpus at a dimension, not of the
    harness alone. The only floor that existed was measured on sift1m (d=128)
    on, and it was applied to whatever ran next: dbpedia-openai-100K
    rows reading 0.94x, 1.00x and 1.01x — exactly the band where the verdict
    decides whether a number is a result at all — were judged against a spread
    measured on another corpus at a twelfth the dimension.

    An *unstamped* floor stays usable and is labelled by `noise_provenance`,
    because refusing it would silently drop banding for every row measured
    before floors carried a dataset. A floor stamped with a *different* dataset
    is refused: that is not a weaker answer, it is an answer to another question.
    """
    floor_ds = noise_meta(runs).get("dataset")
    if not floor_ds:
        return None
    row_ds = next(((r.meta.get("dataset") or {}).get("name") for r in runs
                   if (r.meta.get("dataset") or {}).get("name")), None)
    if row_ds and row_ds != floor_ds:
        return (f"The noise floor was measured on {floor_ds} and these rows are "
                f"{row_ds}; run-to-run spread does not carry between corpora, so "
                f"no row here is banded.")
    return None



def load_noise(runs: list | None = None) -> dict[str, float]:
    """The measured run-to-run spread, per row, from `bench/results/noise.json`.

    Empty when the floor names a different dataset than `runs`, so every row
    falls to `ratio_verdict`'s "no noise floor measured for this row" instead of
    borrowing another corpus's spread.

    `regression.py --measure-noise` writes it from repeated identical passes
    over one unchanged server, contaminated repetitions discarded rather than
    averaged in; how many of each is in the file (`noise_meta`), not here.
    """
    if runs is not None and (noise_dataset_mismatch(runs) or noise_engine_mismatch(runs)
                             or noise_env_mismatch(runs)):
        return {}
    return noise_meta(runs).get("rsd", {}) or {}



def noise_env_mismatch(runs: list) -> str | None:
    """Whether the floor was measured in the environment it would band.

    `compare.env_matches`'s rule, which this page did not apply: an SMT-off
    floor against an SMT-on run printed a bare ratio in the table and "clears
    the measured noise floor" under it, from the floor `compare` had refused.
    """
    if len(runs) < 2:
        return None
    meta = noise_meta(runs)
    if not meta.get("rsd"):
        return None
    if compare.env_matches(meta, runs[0].label, runs[1].label):
        return None
    return ("The noise floor was measured under another environment hash than "
            "these rows (SMT, governor, isolation: findings 46), so no row here "
            "is banded.")



def noise_engine_mismatch(runs: list) -> str | None:
    """Whether the floor on disk describes both engines whose ratio it bands.

    The dataset rule above keeps an *unstamped* floor usable, because it might
    have been measured on this corpus. This one does not, because it cannot
    have been measured on both engines: `measure_noise` reads one label's
    repetition directories, so a floor is one engine's spread and a file that
    omits `arms` is one whose engine was simply never written down.

    Findings 38: the floor here is six passes of strawmANN and gave W3 an RSD
    of 0.46%, while Qdrant's own spread on that row is 13.16% over twelve runs.
    Every ratio banded with it assumed Qdrant was no noisier than strawmANN.
    """
    if len(runs) < 2:
        return None
    meta = noise_meta(runs)
    if not meta.get("rsd"):
        return None
    arms = meta.get("arms")
    engines = {(r.meta.get("engine_comm") or r.label) for r in runs}
    if arms and engines.issubset(set(arms)):
        return None
    whose = f"was measured on {', '.join(arms)}" if arms else "does not say which engine it was measured on"
    return (f"The noise floor {whose} and these rows are "
            f"{', '.join(sorted(engines))}; run-to-run spread does not carry between "
            f"engines, so no row here is banded.")



def noise_meta(runs: list | None = None) -> dict:
    """The floor that describes these runs, whole, or `{}`.

    `regression.floor_for` prefers the folded per-label `noise.json` a
    `--reps N` run writes for itself over the global one, which on this host is
    a sift1m fold measured on strawmANN alone in another session. Called
    without `runs` it is the global file, as before: the callers that have the
    runs pass them, and the ones that do not are asking about the file.
    """
    return regression.floor_for([r.label for r in runs] if runs else [])



def noise_provenance(runs: list[Run]) -> dict:
    """How the floor was measured, said from the file rather than from memory.

    The page used to say "six identical passes, one discarded as contaminated"
    in prose, while `noise.json` on the same host said `discarded: []`. And a
    floor measured under another harness stamp than these rows' is noted, not
    refused: it is still a spread of this host, and not one of these rows.
    """
    m = noise_meta(runs)
    if not m:
        return {}
    reps = m.get("reps") or {}
    n = m.get("n_dirs") or (max(reps.values()) + len(m.get("discarded") or [])
                            if reps else 0)
    out = {"passes": n, "discarded": len(m.get("discarded") or []),
           "kept": max(reps.values()) if reps else 0, "source": m.get("source", ""),
           # Which arms the spread was measured on. Empty for the borrowed
           # global floor, which is strawmANN's alone; a `--reps N` run folds
           # one per engine and the sentence about assuming Qdrant is no
           # noisier stops being true of it.
           "arms": m.get("arms") or [], "stamp_note": ""}
    # Which of *these* rows the floor speaks for. Silence on a row with no
    # floor already reads as "judged and fine" in `ratio_verdict`; saying it
    # once, in aggregate, is what stops a reader assuming the whole table was
    # judged. The ingest rows are the ones this matters most for: the floor is
    # built from search passes over an already-built graph, so no build time
    # on the page has a spread behind it, and the section that reports build
    # time carries no verdicts at all.
    rsd = m.get("rsd") or {}
    ids = list(dict.fromkeys(r["id"] for run in runs for r in run.rows))
    out["covered"] = sum(1 for i in ids if i in rsd)
    out["rows"] = len(ids)
    out["uncovered_ingest"] = [i for i in ids if i not in rsd and i in UPLOAD_ROWS]
    ds_note = noise_dataset_mismatch(runs) or noise_engine_mismatch(runs)
    if ds_note:
        out["stamp_note"] = ds_note
        return out
    fh = m.get("harness_hash")
    stamps = {r.label: (r.meta.get("harness") or None) for r in runs}
    if fh is None:
        out["stamp_note"] = ("The floor carries no harness stamp (measured before floors "
                             "were stamped), so it may not describe these rows' "
                             "configuration.")
    elif fh == "mixed":
        out["stamp_note"] = ("The floor was measured over repetitions with differing "
                             "harness stamps; its spread is not one configuration's.")
    else:
        from workloads import stamp_hash
        off = [lbl for lbl, st in stamps.items() if st and stamp_hash(st) != fh]
        if off:
            out["stamp_note"] = (f"The floor was measured under a different harness "
                                 f"stamp than {' and '.join(off)}, so its spread may "
                                 f"not describe these rows.")
    return out


#: Half-width past which a noise band stops being a verdict and starts being
#: an admission that the row did not discriminate. 25% is chosen, not measured:
#: it is comfortably wider than any real effect this table reports (the largest
#: is W5 at 4.8x, the smallest called difference is 2%) and comfortably narrower
#: than the ±27% and ±64% bands a single unlucky pass produced.
UNDISCRIMINATING_BAND = 0.25



def reps_of(runs) -> int | None:
    """How many passes each arm folded, when they agree. `None` otherwise."""
    ns = {(r.meta or {}).get("reps") for r in runs}
    return ns.pop() if len(ns) == 1 and isinstance(next(iter(ns), None), int) else None



def ratio_verdict(wid: str, ratio: float | None, noise: dict[str, float],
                  reps: int | None = None) -> str:
    """Whether a ratio clears the noise floor, or is indistinguishable from 1.

    Two engines, so the spreads add: a 3σ threshold on the ratio is
    `3 · √2 · rsd` either side of 1.0 (`regression.noise_band(rsd, arms=2)`,
    the one definition of the threshold). The floor was measured on strawmann
    alone, on this host, so using it for both sides assumes Qdrant is no
    noisier, which is an assumption and is labelled as one.

    A ratio inside that band is not a small difference, it is *no measured
    difference*, and the entire point of having a floor is that the table says
    so instead of leaving a reader to treat 1.05x as a result.
    """
    if ratio is None:
        return ""
    if wid not in noise:
        # Silence here reads as "judged and fine": W4 carried "clears the ±19%
        # noise floor" while W10-ef32 carried a bare 2.11x, and nothing told a
        # reader that the second had never been measured for spread at all.
        # `noise.json` does not cover every row, and the count moves as
        # rows are added; `noise_provenance` reports the coverage of
        # *these* rows rather than a number frozen in a comment.
        return "no noise floor measured for this row"
    band = noise_band(noise[wid], arms=2)
    # `noise_band` returns `max(3*sqrt(2)*rsd, MIN_EFFECT)`, so on a quiet row
    # the number printed is the policy minimum and not this host's measured
    # spread. Calling both of them "the noise floor" told a reader that 2% had
    # been measured here; it is the smallest effect this project will report,
    # and the measured spread is usually well under it.
    measured = SIGMA * (2 ** 0.5) * noise[wid]
    what = ("minimum reportable effect" if band > measured + 1e-12
            else "measured noise floor")
    if abs(ratio - 1.0) > band:
        return f"clears the ±{band * 100:.0f}% {what}"
    # A band this wide is not a verdict, it is an admission.
    #
    # "within the ±64% band: no measured difference, not a small one" is the
    # right sentence for ±5% and a false comfort at ±64%: nothing this row
    # could plausibly have measured would have cleared it, so the row did not
    # discriminate between "identical" and "one engine twice the other". The
    # sift1m run printed exactly that, from a three-sample rsd where
    # Qdrant's W3 read 1,368, 2,063 and 1,580 — one unlucky pass, and an rsd
    # estimated from three points is itself a very noisy number.
    #
    # Past this width the honest report is that the row needs more passes.
    if band >= UNDISCRIMINATING_BAND:
        return (f"too noisy to judge: ±{band * 100:.0f}% over {reps or '?'} passes, "
                f"wider than any difference this row could show — needs more passes")
    return f"inconclusive: within the ±{band * 100:.0f}% {what}"
