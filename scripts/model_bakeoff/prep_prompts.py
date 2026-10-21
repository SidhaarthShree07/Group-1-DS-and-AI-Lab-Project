"""Build the bake-off test set: sampled cells from 5 Raha benchmark tables + identical prompts for every model.

Ground truth = Raha's rule: a cell is an error when norm(dirty) != norm(clean).
Column profile and dependency evidence are computed from the DIRTY table only (no clean values leak into prompts).
"""
import collections, html, json, os, random, re, sys
import numpy as np
import pandas as pd

E = os.environ.get("E", "workdir")  # working folder with data/, results/ and weights/
RAHA = f"{E}/data/raha_src/datasets"
SEED = int(os.environ.get("SEED", 42))
OUT = f"{E}/data/bakeoff_prompts.jsonl" if SEED == 42 else f"{E}/data/bakeoff_prompts_s{SEED}.jsonl"
N_ERR, N_OK = 100, 200
TABLES = {  # name: (raha folder, annotation file)
    "hospital": ("hospital", "hospital_annotation.jsonl"),
    "flights": ("flights", "flight_annotation.jsonl"),
    "beers": ("beers", "beers_annotation.jsonl"),
    "rayyan": ("rayyan", "rayyan_annotation.jsonl"),
    "movies": ("movies_1", "movie_annotation.jsonl"),
}
INDEX_NAMES = {"index", "tuple_id", "row_id", "unnamed: 0"}
MISSING = {"", "nan", "null", "none", "n/a", "na", "-", "?", "not available"}


def norm(v):
    v = html.unescape(v)
    v = re.sub("[\t\n ]+", " ", v)
    return v.strip("\t\n ")


def ckey(c):
    return re.sub(r"[^a-z0-9]", "", c.lower())


def sig(v):
    out, prev, run = [], None, 0
    for ch in v:
        k = "A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else ch
        if k == prev:
            run += 1
        else:
            if prev is not None:
                out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
            prev, run = k, 1
    if prev is not None:
        out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
    return "".join(out)


def to_float(v):
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


class Table:
    def __init__(self, name):
        folder, annfile = TABLES[name]
        self.name = name
        clean = pd.read_csv(f"{RAHA}/{folder}/clean.csv", dtype=str, keep_default_na=False, low_memory=False)
        dirty = pd.read_csv(f"{RAHA}/{folder}/dirty.csv", dtype=str, keep_default_na=False, low_memory=False)
        dirty.columns = clean.columns            # beers/movies headers differ between files; align by position
        self.cols = [c for c in clean.columns if c.lower() not in INDEX_NAMES]
        self.clean = clean[self.cols].map(norm)
        self.dirty = dirty[self.cols].map(norm)
        self.raw_dirty = dirty[self.cols]
        self.n = len(self.dirty)
        self.err = (self.clean != self.dirty)
        self.ann = {}
        for line in open(f"{E}/data/{annfile}"):
            try:
                d = json.loads(line)
                self.ann[(int(d["row_id"]), ckey(d["column"]))] = d["error_type"]
            except Exception:
                pass
        self._prof, self._idx, self._fd = {}, {}, None

    # ---- profile
    def profile(self, col):
        if col in self._prof:
            return self._prof[col]
        s = self.dirty[col]
        vc = s.value_counts()
        nums = [to_float(v) for v in s if v.lower() not in MISSING]
        nums_ok = [x for x in nums if x is not None]
        is_num = len(nums) > 0 and len(nums_ok) / max(1, len(nums)) >= 0.9 and len(nums_ok) >= 10
        sigs = collections.Counter(sig(v) for v in s if v.lower() not in MISSING)
        p = {"nd": int(s.nunique()), "vc": vc, "miss": float(np.mean([v.lower() in MISSING for v in s])),
             "num": is_num, "sigs": sigs.most_common(3), "nsig": sum(sigs.values())}
        if is_num:
            a = np.array(nums_ok)
            p.update(med=float(np.median(a)), p1=float(np.percentile(a, 1)), p99=float(np.percentile(a, 99)), mn=float(a.min()), mx=float(a.max()))
        self._prof[col] = p
        return p

    def idx(self, col):
        if col not in self._idx:
            self._idx[col] = self.dirty.groupby(col).indices
        return self._idx[col]

    def fd(self):
        """strength[(A,B)] = share of rows where B equals the majority B of its A group (A -> B)."""
        if self._fd is None:
            self._fd = {}
            for A in self.cols:
                if self.dirty[A].nunique() > 0.6 * self.n:
                    continue
                for B in self.cols:
                    if A == B:
                        continue
                    g = self.dirty.groupby([A, B]).size().reset_index(name="k")
                    self._fd[(A, B)] = float(g.groupby(A)["k"].max().sum() / self.n)
        return self._fd

    def short(self, v, m=40):
        return v if len(v) <= m else v[: m - 1] + "…"

    # ---- prompt
    def prompt(self, i, col):
        val = self.dirty.at[i, col]
        P = self.profile(col)
        row = "\n".join(f"- {c}: {self.short(self.dirty.at[i, c])!r}" + ("   <-- SUSPECT" if c == col else "") for c in self.cols)
        prof = [f"type: {'numeric' if P['num'] else 'text/categorical'}, {P['nd']} distinct values, {P['miss']*100:.1f}% missing"]
        top = ", ".join(f"{self.short(k, 24)!r} ({v})" for k, v in P["vc"].head(5).items())
        prof.append(f"most frequent values: {top}")
        if P["sigs"]:
            prof.append("common patterns: " + ", ".join(f"{s} ({c/max(1,P['nsig'])*100:.0f}%)" for s, c in P["sigs"]))
        if P["num"]:
            prof.append(f"median {P['med']:g}, 1st-99th percentile {P['p1']:g} to {P['p99']:g}, min {P['mn']:g}, max {P['mx']:g}")
        prof.append(f"this value's pattern: {sig(val)}; occurs {int(P['vc'].get(val, 0))} time(s) in the column")
        fd = self.fd()
        dets = sorted([(s, A) for (A, B), s in fd.items() if B == col and s >= 0.9], reverse=True)[:2]
        evid, best = [], None
        for s, A in dets:
            rows = self.idx(A).get(self.dirty.at[i, A], [])
            if len(rows) >= 2:
                cnt = collections.Counter(self.dirty.at[r, col] for r in rows)
                mv, k = cnt.most_common(1)[0]
                evid.append(f"{A}={self.short(self.dirty.at[i, A])!r} -> {col} is {self.short(mv)!r} in {k} of {len(rows)} rows (this row has {self.short(val)!r})")
                best = best or (A, rows)
        deps = sorted([(s, B) for (A, B), s in fd.items() if A == col and s >= 0.9], reverse=True)[:2]
        rows = self.idx(col).get(val, [])
        if len(rows) >= 2:
            for s, B in deps:
                cnt = collections.Counter(self.dirty.at[r, B] for r in rows)
                mv, k = cnt.most_common(1)[0]
                if self.dirty.at[i, B] != mv:
                    evid.append(f"{col}={self.short(val)!r} -> {B} is {self.short(mv)!r} in {k} of {len(rows)} rows (this row has {self.short(self.dirty.at[i, B])!r})")
        neigh = []
        if best:
            A, rows = best
            others = [r for r in rows if r != i][:3]
            show = [A, col] + [c for c in self.cols if c not in (A, col)][:3]
            for r in others:
                neigh.append("  " + "; ".join(f"{c}={self.short(self.dirty.at[r, c], 24)!r}" for c in show))
        return (
            "Task: decide whether the marked cell is an error. Use the row, the column profile and the dependency evidence.\n"
            "Error types: missing (empty, null or placeholder value), pattern (wrong format, typo or inconsistent representation), "
            "rule (violates a dependency with other columns), outlier (implausible value for this column).\n\n"
            f"Table: {self.name} | Column: {col} | Cell value: {val!r}\n\nRow:\n{row}\n\n"
            "Column profile:\n" + "\n".join("- " + x for x in prof) + "\n\n"
            "Dependency evidence:\n" + ("\n".join("- " + x for x in evid) if evid else "- no strong dependency found") + "\n\n"
            "Similar rows (same value of the strongest determinant):\n" + ("\n".join(neigh) if neigh else "  none") + "\n\n"
            'Answer with JSON only, in this format: {"is_error": true or false, "error_type": "missing|pattern|rule|outlier|none", '
            '"evidence": "<one short sentence>", "repair": "<corrected value, or null if not an error>"}'
        )


def main():
    rng = random.Random(SEED)
    recs = []
    for name in TABLES:
        T = Table(name)
        err_cells = collections.defaultdict(list)
        for j, c in enumerate(T.cols):
            for i in np.flatnonzero(T.err[c].values):
                err_cells[c].append((int(i), c))
        ecols = [c for c in T.cols if err_cells[c]]
        pool = []
        for c in ecols:
            rng.shuffle(err_cells[c]); pool += err_cells[c][:30]
        rng.shuffle(pool)
        chosen_err = pool[:N_ERR]
        ok_pool = [(i, c) for c in ecols for i in range(T.n) if not T.err.at[i, c]]
        chosen_ok = rng.sample(ok_pool, min(N_OK, len(ok_pool)))
        for is_err, cells in ((1, chosen_err), (0, chosen_ok)):
            for i, c in cells:
                recs.append({
                    "id": f"{name}:{i}:{c}", "table": name, "row": i, "column": c, "is_error": is_err,
                    "dirty": T.dirty.at[i, c], "clean": T.clean.at[i, c],
                    "gold_type": T.ann.get((i, ckey(c))) if is_err else None,
                    "prompt": T.prompt(i, c),
                })
        print(name, "rows", T.n, "cols", len(T.cols), "error cells", int(T.err.values.sum()), "error cols", len(ecols), "sampled", len(chosen_err), "+", len(chosen_ok), flush=True)
    rng.shuffle(recs)
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    lens = [len(r["prompt"]) for r in recs]
    print("wrote", OUT, len(recs), "prompts; chars min/median/max", min(lens), int(np.median(lens)), max(lens))
    print("gold types:", collections.Counter(r["gold_type"] for r in recs if r["is_error"]))


if __name__ == "__main__":
    main()
