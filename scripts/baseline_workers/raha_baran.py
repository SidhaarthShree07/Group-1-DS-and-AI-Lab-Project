#!/usr/bin/env python3
"""Raha (every strategy of the package) and Baran on one table. Runs in the Raha environment (see run_baselines_m5.py).

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

Input JSON: dirty, name, label_order {"<budget>": [rows in labelling order]}, masked_clean {"<budget>": path}, oracle
(clean file, used only by the self-sampling mode, one row at a time), jobs [{budget, seed, mode: "fixed"|"own"}].
A masked clean file equals the dirty file outside the labelled rows, so no other truth can leak into Raha or Baran.
"""
import csv
import json
import random
import signal
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import raha  # noqa: E402
import raha.correction  # noqa: E402

WORKERS = 12


class Detect(raha.Detection):
    order = None  # rows to label, in order; None = Raha samples its own rows

    def sample_tuple(self, d):
        super().sample_tuple(d)
        if self.order is not None:
            d.sampled_tuple = self.order[len(d.labeled_tuples)]

    def label_with_ground_truth(self, d):
        if self.order is not None:
            return super().label_with_ground_truth(d)
        row = int(d.sampled_tuple)  # ask the oracle for this one row only
        with open(d.oracle_path, encoding="utf-8", newline="") as f:
            for i, rec in enumerate(csv.reader(f)):
                if i == row + 1:
                    truth = [d.value_normalizer(v) for v in rec]
                    break
        d.labeled_tuples[row] = 1
        for j in range(d.dataframe.shape[1]):
            d.labeled_cells[(row, j)] = [int(d.dataframe.iloc[row, j] != truth[j]), truth[j]]


class Correct(raha.Correction):
    order = None

    def sample_tuple(self, d):
        d.sampled_tuple = self.order[len(d.labeled_tuples)]


def main():
    cfg = json.load(open(sys.argv[1]))
    any_clean = next(iter(cfg["masked_clean"].values()))
    random.seed(0)
    np.random.seed(0)
    t0 = time.time()
    app = Detect()
    app.SAVE_RESULTS = False
    base = app.initialize_dataset({"name": cfg["name"], "path": cfg["dirty"], "clean_path": any_clean})
    app.run_strategies(base)
    profiles, strategy_seconds = base.strategy_profiles, time.time() - t0
    by_algorithm = {}
    for p in profiles:
        a = json.loads(p["name"])[0]
        n = by_algorithm.setdefault(a, [0, 0])
        n[0] += 1
        n[1] += bool(p["output"])
    out, clusters = [], {}

    def too_long(*_):
        raise TimeoutError("Baran time limit")

    signal.signal(signal.SIGALRM, too_long)
    for job in cfg["jobs"]:
        k, seed, own = job["budget"], job["seed"], job["mode"] == "own"
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        rec = {**job, "strategies": by_algorithm, "strategy_seconds": round(strategy_seconds, 1)}
        try:
            app = Detect()
            app.SAVE_RESULTS, app.LABELING_BUDGET = False, k
            app.order = None if own else cfg["label_order"][str(k)]
            d = app.initialize_dataset({"name": cfg["name"], "path": cfg["dirty"],
                                        "clean_path": any_clean if own else cfg["masked_clean"][str(k)]})
            d.oracle_path = cfg.get("oracle")
            d.strategy_profiles = profiles
            if k not in clusters:  # features and clusters do not depend on the labels or the seed
                app.generate_features(d)
                app.build_clusters(d)
                clusters[k] = (d.column_features, d.clusters_k_j_c_ce, d.cells_clusters_k_j_ce)
            d.column_features, d.clusters_k_j_c_ce, d.cells_clusters_k_j_ce = clusters[k]
            while len(d.labeled_tuples) < k:
                app.sample_tuple(d)
                app.label_with_ground_truth(d)
            app.propagate_labels(d)
            app.predict_labels(d)
            cols = list(d.dataframe.columns)
            rec["flags"] = sorted([int(i), cols[j]] for i, j in d.detected_cells)
            rec["labelled_rows"] = sorted(int(r) for r in d.labeled_tuples)
            rec["detect_seconds"] = round(time.time() - t0, 1)
            if not own and not cfg.get("skip_baran"):
                t1 = time.time()
                signal.alarm(int(cfg.get("baran_seconds", 0)))  # 0 = no limit; the detection result is kept
                fresh = raha.dataset.Dataset({"name": cfg["name"], "path": cfg["dirty"], "clean_path": cfg["masked_clean"][str(k)]})
                fresh.detected_cells = dict(d.detected_cells)
                fix = Correct()
                fix.SAVE_RESULTS, fix.LABELING_BUDGET, fix.NUM_WORKERS, fix.order = False, k, WORKERS, app.order
                try:
                    corrected = fix.run(fresh)
                    rec["repairs"] = sorted([int(i), cols[j], str(v)] for (i, j), v in corrected.items())
                except TimeoutError:
                    rec["baran_status"] = f"failed: Baran over {cfg['baran_seconds']} s"
                signal.alarm(0)
                rec["baran_seconds"] = round(time.time() - t1, 1)
            rec["status"] = "completed"
        except Exception as e:  # reported, never dropped
            rec["status"] = f"failed: {type(e).__name__}: {e}"
        out.append(rec)
        print(cfg["name"], k, seed, job["mode"], rec["status"], flush=True)
        json.dump(out, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
