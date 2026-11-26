#!/usr/bin/env python3
"""dBoost with every given setting on one dirty table (Raha environment). No clean data is read.

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)
"""
import csv
import json
import multiprocessing
import os
import sys
import tempfile
import time

import pandas as pd
import raha.tools.dBoost.dboost.imported_dboost as db


def run(args):
    dirty, setting = args
    t0, work = time.time(), tempfile.mkdtemp()
    path = os.path.join(work, "t.csv")
    pd.read_csv(dirty, dtype=str, keep_default_na=False).to_csv(path, index=False)
    rec = {"setting": setting}
    try:
        # a trailing "s=<x>" sets dBoost's --statistical value (0.5 as in Raha when absent)
        stat = setting[-1][2:] if setting[-1].startswith("s=") else "0.5"
        model = [x for x in setting if not x.startswith("s=")]
        db.run(["-F", ",", "--statistical", stat, "--" + model[0]] + model[1:] + [path])
        out = path + "-dboost_output.csv"
        rec["status"] = "completed" if os.path.exists(out) else "failed: no output file"
        cells = set()
        if os.path.exists(out):
            with open(out, encoding="utf-8", newline="") as f:
                for r in csv.reader(f):
                    if int(r[0].strip()) > 0:
                        cells.add((int(r[0].strip()) - 1, int(r[1].strip())))
        rec["cells"] = sorted(cells)
    except BaseException as e:  # dBoost may call sys.exit
        rec.update(status=f"failed: {type(e).__name__}: {e}", cells=[])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    with multiprocessing.Pool(min(34, len(cfg["grid"]))) as pool:
        json.dump(pool.map(run, [(cfg["dirty"], s) for s in cfg["grid"]], chunksize=1), open(sys.argv[2], "w"))
