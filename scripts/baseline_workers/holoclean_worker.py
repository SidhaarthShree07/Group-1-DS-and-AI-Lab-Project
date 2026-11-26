#!/usr/bin/env python3
"""The official HoloClean (probabilistic repair) on one dirty table. Runs in the HoloClean environment (Python 3.6).

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

Usage: holoclean_worker.py <HoloClean clone> <socket dir> <config.json> <out.json>. Config: name, dirty, constraints, seed.
No clean data is read. Changes to HoloClean: the database connection uses a local Unix socket. Its own preprocessing
lowercases values and its SQL drops the characters { } " from inferred values, so an inferred value is compared and
mapped back to the most frequent original spelling in the same dirty column under that same transformation.
"""
import collections
import json
import logging
import re
import sys
import time
from multiprocessing import Pool

HC, SOCKET, CFG, OUT = sys.argv[1:5]
sys.path.insert(0, HC)
import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
import sqlalchemy as sql  # noqa: E402
import dataset.dbengine as dbe  # noqa: E402


def socket_engine(self, user, pwd, db, host="localhost", port=5432, pool_size=20, timeout=60000):
    self.timeout = timeout
    self._pool = Pool(pool_size) if pool_size > 1 else None
    self.conn_args = "dbname=%s user=%s host=%s port=5432" % (db, user, SOCKET)
    self.engine = sql.create_engine("postgresql+psycopg2://", creator=lambda: psycopg2.connect(self.conn_args),
                                    pool_size=pool_size)
    self.conn = self.engine


dbe.DBengine.__init__ = socket_engine
import holoclean  # noqa: E402
from dataset import AuxTables  # noqa: E402
from detect import NullDetector, ViolationDetector  # noqa: E402
from repair.featurize import ConstraintFeaturizer, FreqFeaturizer, InitAttrFeaturizer, OccurAttrFeaturizer  # noqa: E402

NULL = "_nan_"


def key(value):
    return re.sub(r'[{"}]', "", value.strip().lower())


def main():
    cfg = json.load(open(CFG))
    logging.getLogger().setLevel(logging.WARNING)
    t0 = time.time()
    rec = {"name": cfg["name"], "seed": cfg["seed"]}
    try:
        dirty = pd.read_csv(cfg["dirty"], dtype=str, keep_default_na=False, encoding="utf-8")
        hc = holoclean.HoloClean(db_name=cfg.get("db", "holo"), domain_thresh_1=0, domain_thresh_2=0, weak_label_thresh=0.99,
                                 max_domain=10000, cor_strength=0.6, nb_cor_strength=0.8, epochs=10, weight_decay=0.01,
                                 learning_rate=0.001, threads=1, batch_size=1, verbose=False, timeout=3 * 60000,
                                 feature_norm=False, weight_norm=False, print_fw=False, seed=cfg["seed"]).session
        hc.load_data(cfg["name"], cfg["dirty"])
        lines = [x for x in open(cfg["constraints"], encoding="utf-8").read().splitlines() if x.strip()]
        detectors, features = [NullDetector()], [InitAttrFeaturizer(), OccurAttrFeaturizer(), FreqFeaturizer()]
        if lines:
            hc.load_dcs(cfg["constraints"])
            hc.ds.set_constraints(hc.get_dcs())
            detectors.append(ViolationDetector())
            features.append(ConstraintFeaturizer())
        else:
            hc.ds.set_constraints([])
        hc.detect_errors(detectors)
        noisy = hc.ds.aux_table[AuxTables.dk_cells].df.reset_index()
        rec["noisy"] = sorted([int(r), c] for r, c in zip(noisy["_tid_"], noisy["attribute"]))
        hc.setup_domain()
        hc.repair_errors(features)
        spelling = {c: collections.defaultdict(collections.Counter) for c in dirty.columns}
        for c in dirty.columns:
            for v in dirty[c]:
                spelling[c][key(v)][v] += 1
        inferred = hc.ds.aux_table[AuxTables.inf_values_dom].df.reset_index()
        rec["inferred_cells"] = len(inferred)
        repairs, unmapped = [], 0
        for r, c, v in zip(inferred["_tid_"], inferred["attribute"], inferred["rv_value"]):
            old, new = dirty.at[int(r), c], "" if v in (NULL, None) else v
            if key(new) == key(old):
                continue
            options = spelling[c].get(key(new))
            unmapped += options is None
            repairs.append([int(r), c, options.most_common(1)[0][0] if options else new])
        rec.update(status="completed", repairs=sorted(repairs), unmapped=unmapped, constraints=len(lines))
    except Exception as e:  # reported, never dropped
        import traceback
        rec.update(status="failed: %s: %s" % (type(e).__name__, str(e)[:200]), trace=traceback.format_exc()[-1500:])
    rec["seconds"] = round(time.time() - t0, 1)
    json.dump(rec, open(OUT, "w"))


if __name__ == "__main__":
    main()
