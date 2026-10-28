#!/usr/bin/env python3
"""Download every raw data source from its original owner, at a pinned version, and check SHA-256.

Author: Sidhaarth Shree

Nothing is stored in git: files go to data/raw/ (ignored). Sources and versions are the ones in data/download_log.md.

  Raha (Hospital, Flights, Beers, Rayyan, Movies, Tax)   github.com/BigDaMa/raha, pinned commit
  BClean (Inpatient, Facilities)                           github.com/thethe-github/BClean, pinned commit
  ZeroED (Billionaire)                                     github.com/ZJU-DataVIP-Lab/ZeroED, pinned commit
  Clean Me If You Can, Germany slice                       Zenodo record 20841898 (two members of a 7.3 GB zip, by range request)
  UCI Adult, Bike Sharing, Wine Quality                    archive.ics.uci.edu
  TableEG annotation files                                 Google Drive only: copy the two TableEG zip files (see the TableEG
                                                           README) into data/raw/tableeg/ and this script extracts the
                                                           annotation files. They are small; the zips are about 840 MB.

Usage: python scripts/download_data.py [--root .] [--only raha bclean zeroed germany uci tableeg]
Exit code 0 when every requested source is present and verified, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.load_core import CORE_TABLE_CONFIGS, PINNED_RAHA_COMMIT  # noqa: E402
from src.data.load_external import SOURCES  # noqa: E402

BCLEAN = "bb8e09e3119dcc85475184592a311901a6b4c290"
ZEROED = "2b84266e1d8b2014d36643d343429d9a20bd769e"
ZENODO = "https://zenodo.org/records/20841898/files/AddressTable.zip?download=1"
GERMANY = {"clean": "AddressTable/de_slice/named/slice_de_clean.csv", "dirty": "AddressTable/de_slice/named/slice_de_dirty.csv"}
UCI = {  # file: (url, sha256 observed on 28 Oct 2026)
    "adult.zip": ("https://archive.ics.uci.edu/static/public/2/adult.zip",
                  "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"),
    "bike+sharing+dataset.zip": ("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
                                 "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"),
    "wine+quality.zip": ("https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
                         "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4"),
}
ADULT_DATA_SHA = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"  # adult.data inside adult.zip
TABLEEG_NEEDED = ["hospital/hospital_annotation.jsonl", "flight/flight_annotation.csv", "beers/beers_annotation.csv",
                  "rayyan/rayyan_annotation.jsonl", "movie_metadata/movie_annotation.jsonl"]
DRIVE = "https://drive.google.com/drive/folders/10LdB9LGgymbI6W8D2936uRF6eFRL24xy"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and (expected is None or sha256(dest) == expected)):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    got = sha256(dest)
    ok = expected is None or got == expected
    print(f"  {'OK ' if ok else 'BAD'} {dest.name} sha256={got[:16]}…")
    return ok


def raha(raw: Path) -> bool:
    ok = True
    base = f"https://raw.githubusercontent.com/BigDaMa/raha/{PINNED_RAHA_COMMIT}/datasets"
    pairs = [(c["raha_subpath"], c["expected_sha256_clean"], c["expected_sha256_dirty"]) for c in CORE_TABLE_CONFIGS]
    tax = next(s for s in SOURCES if s["name"] == "tax")
    pairs.append(("tax", *tax["sha"]))
    for sub, sc, sd in pairs:
        for name, sha in (("clean.csv", sc), ("dirty.csv", sd)):
            ok &= fetch(f"{base}/{sub}/{name}", raw / "raha" / "datasets" / sub / name, sha)
    return ok


def bclean(raw: Path) -> bool:
    ok = True
    for s in SOURCES:
        if s["source"] == "BClean":
            for rel, sha in zip((s["clean"], s["dirty"]), s["sha"]):
                url = f"https://raw.githubusercontent.com/thethe-github/BClean/{BCLEAN}/dataset/" + rel.split("dataset/", 1)[1]
                ok &= fetch(url, raw / rel, sha)
    return ok


def zeroed(raw: Path) -> bool:
    ok = True
    s = next(x for x in SOURCES if x["name"] == "billionaire")
    for rel, sha in zip((s["clean"], s["dirty"]), s["sha"]):
        ok &= fetch(f"https://raw.githubusercontent.com/ZJU-DataVIP-Lab/ZeroED/{ZEROED}/" + rel.split("/", 1)[1], raw / rel, sha)
    return ok


def germany(raw: Path) -> bool:
    from remotezip import RemoteZip  # only needed here

    s = next(x for x in SOURCES if x["name"] == "germany")
    ok = True
    todo = [(label, member, raw / rel, sha) for (label, member), rel, sha in zip(GERMANY.items(), (s["clean"], s["dirty"]), s["sha"])]
    if all(p.exists() and sha256(p) == sha for _, _, p, sha in todo):
        for _, _, p, _ in todo:
            print(f"  OK  {p.name} (already present)")
        return True
    with RemoteZip(ZENODO) as archive:
        for label, member, dest, sha in todo:
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  range-downloading {member}")
            with archive.open(member) as src, open(dest, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            good = sha256(dest) == sha
            print(f"  {'OK ' if good else 'BAD'} {dest.name}")
            ok &= good
    return ok


def uci(raw: Path) -> bool:
    ok = True
    for name, (url, sha) in UCI.items():
        ok &= fetch(url, raw / name, sha)
    adult = raw / "adult"
    if not (adult / "adult.data").exists():
        with zipfile.ZipFile(raw / "adult.zip") as z:
            z.extractall(adult)
    good = sha256(adult / "adult.data") == ADULT_DATA_SHA
    print(f"  {'OK ' if good else 'BAD'} adult.data (adult.test is present in the archive but is never read)")
    return ok & good


def tableeg(raw: Path) -> bool:
    base = raw / "tableeg"
    source = base / "source"
    if all((source / rel).exists() for rel in TABLEEG_NEEDED):
        print("  OK  TableEG annotation files already present")
        return True
    zips = sorted(base.glob("TableEG*.zip"))
    if not zips:
        print(f"  MISSING TableEG zip files in {base}. They are only on Google Drive ({DRIVE});\n"
              f"          copy them into that folder and run this script again.")
        return False
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                if "/source/" in member and "annotation" in member.lower():
                    target = source / member.split("/source/", 1)[1]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
    ok = all((source / rel).exists() for rel in TABLEEG_NEEDED)
    print(f"  {'OK ' if ok else 'BAD'} extracted TableEG annotation files to {source}")
    return ok


STEPS = {"raha": raha, "bclean": bclean, "zeroed": zeroed, "germany": germany, "uci": uci, "tableeg": tableeg}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", nargs="*", choices=list(STEPS), default=list(STEPS))
    args = ap.parse_args()
    raw = Path(args.root).resolve() / "data" / "raw"
    results = {}
    for name in args.only:
        print(f"[{name}]")
        try:
            results[name] = STEPS[name](raw)
        except Exception as e:  # report and continue with the other sources
            print(f"  FAILED: {type(e).__name__}: {e}")
            results[name] = False
    bad = [k for k, v in results.items() if not v]
    print("\nall requested sources verified" if not bad else f"\nnot verified: {', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
