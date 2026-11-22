"""Unpack the 'everything in one file' Quranpedia bundle into data/raw/.
Usage: python -m qurangpt.ingestion.extract path/to/bundle.zip
"""
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

from qurangpt.config import RAW


def main(src: str):
    src = Path(src)
    RAW.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            z.extractall(RAW)
    elif tarfile.is_tarfile(src):
        with tarfile.open(src) as t:
            t.extractall(RAW)
    elif src.is_dir():
        for p in src.rglob("*"):
            if p.is_file():
                shutil.copy2(p, RAW / p.name)
    else:
        raise SystemExit("Not a zip/tar/folder. If it is a single .json.gz just copy it to data/raw/")
    files = sorted(p.relative_to(RAW) for p in RAW.rglob("*") if p.is_file())
    print(f"{len(files)} files in data/raw:")
    for f in files[:60]:
        print("  ", f)


if __name__ == "__main__":
    main(sys.argv[1])
