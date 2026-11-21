"""Download the InfraredSolarModules dataset (about 15 MB) into ``data/``.

Usage:  python -m solarfault.download
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://raw.githubusercontent.com/RaptorMaps/InfraredSolarModules/"
       "master/2020-02-14_InfraredSolarModules.zip")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    target = DATA_DIR / "InfraredSolarModules"
    if (target / "module_metadata.json").exists():
        print(f"Already downloaded: {target}")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        zf.extractall(DATA_DIR, members=members)
    print(f"Extracted to {target}")


if __name__ == "__main__":
    main()
