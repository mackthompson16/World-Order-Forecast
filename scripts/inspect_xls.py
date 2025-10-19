import sys
from pathlib import Path
import pandas as pd


def main(path_str: str):
    p = Path(path_str)
    print(f"File: {p}  Exists: {p.exists()}  Size: {p.stat().st_size if p.exists() else 0}")
    try:
        x = pd.ExcelFile(p)
    except Exception as e:
        print(f"[excel-open-fail] {e}")
        return 1
    print("Sheets:", x.sheet_names)
    for sn in x.sheet_names:
        print(f"\n-- Sheet: {sn} --")
        try:
            df = x.parse(sn, nrows=12)
            print("cols:", list(df.columns))
            print(df.head(5).to_string(index=False))
        except Exception as e:
            print(f"[parse-fail] {sn}: {e}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_xls.py <path-to-xls/xlsx>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

