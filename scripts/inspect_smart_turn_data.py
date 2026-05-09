"""Inspect Smart Turn v3.1 dataset metadata via HuggingFace datasets-server.

Avoids downloading the 36.8GB by using the parquet stats API.
"""

import json
import urllib.request

DATASET_ID = "pipecat-ai/smart-turn-data-v3.1-train"
BASE = "https://datasets-server.huggingface.co"


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    # Stats endpoint gives column statistics from the parquet metadata
    print(f"Fetching stats for {DATASET_ID}...\n")

    info = fetch(f"{BASE}/info?dataset={DATASET_ID}")
    print("=== Splits ===")
    splits_info = info.get("dataset_info", {}).get("default", {}).get("splits", {})
    for split_name, split_data in splits_info.items():
        print(f"  {split_name}: {split_data.get('num_examples', '?')} examples")

    features = info.get("dataset_info", {}).get("default", {}).get("features", {})
    print("\n=== Features ===")
    for name, schema in features.items():
        if isinstance(schema, dict) and "_type" in schema:
            print(f"  {name}: {schema.get('_type')} {schema}")
        else:
            print(f"  {name}: {schema}")

    print("\n=== Statistics for label columns ===")
    stats = fetch(f"{BASE}/statistics?dataset={DATASET_ID}&config=default&split=train")
    for col_stat in stats.get("statistics", []):
        col_name = col_stat.get("column_name")
        if col_name in {
            "endpoint_bool",
            "midfiller",
            "endfiller",
            "language",
            "synthetic",
            "dataset",
        }:
            print(f"\n  --- {col_name} ---")
            stat_data = col_stat.get("column_statistics", {})
            print(f"    type: {col_stat.get('column_type')}")
            for k, v in stat_data.items():
                if k in {"value_counts", "frequencies", "top10", "histogram"}:
                    if isinstance(v, dict):
                        # Show top entries
                        entries = list(v.items())[:15]
                        print(f"    {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in entries))
                    else:
                        print(f"    {k}: {str(v)[:300]}")
                elif isinstance(v, (int, float, str)) or isinstance(v, list) and len(v) <= 20:
                    print(f"    {k}: {v}")

    print("\n=== Audio length stats ===")
    for col_stat in stats.get("statistics", []):
        if col_stat.get("column_name") == "audio":
            print(json.dumps(col_stat, indent=2)[:500])


if __name__ == "__main__":
    main()
