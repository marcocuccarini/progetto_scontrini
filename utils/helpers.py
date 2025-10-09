import json
import hashlib
import os

def compute_key(row_idx, receipt_description):
    key_str = f"{row_idx}|{receipt_description}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

def append_record_ndjson(record, filepath):
    line = json.dumps(record, ensure_ascii=False)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

def write_snapshot(ndjson_file, json_file):
    snapshot = []
    if os.path.exists(ndjson_file):
        with open(ndjson_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    snapshot.append(json.loads(line))
    with open(json_file, "w", encoding="utf-8") as out_f:
        json.dump(snapshot, out_f, indent=2, ensure_ascii=False)
    print(f"Snapshot saved to {json_file} ({len(snapshot)} records)")
