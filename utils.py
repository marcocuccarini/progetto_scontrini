# utils.py

import os
import json
import hashlib

# === Config ===
OUTPUT_NDJSON = "results/matches.ndjson"
OUTPUT_JSON = "results/matches.json"
OUTPUT_SIMPLE_JSON = "results/predizioni_simplified.json"
RUN_STATE_FILE = "results/run_state.json"


# === Run state helpers ===
def load_run_state():
    if os.path.exists(RUN_STATE_FILE):
        try:
            with open(RUN_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load run state: {e}")
    return {"last_row_idx": -1, "processed_keys": []}


def save_run_state(last_row_idx, processed_keys):
    state = {"last_row_idx": last_row_idx, "processed_keys": list(processed_keys)}
    with open(RUN_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    print(f"💾 Run state saved (row={last_row_idx}, processed={len(processed_keys)})")


def extract_items_from_row(row):
    """
    Extracts items from OCR row. Supports json_callback or similar nested JSON columns.
    Returns a list of dicts with ReceiptDescription and ItemName.
    """
    items_list = []

    for col in row.index:
        if "json" in col.lower() or "callback" in col.lower():
            raw = row[col]
            if not raw or not isinstance(raw, str):
                continue
            try:
                json_data = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    json_data = json.loads(json.loads(raw))
                except Exception:
                    continue

            item_details = json_data.get("ItemDetails", {})
            items = item_details.get("Items", [])
            for item in items:
                desc = item.get("ReceiptDescription", "").strip()
                name = item.get("ItemName", "").strip()
                if desc:
                    items_list.append({"ReceiptDescription": desc, "ItemName": name})

    return items_list


import hashlib

def compute_key(row_idx, receipt_description):
    """
    Computes a unique SHA256 key for a receipt item.
    """
    key_str = f"{row_idx}|{receipt_description}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()



# === NDJSON / snapshot helpers ===
def append_record_ndjson(record):
    line = json.dumps(record, ensure_ascii=False)
    os.makedirs(os.path.dirname(OUTPUT_NDJSON), exist_ok=True)
    with open(OUTPUT_NDJSON, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def write_snapshot():
    snapshot = []
    if os.path.exists(OUTPUT_NDJSON):
        with open(OUTPUT_NDJSON, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshot.append(json.loads(line))
                except Exception as e:
                    print(f"⚠️ Failed to parse NDJSON line: {e}")

    if snapshot:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
            json.dump(snapshot, out_f, indent=2, ensure_ascii=False)
        print(f"💾 Snapshot saved to {OUTPUT_JSON} ({len(snapshot)} records)")
    else:
        print("⚠️ No records to save in snapshot")


# === Simplified JSON helper ===
def update_simplified_json(receipt_description, final_decision):
    simplified_entry = {
        "descrizione": receipt_description,
        "match": final_decision.get("match_finale", [])
    }

    if os.path.exists(OUTPUT_SIMPLE_JSON):
        try:
            with open(OUTPUT_SIMPLE_JSON, "r", encoding="utf-8") as f:
                simple_data = json.load(f)
        except Exception:
            simple_data = []
    else:
        simple_data = []

    simple_data.append(simplified_entry)
    with open(OUTPUT_SIMPLE_JSON, "w", encoding="utf-8") as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
