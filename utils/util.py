# === Helper: estrazione item da JSON ===
def extract_items_from_row(row):
    items_list = []
    for col in row.index:
        if any(k in col.lower() for k in ["json", "callback", "data"]):
            raw = row[col]
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                json_data = json.loads(raw)
            except Exception:
                try:
                    json_data = json.loads(json.loads(raw))
                except Exception:
                    continue

            items = json_data.get("ItemDetails", {}).get("Items", [])
            for item in items:
                desc = item.get("ReceiptDescription", "").strip()
                name = item.get("ItemName", "").strip()
                if desc:
                    items_list.append({"ReceiptDescription": desc, "ItemName": name})
    return items_list

# === Helper: compute unique key per item ===
def compute_key(row_idx, receipt_description):
    key_str = f"{row_idx}|{receipt_description}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

# === Gestione stato run (resume) ===
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

# === Gestione file NDJSON e snapshot ===
def append_record_ndjson(record):
    line = json.dumps(record, ensure_ascii=False)
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
                    rec = json.loads(line)
                    snapshot.append(rec)
                except Exception:
                    continue
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
        json.dump(snapshot, out_f, indent=2, ensure_ascii=False)
    print(f"💾 Snapshot saved to {OUTPUT_JSON} ({len(snapshot)} records)")