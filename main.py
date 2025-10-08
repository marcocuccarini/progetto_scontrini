from google import genai
import pandas as pd
import json
import os
import hashlib
import time
from datetime import datetime

# === CONFIG ===
products_file = "dataset/prodotti.csv"
ocr_file = "dataset/data.csv"
OUTPUT_NDJSON = "results/matches.ndjson"     # incremental append file
OUTPUT_JSON = "results/matches.json"         # optional snapshot file
SNAPSHOT_EVERY = 50                  # write full matches.json every N appended results
SLEEP_BETWEEN_CALLS = 0.0            # set >0 if you need to throttle calls

# === Gemini client ===
client = genai.Client()  # GEMINI_API_KEY must be in env

# === Load CSVs (semicolon-separated as in your files) ===
products_df = pd.read_csv(products_file, sep=";", dtype=str).fillna("")
ocr_df = pd.read_csv(ocr_file, sep=";", dtype=str).fillna("")

products = products_df.to_dict(orient="records")

# === helper: extract ReceiptDescription from nested JSON columns ===
def extract_receipt_descriptions(row):
    json_data = None
    for col in row.index:
        if any(k in col.lower() for k in ["json", "callback", "data"]):
            raw = row[col]
            if not isinstance(raw, str) or raw.strip() == "":
                continue
            # Try single decode, then double decode
            try:
                json_data = json.loads(raw)
                break
            except Exception:
                try:
                    json_data = json.loads(json.loads(raw))
                    break
                except Exception:
                    continue

    descriptions = []
    if json_data:
        try:
            items = json_data.get("ItemDetails", {}).get("Items", [])
            for item in items:
                desc = item.get("ReceiptDescription")
                if desc:
                    descriptions.append(desc.strip())
        except Exception:
            pass
    return descriptions

# === helper: compute resume key ===
def compute_key(row_idx, receipt_description):
    key_str = f"{row_idx}|{receipt_description}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

# === load already processed keys from NDJSON (if exists) ===
processed_keys = set()
processed_count = 0
if os.path.exists(OUTPUT_NDJSON):
    with open(OUTPUT_NDJSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                k = rec.get("key")
                if k:
                    processed_keys.add(k)
                    processed_count += 1
            except Exception:
                # ignore malformed lines
                continue
print(f"Loaded {len(processed_keys)} previously processed items from {OUTPUT_NDJSON}")

# === helper: append a record atomically (append + fsync) ===
def append_record_ndjson(record):
    line = json.dumps(record, ensure_ascii=False)
    with open(OUTPUT_NDJSON, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass  # not critical on some platforms

# === helper: write full snapshot from NDJSON to JSON ===
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
    print(f"Snapshot saved to {OUTPUT_JSON} ({len(snapshot)} records)")

# === MAIN processing loop ===
appended_since_snapshot = 0

try:
    for idx, row in ocr_df.iterrows():
        # extract descriptions
        descriptions = []
        if "ReceiptDescription" in row and isinstance(row["ReceiptDescription"], str) and row["ReceiptDescription"].strip():
            descriptions = [row["ReceiptDescription"].strip()]
        else:
            descriptions = extract_receipt_descriptions(row)

        if not descriptions:
            print(f"⚠️  No valid description found for row {idx}. Skipping.")
            continue

        for receipt_description in descriptions:
            key = compute_key(idx, receipt_description)
            if key in processed_keys:
                print(f"⏭ Skipping already processed row {idx} (key {key[:8]})")
                continue

            prompt = f"""
You are an assistant that matches a receipt description to products.
Receipt description: "{receipt_description}"
Products: {json.dumps(products)}
Return the top 3 matches with their id, name, brand, confidence (0.0-1.0), and a brief explanation.
Format the output as JSON.
"""

            # call Gemini
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                response_text = response.text.strip()
            except Exception as e:
                print(f"❌ Gemini API error at row {idx}: {e}. Will retry after short wait.")
                time.sleep(5)  # small backoff; you might want more sophisticated retry
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    response_text = response.text.strip()
                except Exception as e2:
                    print(f"❌ Retry failed for row {idx}: {e2}. Skipping this item.")
                    # Save a failed record so you don't retry endlessly
                    rec = {
                        "row_idx": int(idx),
                        "key": key,
                        "receipt_description": receipt_description,
                        "match_result": {"error": f"Gemini error: {str(e2)}"},
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                    append_record_ndjson(rec)
                    processed_keys.add(key)
                    appended_since_snapshot += 1
                    continue

            # parse response JSON (if possible)
            try:
                match_data = json.loads(response_text)
            except json.JSONDecodeError:
                match_data = {"error": "Could not parse Gemini output", "raw": response_text}

            # build record and append immediately
            record = {
                "row_idx": int(idx),
                "key": key,
                "receipt_description": receipt_description,
                "match_result": match_data,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            append_record_ndjson(record)
            processed_keys.add(key)
            appended_since_snapshot += 1
            processed_count += 1

            print(f"✅ Saved row {idx} key {key[:8]} (total saved: {processed_count})")

            # optional throttle
            if SLEEP_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_BETWEEN_CALLS)

            # periodic snapshot
            if appended_since_snapshot >= SNAPSHOT_EVERY:
                write_snapshot()
                appended_since_snapshot = 0

except KeyboardInterrupt:
    print("\n⏸ Interrupted by user. Writing snapshot and exiting...")
    write_snapshot()
    print("Exit after KeyboardInterrupt.")
except Exception as e:
    print(f"\n❌ Fatal error: {e}. Writing snapshot and exiting...")
    write_snapshot()
    raise

# final snapshot
write_snapshot()
print("\n✅ Processing complete.")
