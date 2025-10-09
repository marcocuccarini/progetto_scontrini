import os
import re
import json
import time
import hashlib
import pandas as pd
from datetime import datetime
from google import genai

# === CONFIG ===
products_file = "dataset/prodotti.csv"
ocr_file = "dataset/data.csv"
OUTPUT_NDJSON = "results/matches.ndjson"
OUTPUT_JSON = "results/matches.json"
SNAPSHOT_EVERY = 50
SLEEP_BETWEEN_CALLS = 0.0

os.makedirs("results", exist_ok=True)

client = genai.Client()  # GEMINI_API_KEY must be set in environment

# === Load CSVs ===
products_df = pd.read_csv(products_file, sep=";", dtype=str).fillna("")
ocr_df = pd.read_csv(ocr_file, sep=";", dtype=str).fillna("")
products = products_df.to_dict(orient="records")

# === helper: extract items from nested JSON columns ===
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

# === helper: compute resume key ===
def compute_key(row_idx, receipt_description):
    key_str = f"{row_idx}|{receipt_description}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

# === load already processed keys from NDJSON ===
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
                continue
print(f"Loaded {len(processed_keys)} previously processed items from {OUTPUT_NDJSON}")

# === helper: append a record atomically ===
def append_record_ndjson(record):
    line = json.dumps(record, ensure_ascii=False)
    with open(OUTPUT_NDJSON, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

# === helper: write full snapshot ===
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

# === helper: parse Gemini output ===
def parse_gemini_output(raw_output):
    if not raw_output:
        return {"error": "Empty response"}
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"error": f"Could not parse Gemini output: {e}", "raw": raw_output}

# === MAIN processing loop ===
appended_since_snapshot = 0

try:
    for idx, row in ocr_df.iterrows():
        items = extract_items_from_row(row)
        if not items:
            print(f"⚠️  No items found in row {idx}. Skipping.")
            continue

        for item in items:
            receipt_description = item["ReceiptDescription"]
            item_name_csv = item["ItemName"]

            key = compute_key(idx, receipt_description)
            if key in processed_keys:
                print(f"⏭ Skipping already processed row {idx} (key {key[:8]})")
                continue

            # --- Gemini prompt (kept the same as your original) ---
            prompt = f"""

Sei un assistente che deve trovare corrispondenze tra prodotti OCR da scontrini e una lista di prodotti in promozione. 
Ti verrà fornita una lista di prodotti in offerta, ognuno con questi campi:
- id
- brand
- nome
- nome_normalizzato (può essere vuoto)
- ean

Ti verrà anche fornita una descrizione di un prodotto letta da uno scontrino (campo "ReceiptDescription"), 
che può contenere abbreviazioni, punteggiatura, o errori OCR.

Il tuo compito è:
1. Cercare il prodotto nella lista che più probabilmente corrisponde alla descrizione OCR.
2. Restituire i tre migliori match, ciascuno con:
   - id
   - nome
   - brand
   - punteggio di confidenza (da 0.0 a 1.0)
   - spiegazione sintetica del perché ritieni ci sia una corrispondenza.
3. Se non c’è nessuna corrispondenza credibile (es. confidenza < 0.4), specifica "Nessuna corrispondenza affidabile".
4. Non inventare id, nomi o campi. Usa esclusivamente le voci così come fornite nella lista.

---

Ecco la lista dei prodotti in promozione:
{json.dumps(products)}

---

Descrizione OCR da scontrino:
{json.dumps(receipt_description)}

---

Restituisci la risposta **in formato JSON**.
"""

            # --- call Gemini ---
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                response_text = response.text.strip()
            except Exception as e:
                print(f"❌ Gemini API error at row {idx}: {e}. Skipping.")
                rec = {
                    "row_idx": int(idx),
                    "key": key,
                    "receipt_description": receipt_description,
                    "ItemName": item_name_csv,
                    "match_result": {"error": f"Gemini error: {str(e)}"},
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                append_record_ndjson(rec)
                processed_keys.add(key)
                appended_since_snapshot += 1
                continue

            match_data = parse_gemini_output(response_text)

            record = {
                "row_idx": int(idx),
                "key": key,
                "receipt_description": receipt_description,
                "ItemName": item_name_csv,
                "match_result": match_data,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            append_record_ndjson(record)
            processed_keys.add(key)
            appended_since_snapshot += 1
            processed_count += 1

            print(f"✅ Saved row {idx} key {key[:8]} (total saved: {processed_count})")

            if SLEEP_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_BETWEEN_CALLS)

            if appended_since_snapshot >= SNAPSHOT_EVERY:
                write_snapshot()
                appended_since_snapshot = 0

except KeyboardInterrupt:
    print("\n⏸ Interrupted by user. Writing snapshot and exiting...")
    write_snapshot()
except Exception as e:
    print(f"\n❌ Fatal error: {e}. Writing snapshot and exiting...")
    write_snapshot()
    raise

# final snapshot
write_snapshot()
print("\n✅ Processing complete.")
