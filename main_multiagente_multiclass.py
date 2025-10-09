import os
import re
import json
import time
import hashlib
import pandas as pd
from datetime import datetime
from google import genai
from classes.PromptEngine import PromptEngine
from classes.Pipeline import PipelineRunner
from utils.utils import *

# === CONFIG ===
products_file = "dataset/prodotti.csv"
ocr_file = "dataset/data.csv"
OUTPUT_NDJSON = "results/matches.ndjson"
OUTPUT_JSON = "results/matches.json"
OUTPUT_SIMPLE_JSON = "results/predizioni_simplified.json"
RUN_STATE_FILE = "results/run_state.json"
SNAPSHOT_EVERY = 50
SLEEP_BETWEEN_CALLS = 0.0

os.makedirs("results", exist_ok=True)

client = genai.Client()  # GEMINI_API_KEY deve essere impostata nell’ambiente
engine = PromptEngine(client)
pipeline = PipelineRunner(engine)

# === Load CSVs ===
products_df = pd.read_csv(products_file, sep=";", dtype=str).fillna("")
ocr_df = pd.read_csv(ocr_file, sep=";", dtype=str).fillna("")
products = products_df.to_dict(orient="records")


# === Stato iniziale ===
run_state = load_run_state()
processed_keys = set(run_state.get("processed_keys", []))
start_idx = run_state.get("last_row_idx", -1) + 1
print(f"🔁 Resuming from row {start_idx} with {len(processed_keys)} previously processed items")

# === MAIN LOOP ===
appended_since_snapshot = 0

try:
    for idx, row in ocr_df.iterrows():
        if idx < start_idx:
            continue

        items = extract_items_from_row(row)
        if not items:
            print(f"⚠️  No items found in row {idx}. Skipping.")
            continue

        for item in items:
            receipt_description = item["ReceiptDescription"]
            item_name = item["ItemName"]
            key = compute_key(idx, receipt_description)

            if key in processed_keys:
                print(f"⏭ Skipping already processed row {idx} (key {key[:8]})")
                continue

            print("\n=============================================")
            print(f"🧾 Row {idx} | Receipt: {receipt_description}")
            print("=============================================\n")

            # === Esegui pipeline completa ===
            results = pipeline.run(
                products=products,
                receipt_description=receipt_description,
                item_name=item_name,
                row_idx=idx,
                key=key
            )

            append_record_ndjson(results)
            processed_keys.add(key)
            appended_since_snapshot += 1
            save_run_state(idx, processed_keys)
            print(f"✅ Row {idx} processed (key {key[:8]})")

            # === Aggiorna file JSON semplificato ===
            simplified_entry = {
                "descrizione": receipt_description,
                "match": results["final_decision"].get("match_finale", [])
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

# === Final snapshot ===
write_snapshot()
save_run_state(len(ocr_df) - 1, processed_keys)
print("\n✅ Processing complete.")
