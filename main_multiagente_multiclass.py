import os
import time
import pandas as pd
from classes.model import LLMModel
from classes.prompt_engine import PromptEngine
from classes.pipeline_runner import PipelineRunner
from utils import *
import google.generativeai as genai


# === Config ===
PRODUCTS_FILE = "dataset/prodotti.csv"
OCR_FILE = "dataset/_SELECT_commessa_id_activity_image_status_url_data_callback_rece_202510131147.csv"
SNAPSHOT_EVERY = 50
SLEEP_BETWEEN_CALLS = 0.0

os.makedirs("results", exist_ok=True)

# === Initialize model, engine, pipeline ===
model = LLMModel(model_name="gemini-2.5-flash")
engine = PromptEngine(model)
pipeline = PipelineRunner(engine)

# === Load CSVs ===
products_df = pd.read_csv(PRODUCTS_FILE, sep=";", dtype=str).fillna("")
ocr_df = pd.read_csv(OCR_FILE, sep=";", dtype=str).fillna("")
products = products_df.to_dict(orient="records")

# === Resume state ===
run_state = load_run_state()
processed_keys = set(run_state.get("processed_keys", []))
start_idx = run_state.get("last_row_idx", -1) + 1
print(f"🔁 Resuming from row {start_idx} with {len(processed_keys)} previously processed items")

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
            item_name = item.get("ItemName", "")
            key = compute_key(idx, receipt_description)

            if key in processed_keys:
                print(f"⏭ Skipping already processed row {idx} (key {key[:8]})")
                continue

            print(f"\n🧾 Row {idx} | Receipt: {receipt_description}\n")

            # === Run pipeline safely (detect free tier limit) ===
            try:
                results = pipeline.run(
                    products=products,
                    receipt_description=receipt_description,
                    item_name=item_name,
                    row_idx=idx,
                    key=key
                )

                # If model outputs a limit warning inside the response text
                if "free" in str(results).lower() and "limit" in str(results).lower():
                    raise Exception("Free tier usage limit reached (detected in response).")

            except Exception as e:
                message = str(e).lower()

                # === Detect Gemini Free Limit ===
                if ("quota" in message or
                    "exceeded" in message or
                    "billing" in message or
                    ("free" in message and "limit" in message) or
                    "upgrade" in message):

                    print("\n🛑 GEMINI FREE TIER LIMIT REACHED!")
                    print("💾 Saving progress before stopping...")

                    write_snapshot()
                    save_run_state(idx, processed_keys)

                    print("✅ Saved. You can run the script later to resume.")
                    exit(0)

                # Unexpected error → rethrow
                raise

            # === Append NDJSON ===
            append_record_ndjson(results)
            processed_keys.add(key)
            appended_since_snapshot += 1

            # === Update simplified JSON ===
            update_simplified_json(receipt_description, results.get("final_decision", {}))

            # === Save run state ===
            save_run_state(idx, processed_keys)

            if SLEEP_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_BETWEEN_CALLS)

            # === Write snapshot periodically ===
            if appended_since_snapshot >= SNAPSHOT_EVERY:
                write_snapshot()
                appended_since_snapshot = 0

except KeyboardInterrupt:
    print("\n⏸ Interrupted by user. Writing final snapshot...")
    write_snapshot()
except Exception as e:
    print(f"\n❌ Fatal error: {e}. Writing snapshot...")
    write_snapshot()
    raise

# === Final snapshot ===
write_snapshot()
save_run_state(len(ocr_df) - 1, processed_keys)
print("\n✅ Processing complete. JSON and NDJSON files updated.")
