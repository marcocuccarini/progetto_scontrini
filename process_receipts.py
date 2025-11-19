import os
import time
import json
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
INTERVAL_HOURS = 12     # run every 12 hours

PRODOTTI_DIR = "dataset/prodotti"
OCR_FILE = "dataset/_SELECT_commessa_id_activity_image_status_url_data_callback_rece_202510131147.csv"

APIS = [
    {
        "API_KEY": "AIzaSyBymZe7dunVj3j76DqA9q6VuDgimwYmryA",
        "MODEL_NAME": "gemini-2.5-flash",
        "LABEL": "API_1"
    },
    {
        "API_KEY": "AIzaSyBisxgy9D6A-EOAD5HulVouMIU56qLROe8",
        "MODEL_NAME": "gemini-2.5-flash",
        "LABEL": "API_2"
    }
]

# -------------------------------------------------------------------
# LOCAL STATE MANAGEMENT
# -------------------------------------------------------------------

def load_run_state(state_id):
    """Load per-prodotti progress state."""
    state_file = f"results/{state_id}.json"
    if not os.path.exists(state_file):
        return {"last_row_idx": -1, "processed_keys": []}
    with open(state_file, "r") as f:
        return json.load(f)


def save_run_state(state_id, last_row_idx, processed_keys):
    """Save per-prodotti progress state."""
    state_file = f"results/{state_id}.json"
    with open(state_file, "w") as f:
        json.dump(
            {"last_row_idx": last_row_idx, "processed_keys": list(processed_keys)},
            f,
            indent=2
        )


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def append_record_ndjson(record, result_dir):
    out_file = f"{result_dir}/results.ndjson"
    with open(out_file, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_simplified_json(receipt_desc, final_decision, result_dir):
    out_file = f"{result_dir}/simplified.json"

    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[receipt_desc] = final_decision

    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_snapshot(result_dir):
    snapshot = f"{result_dir}/SNAPSHOT.txt"
    with open(snapshot, "w") as f:
        f.write("snapshot written\n")


# -------------------------------------------------------------------
# IMPORTED UTILS
# -------------------------------------------------------------------

from utils import extract_items_from_row, compute_key


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def run_pipeline_for_prodotti(prodotti_file):
    print(f"\n📂 Using prodotti file: {prodotti_file}")

    import google.generativeai as genai
    from classes.model import LLMModel
    from classes.prompt_engine import PromptEngine
    from classes.pipeline_runner import PipelineRunner

    SNAPSHOT_EVERY = 50
    SLEEP_BETWEEN_CALLS = 0.0

    base_name = Path(prodotti_file).stem
    RESULT_DIR = f"results/{base_name}"
    os.makedirs(RESULT_DIR, exist_ok=True)

    model = LLMModel(model_name=os.getenv("MODEL_NAME", "gemini-2.5-flash"))
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    engine = PromptEngine(model)
    pipeline = PipelineRunner(engine)

    products_df = pd.read_csv(prodotti_file, sep=";", dtype=str).fillna("")
    products = products_df.to_dict(orient="records")

    ocr_df = pd.read_csv(OCR_FILE, sep=";", dtype=str).fillna("")

    state_id = f"state_{base_name}"
    run_state = load_run_state(state_id)
    processed_keys = set(run_state.get("processed_keys", []))
    start_idx = run_state.get("last_row_idx", -1) + 1

    print(f"🔁 Resuming from row {start_idx} with {len(processed_keys)} processed items")

    appended_since_snapshot = 0

    try:
        for idx, row in ocr_df.iterrows():
            if idx < start_idx:
                continue

            items = extract_items_from_row(row)
            if not items:
                continue

            for item in items:
                receipt_description = item["ReceiptDescription"]
                item_name = item.get("ItemName", "")
                key = compute_key(idx, receipt_description) + "_" + base_name

                if key in processed_keys:
                    continue

                try:
                    results = pipeline.run(
                        products=products,
                        receipt_description=receipt_description,
                        item_name=item_name,
                        row_idx=idx,
                        key=key
                    )

                    if "free" in str(results).lower() and "limit" in str(results).lower():
                        raise Exception("quota")

                except Exception as e:
                    msg = str(e).lower()
                    if any(x in msg for x in ["quota", "limit", "billing", "upgrade"]):
                        print("🛑 API LIMIT — saving state")
                        write_snapshot(RESULT_DIR)
                        save_run_state(state_id, idx, processed_keys)
                        return "API_LIMIT"
                    raise

                append_record_ndjson(results, RESULT_DIR)
                update_simplified_json(
                    receipt_description,
                    results.get("final_decision", {}),
                    RESULT_DIR
                )

                processed_keys.add(key)
                save_run_state(state_id, idx, processed_keys)

                appended_since_snapshot += 1
                if appended_since_snapshot >= SNAPSHOT_EVERY:
                    write_snapshot(RESULT_DIR)
                    appended_since_snapshot = 0

                if SLEEP_BETWEEN_CALLS > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS)

    except KeyboardInterrupt:
        write_snapshot(RESULT_DIR)
        return "INTERRUPTED"

    write_snapshot(RESULT_DIR)
    save_run_state(state_id, len(ocr_df) - 1, processed_keys)
    print(f"✅ Completed prodotti file: {prodotti_file}")
    return None


# -------------------------------------------------------------------
# SCHEDULER LOOP — FIXED VERSION
# -------------------------------------------------------------------

if __name__ == "__main__":
    while True:
        prodotti_files = sorted(Path(PRODOTTI_DIR).glob("*.csv"))

        if not prodotti_files:
            print("❌ No prodotti CSV files found.")
            time.sleep(60)
            continue

        api_exhausted_count = 0

        for api_cfg in APIS:
            print("\n===================================================")
            print(f"🔁 Running with {api_cfg['LABEL']} ({api_cfg['MODEL_NAME']})")
            print("===================================================\n")

            os.environ["GEMINI_API_KEY"] = api_cfg["API_KEY"]
            os.environ["MODEL_NAME"] = api_cfg["MODEL_NAME"]

            api_limit_reached = False

            for prodotti_file in prodotti_files:
                result = run_pipeline_for_prodotti(str(prodotti_file))

                if result == "API_LIMIT":
                    print(f"⛔ API LIMIT for {api_cfg['LABEL']} — switching API...")
                    api_limit_reached = True
                    break

            if api_limit_reached:
                api_exhausted_count += 1
                continue

        if api_exhausted_count == len(APIS):
            print("\n💤 All APIs exhausted — waiting for next cycle...\n")
            time.sleep(INTERVAL_HOURS * 3600)
            continue

        print(f"\n⏳ Sleeping {INTERVAL_HOURS} hours before next cycle...")
        time.sleep(INTERVAL_HOURS * 3600)
