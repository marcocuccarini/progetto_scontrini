import csv
import json
import logging
from tqdm import tqdm
from classes.LLM import LLM

# ------------------- CONFIG -------------------
CSV_RECEIPTS = "dataset/data.csv"
CSV_PRODUCTS = "dataset/prodotti.csv"
CSV_PREDICTIONS = "receipt_match_pairs.csv"
JSON_FILE = "receipt_match_pairs.json"
SUMMARY_FILE = "method_summary.txt"

BATCH_SIZE = 10       # Number of receipt items to process at once
SAMPLE_SIZE = 10      # Number of receipts to process (set None to process all)
CONFIDENCE_THRESHOLD = 0.6

logging.basicConfig(filename="logs.txt", level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------- INITIALIZE LLM -------------------
llm = LLM(model="gemma3:4b", temperature=0.3, max_tokens=1024)

# ------------------- HELPER FUNCTIONS -------------------
def chunk_list(lst, n):
    """Split a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ------------------- LOAD PRODUCTS -------------------
possible_names = []
with open(CSV_PRODUCTS, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        if row.get("nome"):
            possible_names.append(row["nome"])

print(f"Loaded {len(possible_names)} products.")

results = []
predictions = []

# ------------------- PROCESS RECEIPTS -------------------
with open(CSV_RECEIPTS, mode="r", newline="", encoding="utf-8") as f:
    reader = list(csv.DictReader(f, delimiter=";"))

    # Apply subsample if SAMPLE_SIZE is set
    if SAMPLE_SIZE is not None:
        reader = reader[:SAMPLE_SIZE]

    if not reader or "json_callback" not in reader[0]:
        logging.error("Column 'json_callback' not found in CSV")
        exit(1)

    for idx, row in enumerate(tqdm(reader, desc="Processing receipts")):
        json_str = row.get("json_callback")
        if not json_str:
            logging.warning(f"Row {idx} missing json_callback, skipped.")
            continue

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logging.warning(f"Row {idx} invalid JSON: {json_str}")
            continue

        items = data.get("ItemDetails", {}).get("Items", [])
        if not items:
            logging.info(f"Row {idx} has no items, skipped.")
            continue

        receipt_match = False
        matched_pairs = []

        # Extract all receipt descriptions
        descriptions = [item.get("ReceiptDescription") for item in items if item.get("ReceiptDescription")]

        # ------------------- PROCESS IN BATCHES -------------------
        for batch in chunk_list(descriptions, BATCH_SIZE):
            batch_text = "\n".join([f"- {desc}" for desc in batch])

            for candidate in tqdm(possible_names, desc="Comparing candidates", leave=False):
                prompt = (
                    f"You are given a set of receipt items:\n{batch_text}\n"
                    f"And a candidate product: {candidate}\n"
                    "For each receipt item, decide if it matches the candidate product. "
                    "Answer with JSON: { 'matches': [ { 'item': <desc>, 'match': true/false, 'confidence': <0-1> }, ... ] }"
                )

                try:
                    response = llm.run_inference_json(prompt)
                except Exception as e:
                    logging.warning(f"LLM failed on candidate '{candidate}' for batch: {e}")
                    continue

                if not isinstance(response, dict):
                    continue

                matches = response.get("matches", [])
                for m in matches:
                    if m.get("match") and m.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                        matched_pairs.append({
                            "ReceiptItem": m["item"],
                            "MatchedProduct": candidate,
                            "Confidence": m.get("confidence", 0)
                        })
                        receipt_match = True

        # ------------------- SAVE RECEIPT-LEVEL RESULT -------------------
        result_entry = {
            "ReceiptIndex": idx,
            "Match": "yes" if receipt_match else "no",
            "MatchedPairs": matched_pairs
        }
        results.append(result_entry)
        predictions.append(result_entry)

# ------------------- SAVE RESULTS -------------------
# Prepare method summary
method_summary = {
    "method": "Batch comparison with LLM",
    "description": (
        "Receipt items are processed in batches and compared against all candidate products using the LLM. "
        f"Matches are recorded if confidence >= {CONFIDENCE_THRESHOLD}."
    ),
    "llm_model": "gemma3:4b",
    "batch_size": BATCH_SIZE,
    "sample_size": SAMPLE_SIZE,
    "num_products": len(possible_names),
    "num_receipts": len(results)
}

# Save JSON with results + summary
output_data = {
    "method_summary": method_summary,
    "results": results
}
with open(JSON_FILE, mode="w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# Save CSV predictions
with open(CSV_PREDICTIONS, mode="w", newline="", encoding="utf-8") as f:
    fieldnames = ["ReceiptIndex", "Match", "MatchedPairs"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in predictions:
        row_copy = row.copy()
        row_copy["MatchedPairs"] = json.dumps(row_copy["MatchedPairs"], ensure_ascii=False)
        writer.writerow(row_copy)

# Save plain text summary
with open(SUMMARY_FILE, mode="w", encoding="utf-8") as f:
    f.write("Method summary for receipt-product matching\n")
    f.write(json.dumps(method_summary, ensure_ascii=False, indent=4))

print(f"Results saved in {JSON_FILE}")
print(f"Predictions saved in {CSV_PREDICTIONS}")
print(f"Method summary saved in {SUMMARY_FILE}")
print("Detailed logs in logs.txt")
