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

logging.basicConfig(filename="logs.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------- INITIALIZE LLM -------------------
llm = LLM(model="mistral:7b", temperature=0.3, max_tokens=512)

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

        # ------------------- CHECK ALL ITEMS AGAINST ALL CANDIDATES -------------------
        for item in items:
            desc = item.get("ReceiptDescription")
            if not desc:
                continue

            best_match = None
            best_confidence = 0.0

            for candidate in tqdm(possible_names, desc="Comparing candidates", leave=False):
                prompt = (
                    "You are given a product name from a receipt (which may be abbreviated or modified) "
                    f"and a candidate product name: {candidate}\n"
                    f"ReceiptDescription: {desc}\n"
                    "Does the candidate match the receipt description? "
                    "Answer with JSON: { 'match': true/false, 'confidence': <0-1> }"
                )

                response = llm.run_inference_json(prompt)

                confidence = 0.0
                if isinstance(response, dict):
                    match = response.get("match", False)
                    confidence = float(response.get("confidence", 0.0)) if match else 0.0

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = candidate if confidence >= 0.6 else None

            if best_match:
                receipt_match = True
                matched_pairs.append({
                    "ReceiptItem": desc,
                    "MatchedProduct": best_match
                })

        # ------------------- SAVE RECEIPT-LEVEL RESULT -------------------
        result_entry = {
            "ReceiptIndex": idx,
            "Match": "yes" if receipt_match else "no",
            "MatchedPairs": matched_pairs
        }
        results.append(result_entry)
        predictions.append(result_entry)

# ------------------- SAVE RESULTS -------------------
with open(JSON_FILE, mode="w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

with open(CSV_PREDICTIONS, mode="w", newline="", encoding="utf-8") as f:
    fieldnames = ["ReceiptIndex", "Match", "MatchedPairs"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in predictions:
        row_copy = row.copy()
        row_copy["MatchedPairs"] = json.dumps(row_copy["MatchedPairs"], ensure_ascii=False)
        writer.writerow(row_copy)

print(f"Results saved in {JSON_FILE}")
print(f"Predictions saved in {CSV_PREDICTIONS}")
print("Detailed logs in logs.txt")
