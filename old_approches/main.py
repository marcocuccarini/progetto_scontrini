import csv
import json
import logging
from rapidfuzz import process, fuzz
from tqdm import tqdm
from classes.LLM import LLM

# ------------------- CONFIG -------------------
CSV_RECEIPTS = "dataset/data.csv"
CSV_PRODUCTS = "dataset/prodotti.csv"
CSV_PREDICTIONS = "receipt_match_pairs.csv"
JSON_FILE = "receipt_match_pairs.json"
TOP_CANDIDATES = 250
CONFIDENCE_THRESHOLD = 0.6
NUM_RECEIPTS = 10  # first N receipts

logging.basicConfig(filename="logs.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------- INITIALIZE LLM -------------------
llm = LLM(model="gpt-oss:20b", temperature=0.3, max_tokens=512)

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

    reader = reader[:NUM_RECEIPTS]

    for idx, row in enumerate(tqdm(reader, desc="Processing receipts")):
        json_str = row.get("json_callback")
        if not json_str:
            continue

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        items = data.get("ItemDetails", {}).get("Items", [])
        if not items:
            continue

        receipt_match = False
        matched_pairs = []

        for item in items:
            desc = item.get("ReceiptDescription")
            if not desc:
                continue

            # Pre-filter top candidates
            top_candidates = [match[0] for match in process.extract(
                desc, possible_names, scorer=fuzz.WRatio, limit=TOP_CANDIDATES
            )]

            # LLM prompt
            prompt = (
                f"ReceiptDescription: {desc}\n"
                f"Candidate products: {json.dumps(top_candidates, ensure_ascii=False)}\n"
                "Return JSON: { 'matches': [ { 'product': <name>, 'confidence': <0-1> } ] }. "
                "If none match, return empty list."
            )

            response = llm.run_inference_json(prompt)
            matches = response.get("matches", []) if isinstance(response, dict) else []

            # Determine best match (even if below threshold)
            best_match_entry = None
            best_confidence = 0.0
            for m in matches:
                conf = float(m.get("confidence", 0))
                if conf > best_confidence:
                    best_confidence = conf
                    best_match_entry = m

            if best_match_entry:
                is_match = best_confidence >= CONFIDENCE_THRESHOLD
                if is_match:
                    receipt_match = True
                matched_pairs.append({
                    "ReceiptItem": desc,
                    "MatchedProduct": best_match_entry["product"],
                    "Confidence": best_confidence,
                    "Match": "yes" if is_match else "no"
                })
            else:
                # No candidate returned by LLM
                matched_pairs.append({
                    "ReceiptItem": desc,
                    "MatchedProduct": None,
                    "Confidence": 0.0,
                    "Match": "no"
                })

        # Save receipt-level result
        result_entry = {
            "ReceiptIndex": idx,
            "Match": "yes" if receipt_match else "no",
            "MatchedPairs": matched_pairs
        }
        results.append(result_entry)
        predictions.append(result_entry)

# ------------------- SUMMARY -------------------
summary = {
    "method": {
        "phase1": "Extract candidate matches for each receipt item with RapidFuzz (top-N).",
        "phase2": "Use LLM to score candidates based on similarity and context.",
        "phase3": "Select best match above confidence threshold.",
    },
    "parameters": {
        "TOP_CANDIDATES": TOP_CANDIDATES,
        "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "NUM_RECEIPTS": NUM_RECEIPTS
    },
    "notes": [
        "This version skips price-based filtering.",
        "Results include raw receipt description and best matched product.",
        "Confidence threshold determines whether a match is accepted."
    ]
}

# ------------------- SAVE RESULTS -------------------
output_data = {
    "summary": summary,
    "results": results
}

with open(JSON_FILE, mode="w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

with open(CSV_PREDICTIONS, mode="w", newline="", encoding="utf-8") as f:
    fieldnames = ["ReceiptIndex", "Match", "MatchedPairs"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in predictions:
        row_copy = row.copy()
        row_copy["MatchedPairs"] = json.dumps(row_copy["MatchedPairs"], ensure_ascii=False)
        writer.writerow(row_copy)

with open("method_summary.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(summary, indent=4, ensure_ascii=False))

print(f"Results saved in {JSON_FILE}")
print(f"Predictions saved in {CSV_PREDICTIONS}")
print("Summary saved in method_summary.txt")
