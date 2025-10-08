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
TOP_CANDIDATES = 50   # ridotto per efficienza
CONFIDENCE_THRESHOLD = 0.6
NUM_RECEIPTS = 10  # primi N scontrini

logging.basicConfig(filename="logs.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------- INITIALIZE LLM -------------------
llm = LLM(model="gpt-oss:120b", temperature=0.3, max_tokens=512)

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

            # ---------- STEP 1: Interpretazione LLM ----------
            prompt_interpret = f"""
            Analizza il seguente testo preso da uno scontrino:

            \"{desc}\"

            Estrai in JSON le seguenti informazioni se presenti:
            - brand (marca)
            - type (tipo di prodotto, es. latte, bibita, pasta)
            - quantity (es. 1L, 500g, 6x330ml)
            - package (informazioni di confezione)
            - normalized_description (descrizione breve e pulita del prodotto)

            Rispondi solo in JSON.
            """

            response_interpret = llm.run_inference_json(prompt_interpret)
            clean_desc = response_interpret.get("normalized_description", desc) if isinstance(response_interpret, dict) else desc

            # ---------- STEP 2: Pre-filtraggio candidati ----------
            top_candidates = [match[0] for match in process.extract(
                clean_desc, possible_names, scorer=fuzz.WRatio, limit=TOP_CANDIDATES
            )]

            # ---------- STEP 3: Matching con LLM ----------
            prompt_match = (
                f"Abbiamo un prodotto dallo scontrino interpretato così:\n"
                f"{json.dumps(response_interpret, ensure_ascii=False)}\n\n"
                f"Ecco i possibili candidati dal catalogo:\n"
                f"{json.dumps(top_candidates, ensure_ascii=False)}\n\n"
                "Restituisci in JSON:\n"
                "{ 'matches': [ { 'product': <nome prodotto>, 'confidence': <0-1> } ] }"
            )

            response_match = llm.run_inference_json(prompt_match)
            matches = response_match.get("matches", []) if isinstance(response_match, dict) else []

            # ---------- STEP 4: Selezione miglior match ----------
            best_match_entry = None
            best_confidence = 0.0
            for m in matches:
                try:
                    conf = float(m.get("confidence", 0))
                except (ValueError, TypeError):
                    conf = 0.0
                if conf > best_confidence:
                    best_confidence = conf
                    best_match_entry = m

            if best_match_entry:
                is_match = best_confidence >= CONFIDENCE_THRESHOLD
                if is_match:
                    receipt_match = True
                matched_pairs.append({
                    "ReceiptItem": desc,
                    "Normalized": clean_desc,
                    "MatchedProduct": best_match_entry.get("product"),
                    "Confidence": best_confidence,
                    "Match": "yes" if is_match else "no"
                })
            else:
                matched_pairs.append({
                    "ReceiptItem": desc,
                    "Normalized": clean_desc,
                    "MatchedProduct": None,
                    "Confidence": 0.0,
                    "Match": "no"
                })

        # ---------- SAVE receipt-level result ----------
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
        "phase1": "Interpretation of each receipt item using LLM to extract brand, type, quantity, package, and normalized description.",
        "phase2": "Candidate filtering with RapidFuzz (top-N).",
        "phase3": "Matching step with LLM to score candidates.",
        "phase4": "Final selection of best match above confidence threshold.",
    },
    "parameters": {
        "TOP_CANDIDATES": TOP_CANDIDATES,
        "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "NUM_RECEIPTS": NUM_RECEIPTS
    },
    "notes": [
        "Results include raw receipt descriptions and normalized descriptions.",
        "Intermediate interpretations from LLM help improve candidate matching.",
        "Confidence threshold determines whether a product is considered matched."
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
