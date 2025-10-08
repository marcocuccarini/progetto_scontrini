import csv
import json
import logging
import time
import datetime
import os
from tqdm import tqdm
from classes.LLM import LLM

# ------------------- CONFIG -------------------
CSV_RECEIPTS = "dataset/data.csv"
CSV_PRODUCTS = "dataset/prodotti.csv"
CSV_PREDICTIONS = "receipt_match_pairs.csv"
JSON_FILE = "receipt_match_pairs.json"
NORMALIZED_JSON = "all_normalized_data.json"
BATCH_SIZE_PRODUCTS = 100
CONFIDENCE_THRESHOLD = 0.6
NUM_RECEIPTS = 3

logging.basicConfig(
    filename="logs.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------- INITIALIZE LLM -------------------
llm = LLM(model="gpt-oss:20b", temperature=0.3, max_tokens=512)

# ------------------- UTILS -------------------
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def normalize_receipt_item(item):
    desc = item.get("ReceiptDescription", "")
    price_str = item.get("ItemPrice")
    item_price = None
    if price_str:
        try:
            item_price = float(str(price_str).replace(",", "."))
        except ValueError:
            item_price = None
    prompt = (
        f"Estrai in JSON brand, tipo, quantità, confezione da questo articolo dello scontrino: {desc}.\n"
        f"Includi anche normalized_description e mantieni il prezzo come numero: {item_price}\n"
        f"Rispondi solo in JSON con campi: brand, type, quantity, package, normalized_description, price."
    )
    response = llm.run_inference_json(prompt)
    normalized_item = {
        "brand": response.get("brand") if response else None,
        "type": response.get("type") if response else None,
        "quantity": response.get("quantity") if response else None,
        "package": response.get("package") if response else None,
        "normalized_description": response.get("normalized_description") if response else desc,
        "price": item_price
    }
    return normalized_item

# ------------------- LOAD PRODUCTS -------------------
possible_names = []
product_prices = {}

with open(CSV_PRODUCTS, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        if row.get("nome"):
            possible_names.append(row["nome"])
            if row.get("prezzo"):
                try:
                    price_val = float(str(row["prezzo"]).replace(",", "."))
                    product_prices[row["nome"]] = price_val
                except ValueError:
                    continue

print(f"Loaded {len(possible_names)} products.")

# ------------------- LOAD OR CREATE NORMALIZED DATA -------------------
if os.path.exists(NORMALIZED_JSON):
    print(f"Reading normalized data from {NORMALIZED_JSON}...")
    with open(NORMALIZED_JSON, "r", encoding="utf-8") as f:
        all_normalized_data = json.load(f)
    normalized_products = all_normalized_data.get("products", [])
    normalized_receipts_data = all_normalized_data.get("receipts", [])
else:
    print("Normalizing products and receipts...")
    # --- Normalize products ---
    normalized_products = []
    for prod in tqdm(possible_names, desc="Normalizing products"):
        price = product_prices.get(prod)
        prompt = (
            f"Estrai in JSON brand, tipo, quantità, confezione dal prodotto: {prod}.\n"
            f"Includi anche normalized_description e il prezzo: {price}\n"
            f"Rispondi solo in JSON con campi: brand, type, quantity, package, normalized_description, price."
        )
        response = llm.run_inference_json(prompt)
        normalized_products.append({
            "brand": response.get("brand") if response else None,
            "type": response.get("type") if response else None,
            "quantity": response.get("quantity") if response else None,
            "package": response.get("package") if response else None,
            "normalized_description": response.get("normalized_description") if response else prod,
            "price": price,
            "nome_originale": prod
        })

    # --- Normalize receipts ---
    normalized_receipts_data = []
    with open(CSV_RECEIPTS, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))[:NUM_RECEIPTS]
        for idx, row in enumerate(tqdm(reader, desc="Normalizing receipts")):
            json_str = row.get("json_callback")
            if not json_str:
                continue
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logging.warning(f"JSON decode error in receipt {idx+1}")
                continue
            items = data.get("ItemDetails", {}).get("Items", [])
            normalized_items = [normalize_receipt_item(item) for item in items]
            normalized_receipts_data.append({
                "ReceiptIndex": idx,
                "NormalizedItems": normalized_items
            })

    # --- Save normalized data ---
    all_normalized_data = {
        "products": normalized_products,
        "receipts": normalized_receipts_data
    }
    with open(NORMALIZED_JSON, "w", encoding="utf-8") as f:
        json.dump(all_normalized_data, f, ensure_ascii=False, indent=4)
    print(f"Normalized data saved in {NORMALIZED_JSON}")

# ------------------- PROCESS RECEIPTS AND MATCH -------------------
results = []
predictions = []

for receipt_entry in tqdm(normalized_receipts_data, desc="Processing receipts"):
    idx = receipt_entry["ReceiptIndex"]
    items_info = receipt_entry["NormalizedItems"]
    receipt_match = False
    matched_pairs = []

    for item_ext in tqdm(items_info, desc=f"Matching products in receipt {idx+1}", leave=False):
        best_match_entry = None
        best_confidence = 0.0

        for prod_batch in chunk_list(normalized_products, BATCH_SIZE_PRODUCTS):
            # Filtro preliminare su brand e type
            candidates = [
                p for p in prod_batch
                if (item_ext["brand"] and p["brand"] == item_ext["brand"])
                and (item_ext["type"] and p["type"] == item_ext["type"])
            ]
            if not candidates:
                candidates = prod_batch  # fallback

            prompt_match = (
                f"Confronta il seguente articolo dello scontrino: {json.dumps(item_ext, ensure_ascii=False)}\n"
                f"Con i prodotti candidati: {json.dumps(candidates, ensure_ascii=False)}\n"
                "Restituisci JSON: { 'matches': [ { 'product': <nome_originale>, 'confidence': <0-1> } ] }"
            )
            response_match = llm.run_inference_json(prompt_match)
            matches = response_match.get("matches", []) if isinstance(response_match, dict) else []

            for m in matches:
                try:
                    conf = float(m.get("confidence", 0))
                except (ValueError, TypeError):
                    conf = 0.0
                # bonus se il prezzo corrisponde
                if item_ext["price"]:
                    pred_price = next((p["price"] for p in candidates if p["nome_originale"] == m["product"]), None)
                    if pred_price and abs(pred_price - item_ext["price"]) <= item_ext["price"] * 0.1:
                        conf += 0.2
                if conf > best_confidence:
                    best_confidence = conf
                    best_match_entry = m

        if best_match_entry:
            is_match = best_confidence >= CONFIDENCE_THRESHOLD
            if is_match:
                receipt_match = True
            matched_pairs.append({
                "ReceiptItem": item_ext.get("normalized_description"),
                "ItemPrice": item_ext.get("price"),
                "MatchedProduct": best_match_entry.get("product"),
                "Confidence": best_confidence,
                "Match": "yes" if is_match else "no"
            })
        else:
            matched_pairs.append({
                "ReceiptItem": item_ext.get("normalized_description"),
                "ItemPrice": item_ext.get("price"),
                "MatchedProduct": None,
                "Confidence": 0.0,
                "Match": "no"
            })

    result_entry = {
        "ReceiptIndex": idx,
        "Match": "yes" if receipt_match else "no",
        "MatchedPairs": matched_pairs
    }
    results.append(result_entry)
    predictions.append(result_entry)

# ------------------- SAVE RESULTS -------------------
start_time = time.time()
summary = {
    "method": {
        "phase1": "Normalizzazione attributi con LLM (brand, type, quantity, package, price)",
        "phase2": f"Candidate filtering su attributi normalizzati in batch di {BATCH_SIZE_PRODUCTS}",
        "phase3": "Matching con LLM per confidenza finale",
        "phase4": "Bonus confidenza basato sulla corrispondenza prezzo"
    },
    "parameters": {
        "BATCH_SIZE_PRODUCTS": BATCH_SIZE_PRODUCTS,
        "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "NUM_RECEIPTS": NUM_RECEIPTS
    },
    "LLM": {
        "model": llm.model,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens
    },
    "dataset": {
        "CSV_RECEIPTS": CSV_RECEIPTS,
        "CSV_PRODUCTS": CSV_PRODUCTS,
        "num_receipts_processed": len(results),
        "num_products_loaded": len(possible_names)
    },
    "runtime": {
        "start_time": str(datetime.datetime.now()),
        "end_time": None,
        "total_seconds": None
    }
}

# JSON
output_data = {"summary": summary, "results": results}
with open(JSON_FILE, mode="w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# CSV
with open(CSV_PREDICTIONS, mode="w", newline="", encoding="utf-8") as f:
    fieldnames = ["ReceiptIndex", "Match", "MatchedPairs",
                  "Model", "Temperature", "MaxTokens",
                  "BATCH_SIZE_PRODUCTS", "CONFIDENCE_THRESHOLD", "NUM_RECEIPTS"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in predictions:
        row_copy = row.copy()
        row_copy["MatchedPairs"] = json.dumps(row_copy["MatchedPairs"], ensure_ascii=False)
        row_copy["Model"] = llm.model
        row_copy["Temperature"] = llm.temperature
        row_copy["MaxTokens"] = llm.max_tokens
        row_copy["BATCH_SIZE_PRODUCTS"] = BATCH_SIZE_PRODUCTS
        row_copy["CONFIDENCE_THRESHOLD"] = CONFIDENCE_THRESHOLD
        row_copy["NUM_RECEIPTS"] = NUM_RECEIPTS
        writer.writerow(row_copy)

# Runtime summary
end_time = time.time()
summary["runtime"]["end_time"] = str(datetime.datetime.now())
summary["runtime"]["total_seconds"] = round(end_time - start_time, 2)

with open("method_summary.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(summary, indent=4, ensure_ascii=False))

print(f"Results saved in {JSON_FILE}")
print(f"Predictions saved in {CSV_PREDICTIONS}")
print("Summary saved in method_summary.txt")
print(f"Normalized data saved/used from {NORMALIZED_JSON}")
