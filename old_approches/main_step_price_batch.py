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
BATCH_SIZE_RECEIPT = 50  # number of items per batch for LLM
BATCH_SIZE_PRODUCTS = 100  # number of products per batch for fuzzy matching
TOP_CANDIDATES = 50
CONFIDENCE_THRESHOLD = 0.6
NUM_RECEIPTS = 10  # primi N scontrini

logging.basicConfig(
    filename="logs.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------- INITIALIZE LLM -------------------
llm = LLM(model="gpt-oss:120b", temperature=0.3, max_tokens=512)

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

results = []
predictions = []

# ------------------- UTILS -------------------
def filter_by_price(candidates, target_price, tolerance=0.1):
    if target_price is None:
        return candidates
    filtered = []
    for c in candidates:
        p_price = product_prices.get(c)
        if p_price is None:
            continue
        if abs(p_price - target_price) <= target_price * tolerance:
            filtered.append(c)
    return filtered if filtered else candidates


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batch_fuzzy_match_all(query, candidates, batch_size=100, top_k=10):
    """Fuzzy match across all candidates in batches and return top_k matches globally."""
    all_matches = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        matches = process.extract(query, batch, scorer=fuzz.WRatio, limit=top_k)
        all_matches.extend(matches)
    all_matches.sort(key=lambda x: x[1], reverse=True)
    return all_matches[:top_k]


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

        # Extract all items with description and price
        items_info = []
        for item in items:
            desc = item.get("ReceiptDescription")
            price_str = item.get("ItemPrice")
            item_price = None
            if price_str is not None:
                try:
                    item_price = float(str(price_str).replace(",", "."))
                except ValueError:
                    item_price = None
            if desc:
                items_info.append({"desc": desc, "price": item_price})

        # ------------------- PROCESS ITEMS IN BATCH -------------------
        for batch in tqdm(chunk_list(items_info, BATCH_SIZE_RECEIPT),
                          desc=f"Items in receipt {idx+1}", leave=False):
            batch_text = "\n".join([f"- Description: {i['desc']} | Price: {i['price']}" for i in batch])
            prompt_interpret = f"""
            Analyze the following batch of receipt items and extract JSON info:
            {batch_text}
            For each item, extract:
            - brand
            - type
            - quantity
            - package
            - price
            - normalized_description
            Return JSON: {{'items': [ ... ]}}
            """

            response_interpret = llm.run_inference_json(prompt_interpret)
            items_extracted = response_interpret.get("items", []) if isinstance(response_interpret, dict) else batch

            for item_ext in tqdm(items_extracted,
                                 desc=f"Matching products in receipt {idx+1}", leave=False):
                clean_desc = item_ext.get("normalized_description", item_ext.get("desc"))
                item_price = item_ext.get("price")

                # ------------------- BATCH PRODUCT MATCHING -------------------
                top_matches = batch_fuzzy_match_all(clean_desc, possible_names,
                                                    batch_size=BATCH_SIZE_PRODUCTS,
                                                    top_k=TOP_CANDIDATES)
                top_candidates = [match[0] for match in top_matches]
                top_candidates = filter_by_price(top_candidates, item_price)

                prompt_match = (
                    f"Receipt item info: {json.dumps(item_ext, ensure_ascii=False)}\n"
                    f"Candidate products: {json.dumps(top_candidates, ensure_ascii=False)}\n"
                    "Return JSON: { 'matches': [ { 'product': <name>, 'confidence': <0-1> } ] }"
                )

                response_match = llm.run_inference_json(prompt_match)
                matches = response_match.get("matches", []) if isinstance(response_match, dict) else []

                best_match_entry = None
                best_confidence = 0.0
                for m in matches:
                    try:
                        conf = float(m.get("confidence", 0))
                    except (ValueError, TypeError):
                        conf = 0.0
                    if item_price and product_prices.get(m.get("product")):
                        pred_price = product_prices[m["product"]]
                        if abs(pred_price - item_price) <= item_price * 0.1:
                            conf += 0.2
                    if conf > best_confidence:
                        best_confidence = conf
                        best_match_entry = m

                if best_match_entry:
                    is_match = best_confidence >= CONFIDENCE_THRESHOLD
                    if is_match:
                        receipt_match = True
                    matched_pairs.append({
                        "ReceiptItem": item_ext.get("desc"),
                        "Normalized": clean_desc,
                        "ItemPrice": item_price,
                        "MatchedProduct": best_match_entry.get("product"),
                        "Confidence": best_confidence,
                        "Match": "yes" if is_match else "no"
                    })
                    # -------- PRINT MATCH --------
                    print(f"Receipt {idx+1} | Item: '{item_ext.get('desc')}' ({item_price}) "
                          f"→ Matched: '{best_match_entry.get('product')}' | "
                          f"Confidence: {best_confidence:.2f} | "
                          f"{'yes' if is_match else 'no'}")
                else:
                    matched_pairs.append({
                        "ReceiptItem": item_ext.get("desc"),
                        "Normalized": clean_desc,
                        "ItemPrice": item_price,
                        "MatchedProduct": None,
                        "Confidence": 0.0,
                        "Match": "no"
                    })
                    # -------- PRINT NO MATCH --------
                    print(f"Receipt {idx+1} | Item: '{item_ext.get('desc')}' ({item_price}) "
                          f"→ No match found")

        # ------------------- SAVE receipt-level result -------------------
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
        "phase1": "Batch interpretation of receipt items using LLM to extract brand, type, quantity, package, price, normalized description.",
        "phase2": "Candidate filtering with RapidFuzz in batches and optional price filtering.",
        "phase3": "Matching with LLM to assign confidence scores.",
        "phase4": "Final confidence adjustment based on price proximity bonus."
    },
    "parameters": {
        "BATCH_SIZE_RECEIPT": BATCH_SIZE_RECEIPT,
        "BATCH_SIZE_PRODUCTS": BATCH_SIZE_PRODUCTS,
        "TOP_CANDIDATES": TOP_CANDIDATES,
        "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "NUM_RECEIPTS": NUM_RECEIPTS
    }
}

# ------------------- SAVE RESULTS -------------------
output_data = {"summary": summary, "results": results}
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
