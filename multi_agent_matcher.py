import os
import json
import pandas as pd
from classes.LLM import LLM
from agents.base import Agent
from agents import prompt_templates as pt
from utils.helpers import compute_key, append_record_ndjson, write_snapshot

# === Config ===
PRODUCTS_FILE = "data/prodotti.csv"
OCR_FILE = "data/data.csv"
OUTPUT_NDJSON = "results/matches.ndjson"
OUTPUT_JSON = "results/matches.json"
OUTPUT_SIMPLE_JSON = "results/predizioni_simplified.json"

os.makedirs("results", exist_ok=True)

# === Load CSV ===
products_df = pd.read_csv(PRODUCTS_FILE, sep=";", dtype=str).fillna("")
ocr_df = pd.read_csv(OCR_FILE, sep=";", dtype=str).fillna("")
products = products_df.to_dict(orient="records")

# === Initialize LLM and agents ===
llm = LLM(model="mistral:7b")
analista_agent = Agent(llm)
critico_agent = Agent(llm)
arbitro_agent = Agent(llm)

# === Process OCR items ===
processed_keys = set()
simple_data = []

for idx, row in ocr_df.iterrows():
    receipt_description = row.get("ReceiptDescription", "")
    if not receipt_description:
        continue

    key = compute_key(idx, receipt_description)
    if key in processed_keys:
        continue

    # Analista
    analista_output = analista_agent.run(pt.analista_prompt(receipt_description, products))

    # Critico
    critico_output = critico_agent.run(pt.critico_prompt(receipt_description, analista_output))

    # Arbitro
    final_decision = arbitro_agent.run(pt.arbitro_prompt(analista_output, critico_output))

    # Salvataggio NDJSON
    record = {
        "row_idx": idx,
        "key": key,
        "receipt_description": receipt_description,
        "analista_output": analista_output,
        "critico_output": critico_output,
        "final_decision": final_decision
    }
    append_record_ndjson(record, OUTPUT_NDJSON)
    processed_keys.add(key)

    # Salvataggio JSON semplificato
    simple_data.append({
        "descrizione": receipt_description,
        "match": final_decision.get("match_finale", [])
    })
    with open(OUTPUT_SIMPLE_JSON, "w", encoding="utf-8") as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)

# Snapshot finale
write_snapshot(OUTPUT_NDJSON, OUTPUT_JSON)
print("✅ Processing complete.")
