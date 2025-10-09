import os
import re
import json
import time
import hashlib
import pandas as pd
from datetime import datetime
from google import genai

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

client = genai.Client()  # GEMINI_API_KEY deve essere impostata nell'ambiente

# === Load CSVs ===
products_df = pd.read_csv(products_file, sep=";", dtype=str).fillna("")
ocr_df = pd.read_csv(ocr_file, sep=";", dtype=str).fillna("")
products = products_df.to_dict(orient="records")

# === helper: extract items from nested JSON columns ===
def extract_items_from_row(row):
    items_list = []
    for col in row.index:
        if any(k in col.lower() for k in ["json", "callback", "data"]):
            raw = row[col]
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                json_data = json.loads(raw)
            except Exception:
                try:
                    json_data = json.loads(json.loads(raw))
                except Exception:
                    continue

            items = json_data.get("ItemDetails", {}).get("Items", [])
            for item in items:
                desc = item.get("ReceiptDescription", "").strip()
                name = item.get("ItemName", "").strip()
                if desc:
                    items_list.append({"ReceiptDescription": desc, "ItemName": name})
    return items_list

# === helper: compute resume key ===
def compute_key(row_idx, receipt_description):
    key_str = f"{row_idx}|{receipt_description}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

# === resume handling ===
def load_run_state():
    if os.path.exists(RUN_STATE_FILE):
        try:
            with open(RUN_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load run state: {e}")
    return {"last_row_idx": -1, "processed_keys": []}

def save_run_state(last_row_idx, processed_keys):
    state = {
        "last_row_idx": last_row_idx,
        "processed_keys": list(processed_keys)
    }
    with open(RUN_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    print(f"💾 Run state saved (row={last_row_idx}, processed={len(processed_keys)})")

run_state = load_run_state()
processed_keys = set(run_state.get("processed_keys", []))
start_idx = run_state.get("last_row_idx", -1) + 1

print(f"🔁 Resuming from row {start_idx} with {len(processed_keys)} previously processed items")

# === helper: append a record atomically ===
def append_record_ndjson(record):
    line = json.dumps(record, ensure_ascii=False)
    with open(OUTPUT_NDJSON, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

# === helper: write full snapshot ===
def write_snapshot():
    snapshot = []
    if os.path.exists(OUTPUT_NDJSON):
        with open(OUTPUT_NDJSON, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    snapshot.append(rec)
                except Exception:
                    continue
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
        json.dump(snapshot, out_f, indent=2, ensure_ascii=False)
    print(f"💾 Snapshot saved to {OUTPUT_JSON} ({len(snapshot)} records)")

# === helper: parse Gemini output ===
def parse_gemini_output(raw_output):
    if not raw_output:
        return {"error": "Empty response"}
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"error": f"Could not parse Gemini output: {e}", "raw": raw_output}

# === MAIN processing loop ===
appended_since_snapshot = 0
processed_count = len(processed_keys)

try:
    for idx, row in ocr_df.iterrows():
        if idx < start_idx:
            continue  # Skip rows already processed

        items = extract_items_from_row(row)
        if not items:
            print(f"⚠️  No items found in row {idx}. Skipping.")
            continue

        for item in items:
            receipt_description = item["ReceiptDescription"]
            item_name_csv = item["ItemName"]

            key = compute_key(idx, receipt_description)
            if key in processed_keys:
                print(f"⏭ Skipping already processed row {idx} (key {key[:8]})")
                continue

            print("\n=============================================")
            print(f"🧾 Row {idx} | Receipt: {receipt_description}")
            print("=============================================\n")

            # === AGENTE 1: Analista ===
            prompt_analista = f"""
Sei un assistente che deve trovare corrispondenze tra prodotti OCR da scontrini e una lista di prodotti in promozione. 
Ti verrà fornita una lista di prodotti in offerta, ognuno con questi campi:
- id
- brand
- nome
- nome_normalizzato (può essere vuoto)
- ean

Ti verrà anche fornita una descrizione di un prodotto letta da uno scontrino (campo "ReceiptDescription"), 
che può contenere abbreviazioni, punteggiatura, o errori OCR.

Il tuo compito è:
1. Cercare il prodotto nella lista che più probabilmente corrisponde alla descrizione OCR.
2. Restituire i tre migliori match, ciascuno con:
   - id
   - nome
   - brand
   - punteggio di confidenza (da 0.0 a 1.0)
   - spiegazione sintetica del perché ritieni ci sia una corrispondenza.
3. Se non c’è nessuna corrispondenza credibile (es. confidenza < 0.4), specifica "Nessuna corrispondenza affidabile".
4. Devi essere abbastanza critico, considera anche le quantità che posso essere un buon indicatore.
5. Rispondi **solo in formato JSON**.
---
Ecco la lista dei prodotti in promozione:
{json.dumps(products, ensure_ascii=False)}
---
Descrizione OCR:
{json.dumps(receipt_description, ensure_ascii=False)}
"""
            try:
                r1 = client.models.generate_content(model="gemini-2.5-flash", contents=prompt_analista)
                analista_output = parse_gemini_output(r1.text.strip())
            except Exception as e:
                analista_output = {"error": f"Analista error: {str(e)}"}
                r1 = None

            print("\n🧠 [Analista reasoning raw output]:")
            if r1: print(r1.text.strip())

            # === AGENTE 2: Critico ===
            prompt_critico = f"""
Sei un revisore critico che valuta il lavoro dell'agente "Analista".
Ti vengono forniti:
- La descrizione OCR
- Il risultato dell'Analista (match, spiegazioni, punteggi)

Il tuo compito è:
1. Valutare se le corrispondenze e le spiegazioni sono coerenti e affidabili.
2. Evidenziare eventuali errori, omissioni o eccessiva sicurezza.
3. Assegnare un punteggio di affidabilità complessiva (0.0 - 1.0).
4. Devi essere severo nella valutazione. 
5. Restituire un JSON del tipo:
{{ "valutazione": "testo sintetico", "affidabilità": float }}
---
Descrizione OCR: {json.dumps(receipt_description, ensure_ascii=False)}
Risultato Analista: {json.dumps(analista_output, ensure_ascii=False)}
"""
            try:
                r2 = client.models.generate_content(model="gemini-2.5-flash", contents=prompt_critico)
                critico_output = parse_gemini_output(r2.text.strip())
            except Exception as e:
                critico_output = {"error": f"Critico error: {str(e)}"}
                r2 = None

            print("\n🧠 [Critico reasoning raw output]:")
            if r2: print(r2.text.strip())

            # === AGENTE 3: Arbitro ===
            prompt_arbitro = f"""
Sei l'agente finale ("Arbitro") che deve decidere sulla base del lavoro degli agenti precedenti.
Hai:
- L'output dell'Analista
- La revisione critica del Critico

Il tuo compito è:
1. Decidere se accettare, rifiutare o revisionare la decisione dell'Analista.
2. Produrre un JSON finale con la struttura:
{{
  "decision": "accettata" | "rifiutata" | "revisionata",
  "motivazione": "testo sintetico",
  "match_finale": [{{"id": ..., "nome": ..., "brand": ..., "confidenza": ...}}],
  "affidabilità_finale": float
}}
---
Analista: {json.dumps(analista_output, ensure_ascii=False)}
Critico: {json.dumps(critico_output, ensure_ascii=False)}
"""
            try:
                r3 = client.models.generate_content(model="gemini-2.5-flash", contents=prompt_arbitro)
                final_decision = parse_gemini_output(r3.text.strip())
            except Exception as e:
                final_decision = {"error": f"Arbitro error: {str(e)}"}
                r3 = None

            print("\n🧠 [Arbitro reasoning raw output]:")
            if r3: print(r3.text.strip())

            # === SALVATAGGIO COMPLETO ===
            record = {
                "row_idx": int(idx),
                "key": key,
                "receipt_description": receipt_description,
                "ItemName": item_name_csv,
                "analista_output": analista_output,
                "critico_output": critico_output,
                "final_decision": final_decision,
                "raw_reasoning": {
                    "analista": r1.text.strip() if r1 else "",
                    "critico": r2.text.strip() if r2 else "",
                    "arbitro": r3.text.strip() if r3 else ""
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            append_record_ndjson(record)
            processed_keys.add(key)
            appended_since_snapshot += 1
            processed_count += 1
            save_run_state(idx, processed_keys)
            print(f"\n✅ Row {idx} processed (key {key[:8]})")

            # === FILE JSON SEMPLIFICATO ===
            simplified_entry = {
                "descrizione": receipt_description,
                "match": final_decision.get("match_finale", [])
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

# === final snapshot ===
write_snapshot()
save_run_state(idx, processed_keys)
print("\n✅ Processing complete.")
