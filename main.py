import csv
import json
from classes.llm import LLM

# Percorsi dei file
csv_receipts = "dataset/data.csv"           # scontrini
csv_products = "dataset/prodotti.csv"      # lista prodotti
json_file = "output.json"                   # output JSON
csv_predictions = "predizioni.csv"         # nuovo file CSV con predizioni

# Inizializza il modello LLM
llm = LLM(model="mistral", temperature=0.3, max_tokens=512)

# Costruisci la lista dei nomi dei prodotti dal dataset
possible_names = []
with open(csv_products, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        if row['nome']:
            possible_names.append(row['nome'])

print(f"Caricati {len(possible_names)} prodotti.")

results = []
predictions = []

# Leggi il CSV degli scontrini
with open(csv_receipts, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    
    if "json_callback" not in reader.fieldnames:
        print("Colonna 'json_callback' non trovata nel CSV")
    else:
        for row in reader:
            try:
                data = json.loads(row["json_callback"])
                if "ReceiptDescription" in data:
                    desc = data["ReceiptDescription"]

                    # Prompt per fuzzy matching
                    prompt = (
                        "Il seguente testo è un nome prodotto preso da uno scontrino, "
                        "che può essere accorciato, abbreviato o riformulato.\n"
                        f"ReceiptDescription: {desc}\n"
                        f"Lista di nomi candidati: {json.dumps(possible_names, ensure_ascii=False)}\n"
                        "Trova il miglior match tra i nomi candidati. Restituisci un JSON del tipo: "
                        "{ 'best_match': <nome o null>, 'confidence': <valore tra 0 e 1> }"
                    )

                    response = llm.run_inference_json(prompt)
                    results.append({
                        "ReceiptDescription": desc,
                        "LLM_Response": response
                    })

                    # Salva per il CSV delle predizioni
                    predicted_name = response.get('best_match', None)
                    predictions.append({
                        "ReceiptDescription": desc,
                        "PredictedProduct": predicted_name
                    })
            except json.JSONDecodeError:
                print("Valore non valido in json_callback:", row["json_callback"])

# Salva i risultati in JSON
with open(json_file, mode="w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Salva le predizioni in CSV
with open(csv_predictions, mode="w", newline="", encoding="utf-8") as f:
    fieldnames = ["ReceiptDescription", "PredictedProduct"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in predictions:
        writer.writerow(row)

print(f"Risultati salvati in {json_file}")
print(f"Predizioni salvate in {csv_predictions}")
