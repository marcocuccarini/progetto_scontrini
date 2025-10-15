import json

# Carica il JSON da un file
with open("matches.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Lista dove metteremo i dati estratti
extracted_data = []

for entry in data:
    item = {}
    # Estrai ItemName e receipt_description
    item['ItemName'] = entry.get('ItemName', '')
    item['receipt_description'] = entry.get('receipt_description', '')
    
    # Estrai al massimo i primi 3 match finali se esistono
    match_finale = entry.get('final_decision', {}).get('match_finale', [])
    top_matches = match_finale[:3]  # Prendi al massimo 3 elementi
    
    # Salva solo nome, id e confidenza per i match
    item['top_matches'] = [
        {'id': m.get('id', ''), 'nome': m.get('nome', ''), 'confidenza': m.get('confidenza', '')} 
        for m in top_matches
    ]
    
    extracted_data.append(item)

# Stampa risultato
for e in extracted_data:
    print(json.dumps(e, indent=2))
