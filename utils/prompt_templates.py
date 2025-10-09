import json

def analista_prompt(receipt_description: str, products: list) -> str:
    return f"""
Sei un assistente che deve trovare corrispondenze tra prodotti OCR e una lista di prodotti.
Lista prodotti: {json.dumps(products, ensure_ascii=False)}
Descrizione OCR: {json.dumps(receipt_description, ensure_ascii=False)}

Compito:
1. Restituire i 3 migliori match con id, nome, brand, confidenza (0.0-1.0) e spiegazione sintetica.
2. Se non ci sono match affidabili (confidenza <0.4), specificare "Nessuna corrispondenza affidabile".
Rispondi solo in JSON.
"""

def critico_prompt(receipt_description: str, analista_output: dict) -> str:
    return f"""
Sei un revisore critico.
OCR: {json.dumps(receipt_description, ensure_ascii=False)}
Output Analista: {json.dumps(analista_output, ensure_ascii=False)}

Compito:
1. Valutare coerenza e affidabilità dei match.
2. Evidenziare errori o eccessiva sicurezza.
3. Restituire JSON con:
   {{ "valutazione": "testo sintetico", "affidabilità": float }}
"""

def arbitro_prompt(analista_output: dict, critico_output: dict) -> str:
    return f"""
Sei l'Arbitro.
Output Analista: {json.dumps(analista_output, ensure_ascii=False)}
Output Critico: {json.dumps(critico_output, ensure_ascii=False)}

Compito:
1. Accettare, rifiutare o revisionare i match.
2. Restituire JSON con:
   {{ "decision": "accettata|rifiutata|revisionata",
      "motivazione": "testo sintetico",
      "match_finale": [{"id": "...", "nome": "...", "brand": "...", "confidenza": 0.0}],
      "affidabilità_finale": float }}
"""
