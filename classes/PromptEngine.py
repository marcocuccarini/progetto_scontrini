import json
import re
from google import genai

class PromptEngine:
    """Responsabile della generazione dei prompt e dell'interazione con il modello Gemini."""

    def __init__(self, client=None, model="gemini-2.5-flash"):
        self.client = client or genai.Client()
        self.model = model

    @staticmethod
    def parse_output(raw_output: str):
        """Pulisce e interpreta l’output JSON restituito da Gemini."""
        if not raw_output:
            return {"error": "Empty response"}
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {"error": f"Could not parse Gemini output: {e}", "raw": raw_output}

    def generate(self, prompt: str):
        """Genera contenuto con il modello Gemini e restituisce output raw + parsed."""
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            text = response.text.strip()
            parsed = self.parse_output(text)
            return parsed, text
        except Exception as e:
            return {"error": str(e)}, ""

    # === Prompt templates ===
    def make_prompt_analista(self, products, receipt_description):
        return f"""
Sei un assistente che deve trovare corrispondenze tra prodotti OCR da scontrini e una lista di prodotti in promozione. 
Ti verrà fornita una lista di prodotti in offerta e una descrizione OCR (campo "ReceiptDescription").
Il tuo compito è:
1. Trovare i tre prodotti più probabili corrispondenti.
2. Restituire un JSON del tipo:
[
  {{"id": ..., "nome": ..., "brand": ..., "confidenza": float, "spiegazione": "..."}}
]
---
Lista prodotti:
{json.dumps(products, ensure_ascii=False)}
---
Descrizione OCR:
{json.dumps(receipt_description, ensure_ascii=False)}
"""

    def make_prompt_critico(self, receipt_description, analista_output):
        return f"""
Sei un revisore critico che valuta il lavoro dell'agente "Analista".
Restituisci un JSON del tipo:
{{ "valutazione": "testo sintetico", "affidabilità": float }}
---
Descrizione OCR: {json.dumps(receipt_description, ensure_ascii=False)}
Risultato Analista: {json.dumps(analista_output, ensure_ascii=False)}
"""

    def make_prompt_arbitro(self, analista_output, critico_output):
        return f"""
Sei l'agente finale ("Arbitro") che decide sulla base del lavoro dell'Analista e del Critico.
Restituisci un JSON del tipo:
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
