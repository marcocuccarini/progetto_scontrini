import json

class PromptEngine:
    """Handles prompt construction for Analista, Critico, and Arbitro."""

    def __init__(self, model):
        self.model = model

    # === Prompt templates ===
    def make_prompt_analista(self, products, receipt_description):
        return f"""
Sei un assistente che deve trovare corrispondenze tra prodotti OCR e prodotti in promozione.
Ritorna un JSON:
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
Valuta il lavoro dell'Analista e restituisci un JSON:
{{"valutazione": "testo sintetico", "affidabilità": float}}
---
Descrizione OCR: {json.dumps(receipt_description, ensure_ascii=False)}
Risultato Analista: {json.dumps(analista_output, ensure_ascii=False)}
"""

    def make_prompt_arbitro(self, analista_output, critico_output):
        return f"""
Decidi la risposta finale basandoti su Analista e Critico.
Ritorna un JSON:
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

    # === Wrappers for execution ===
    def run_analista(self, products, receipt_description):
        prompt = self.make_prompt_analista(products, receipt_description)
        return self.model.generate(prompt)

    def run_critico(self, receipt_description, analista_output):
        prompt = self.make_prompt_critico(receipt_description, analista_output)
        return self.model.generate(prompt)

    def run_arbitro(self, analista_output, critico_output):
        prompt = self.make_prompt_arbitro(analista_output, critico_output)
        return self.model.generate(prompt)
