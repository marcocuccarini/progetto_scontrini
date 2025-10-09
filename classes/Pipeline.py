from datetime import datetime

class PipelineRunner:
    """Orchestratore della pipeline multi-agente (Analista → Critico → Arbitro)."""

    def __init__(self, prompt_engine):
        self.engine = prompt_engine

    def run(self, products, receipt_description, item_name=None, row_idx=None, key=None):
        """Esegue l’intera pipeline su una singola descrizione OCR."""
        results = {}

        # === Analista
        prompt_analista = self.engine.make_prompt_analista(products, receipt_description)
        analista_output, raw1 = self.engine.generate(prompt_analista)
        results["analista_output"] = analista_output

        # === Critico
        prompt_critico = self.engine.make_prompt_critico(receipt_description, analista_output)
        critico_output, raw2 = self.engine.generate(prompt_critico)
        results["critico_output"] = critico_output

        # === Arbitro
        prompt_arbitro = self.engine.make_prompt_arbitro(analista_output, critico_output)
        final_decision, raw3 = self.engine.generate(prompt_arbitro)
        results["final_decision"] = final_decision

        # === Output completo
        results["raw_reasoning"] = {
            "analista": raw1,
            "critico": raw2,
            "arbitro": raw3
        }
        results["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Info di contesto opzionali
        if row_idx is not None:
            results["row_idx"] = int(row_idx)
        if key is not None:
            results["key"] = key
        if item_name:
            results["ItemName"] = item_name
        results["receipt_description"] = receipt_description

        return results
