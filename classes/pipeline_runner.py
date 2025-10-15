from datetime import datetime

class PipelineRunner:
    """Coordinates the Analista → Critico → Arbitro pipeline."""

    def __init__(self, prompt_engine):
        self.engine = prompt_engine

    def run(self, products, receipt_description, item_name=None, row_idx=None, key=None):
        results = {}

        # Analista
        analista_output, raw1 = self.engine.run_analista(products, receipt_description)
        results["analista_output"] = analista_output

        # Critico
        critico_output, raw2 = self.engine.run_critico(receipt_description, analista_output)
        results["critico_output"] = critico_output

        # Arbitro
        final_decision, raw3 = self.engine.run_arbitro(analista_output, critico_output)
        results["final_decision"] = final_decision

        # Trace & metadata
        results["raw_reasoning"] = {"analista": raw1, "critico": raw2, "arbitro": raw3}
        results["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if row_idx is not None:
            results["row_idx"] = int(row_idx)
        if key is not None:
            results["key"] = key
        if item_name:
            results["ItemName"] = item_name
        results["receipt_description"] = receipt_description

        return results
