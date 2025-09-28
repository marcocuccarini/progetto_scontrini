import json
import ollama

class LLM:
    """
    Wrapper di basso livello per Ollama LLM.
    Gestisce caricamento del modello e inferenza raw.
    """

    def __init__(self, model: str = "mistral", temperature: float = 0.7, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Verifica se il modello è disponibile localmente
        try:
            available = [m["name"] for m in ollama.list()["models"]]
            if self.model not in available:
                print(f"[INFO] Modello '{self.model}' non trovato localmente. Pulling...")
                ollama.pull(self.model)
        except Exception as e:
            print(f"[WARN] Impossibile verificare i modelli locali: {e}")

    def run_inference(self, prompt: str) -> str:
        """
        Esegue un prompt sul modello e restituisce il testo raw.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a reasoning assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"[ERROR] Inference fallita: {e}")
            return json.dumps({"error": str(e)})

    def run_inference_json(self, prompt: str) -> dict:
        """
        Esegue un prompt e restituisce un JSON valido come dict Python.
        """
        raw_output = self.run_inference(prompt)
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            print(f"[WARN] Output non è JSON valido: {raw_output}")
            return {"raw_output": raw_output}
