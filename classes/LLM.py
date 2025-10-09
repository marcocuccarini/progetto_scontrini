import json
import ollama

class LLM:
    """
    Low-level wrapper for Ollama LLM.
    Handles model loading and JSON inference.
    """

    def __init__(self, model: str = "mistral:7b", temperature: float = 0.3, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            models_list = ollama.list().get("models", [])
            available = [m.get("name") or m.get("model") for m in models_list]
            if self.model not in available:
                print(f"[INFO] Model '{self.model}' not found locally. Pulling...")
                ollama.pull(self.model)
        except Exception as e:
            print(f"[WARN] Could not check local models: {e}")

    def run_inference(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a reasoning assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": self.temperature, "num_predict": self.max_tokens}
            )
            return response["message"]["content"]
        except Exception as e:
            return json.dumps({"error": str(e)})

    def run_inference_json(self, prompt: str) -> dict:
        raw_output = self.run_inference(prompt)
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {"raw_output": raw_output}
