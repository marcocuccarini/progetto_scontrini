import os
import re
import json
from google import genai

class LLMModel:
    """
    Wrapper around a Gemini (or other LLM) model.
    Responsible for sending prompts and returning raw + parsed outputs.
    """

    def __init__(self, model_name="gemini-2.5-flash", client=None):
        self.model_name = model_name

        # === Load API Key from ENV ===
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ Manca la variabile d'ambiente GEMINI_API_KEY. "
                             "Imposta la chiave con:\n\n"
                             "export GEMINI_API_KEY=LA_TUA_KEY\n")

        # Initialize Gemini client
        self.client = client or genai.Client(api_key=api_key)

    def _clean_response(self, text: str):
        """Remove markdown fences and trailing junk before JSON parsing."""
        return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    def _try_parse_json(self, text: str):
        """Try to parse model output as JSON, return raw text on failure."""
        cleaned = self._clean_response(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {"error": f"Could not parse JSON: {e}", "raw": text}

    def generate(self, prompt: str):
        """
        Send a prompt to the model and return both parsed and raw outputs.
        Returns: (parsed_dict, raw_text)
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            text = response.text.strip()

            # === Detect free usage limit inside normal response text ===
            lower = text.lower()
            if ("free" in lower and "limit" in lower) or ("quota" in lower) or ("upgrade" in lower):
                raise Exception("GEMINI_FREE_LIMIT_REACHED")

            parsed = self._try_parse_json(text)
            return parsed, text

        except Exception as e:
            # Let the caller handle limit stop
            raise e
