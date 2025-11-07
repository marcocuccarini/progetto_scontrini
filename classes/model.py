import os
import re
import json
import google.generativeai as genai

class LLMModel:
    """
    Wrapper around Gemini model.
    """

    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ Manca la variabile d'ambiente GEMINI_API_KEY. "
                             "Imposta la chiave con:\n\n"
                             "export GEMINI_API_KEY=LA_TUA_KEY\n")

        # ✅ Correct client initialization for new API
        genai.configure(api_key=api_key)

        # ✅ Load model object directly
        self.model = genai.GenerativeModel(self.model_name)

    def _clean_response(self, text: str):
        """Remove ```json fences or leftover wrapper text."""
        return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    def _try_parse_json(self, text: str):
        cleaned = self._clean_response(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {"error": f"Could not parse JSON: {e}", "raw": text}

    def generate(self, prompt: str):
        """
        Send prompt → return (parsed_output, raw_text)
        """
        try:
            # ✅ New SDK call method
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Detect free usage warnings hidden inside output
            lower = text.lower()
            if ("free" in lower and "limit" in lower) or ("quota" in lower) or ("upgrade" in lower):
                raise Exception("GEMINI_FREE_LIMIT_REACHED")

            parsed = self._try_parse_json(text)
            return parsed, text

        except Exception as e:
            raise e
