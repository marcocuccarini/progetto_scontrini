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
        self.client = client or genai.Client()

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
            parsed = self._try_parse_json(text)
            return parsed, text
        except Exception as e:
            return {"error": str(e)}, ""
