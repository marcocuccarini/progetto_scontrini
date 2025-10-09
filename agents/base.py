from classes.LLM import LLM
import json

class Agent:
    """
    Base class for all agents (Analista, Critico, Arbitro)
    """

    def __init__(self, llm_model: LLM):
        self.llm = llm_model

    def run(self, prompt: str) -> dict:
        """
        Runs the agent prompt through the LLM and returns JSON.
        """
        return self.llm.run_inference_json(prompt)
