import requests
import logging
import os

class OpenRouterAPI:
    """
    Simple wrapper for calling a model via OpenRouter's unified API.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key=None, model=None):
        """
        Initialize the OpenRouter API client.
        
        Args:
            api_key (str, optional): OpenRouter API key. If not provided, will try to get from environment.
            model (str, optional): Model to use for completion. Example: "openai/gpt-4o"
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model

    def send_message(self, messages):
        """
        Send a message to the model and get a response.
        
        Args:
            messages: list of {"role": "user"|"assistant"|"system", "content": "string"}
            
        Returns:
            The last assistant message as a string.
        """
        if not self.api_key:
            logging.error("No OpenRouter API key provided")
            return "Error: No API key provided. Please check your configuration."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional: identify your app for OpenRouter's ranking
            "HTTP-Referer": "https://replit.com/bart-experiment",
            "X-Title": "BART LLM Experiment"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False  # For simplicity, we do a standard (non-streaming) request
        }

        try:
            resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # Each "choice" is a potential completion. We typically use choice[0].
            # We assume a normal non-streaming response => 'message' field is present.
            choices = data.get("choices", [])
            if not choices:
                logging.warning("No completion returned from API. data=%s", data)
                return ""

            return choices[0]["message"]["content"] or ""

        except requests.RequestException as e:
            logging.error(f"OpenRouter API call failed: {e}")
            return f"Error communicating with API: {str(e)}"
