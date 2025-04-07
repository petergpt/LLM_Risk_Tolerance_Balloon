import requests
import logging
import os
import time

class OpenRouterAPI:
    """
    Simple wrapper for calling a model via OpenRouter's unified API.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key=None, model=None, debug=False):
        """
        Initialize the OpenRouter API client.

        Args:
            api_key (str, optional): OpenRouter API key.
            model (str, optional): Model to use for completion (e.g. "openai/gpt-4").
            debug (bool): If True, log full request/response payloads for debugging.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.debug = debug

    def send_message(self, messages):
        """
        Send a message to the model with built-in retry logic
        to handle rate limits (429) or internal errors (500).

        Returns:
            A string with the model's response or "" on persistent error.
        """
        if not self.api_key:
            logging.error("No OpenRouter API key provided.")
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://replit.com/bart-experiment",
            "X-Title": "BART LLM Experiment"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        if self.debug:
            logging.info(f"[DEBUG {self.model}] Sending request payload: {payload}")

        max_retries = 3
        backoff_seconds = 5

        for attempt in range(max_retries):
            resp = None
            try:
                resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()  # Raises error for 4xx/5xx
                data = resp.json()

                if self.debug:
                    logging.info(f"[DEBUG {self.model}] Raw JSON response: {data}")

                choices = data.get("choices", [])
                if not choices:
                    logging.warning(f"No completion returned from API. data={data}")
                    return ""
                return choices[0]["message"]["content"] or ""

            except requests.exceptions.RequestException as e:
                msg = f"OpenRouter API call failed: {e}"
                if resp is not None:
                    status_code = resp.status_code
                    try:
                        data = resp.json()
                    except Exception:
                        data = {}
                    logging.warning(f"No completion returned from API. data={data}")

                    if status_code in (429, 500):
                        custom_delay = self._extract_retry_delay(data)
                        if custom_delay:
                            logging.warning(f"Rate-limit/500 error. Sleeping {custom_delay}s before retry...")
                            time.sleep(custom_delay)
                        else:
                            logging.warning(f"Rate-limit/500 error. Sleeping {backoff_seconds}s before retry...")
                            time.sleep(backoff_seconds)
                            backoff_seconds *= 3
                        continue
                else:
                    logging.warning(msg)
                break

        # If we exhaust retries, return ""
        return ""

    def _extract_retry_delay(self, data):
        """
        Attempt to parse 'retryDelay' from the API response if present.
        Returns integer seconds or None.
        """
        error = data.get("error", {})
        details = error.get("details", [])
        for d in details:
            if "@type" in d and "RetryInfo" in d["@type"]:
                if "retryDelay" in d:
                    delay_str = d["retryDelay"]
                    if isinstance(delay_str, str) and delay_str.endswith("s"):
                        try:
                            return int(delay_str[:-1])
                        except ValueError:
                            pass
        return None
