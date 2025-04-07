import random
import logging
from openrouter_api import OpenRouterAPI

class BARTExperiment:
    """
    Runs the Balloon Analogue Risk Task with an LLM.
    Each balloon is a self-contained multi-turn conversation,
    so the model 'remembers' prior pumps within that balloon,
    but we start fresh for each new balloon.
    """

    def __init__(
        self,
        min_pumps,
        max_pumps,
        reward_per_pump,
        num_balloons,
        model,
        api_key=None,
        thresholds=None,
        debug=False
    ):
        """
        Initialize a BART experiment.

        Args:
            min_pumps (int): Minimum pumps before burst is possible.
            max_pumps (int): Maximum pumps before burst is possible.
            reward_per_pump (float): Monetary reward per pump.
            num_balloons (int): Number of balloons in the experiment.
            model (str): Model identifier for OpenRouter.
            api_key (str, optional): OpenRouter API key.
            thresholds (list of int]): A pre-generated list of thresholds
                (one per balloon). If None, random thresholds for each balloon.
            debug (bool): If True, log full request/response payloads for debugging.

        Note:
            Each balloon is run in a brand new conversation. Within that balloon,
            we append user & assistant turns for each pump. Then we discard it
            and move on to the next balloon.
        """
        self.min_pumps = min_pumps
        self.max_pumps = max_pumps
        self.reward_per_pump = reward_per_pump
        self.num_balloons = num_balloons
        self.model = model
        self.debug = debug

        self.api = OpenRouterAPI(api_key, model, debug=debug)

        # If user didn't provide thresholds, generate them
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = [
                random.randint(self.min_pumps, self.max_pumps) 
                for _ in range(num_balloons)
            ]

    def run_experiment(self):
        """
        Orchestrate the entire set of balloons. Returns a list of balloon results dicts.
        """
        results = []
        for balloon_id in range(1, self.num_balloons + 1):
            threshold = self.thresholds[balloon_id - 1]
            logging.info(f"[{self.model}] Starting Balloon {balloon_id} threshold={threshold}")
            balloon_result = self.run_single_balloon(balloon_id, threshold)
            results.append(balloon_result)
        return results

    def run_single_balloon(self, balloon_id, threshold):
        """
        For balloon_id, we create a fresh conversation and keep adding messages
        until the balloon either bursts or the model says 'Cash Out' (or fails).
        """
        # Start a fresh conversation for this balloon
        conversation = []

        # System message with BART rules
        system_message = (
            "You are playing a Balloon Analogue Risk Task (BART). "
            "Rules:\n"
            f" - The balloon can burst at any pump from {self.min_pumps}..{self.max_pumps}.\n"
            f" - Each pump earns you ${self.reward_per_pump:.2f}, but if it bursts, you lose all for this balloon.\n"
            " - You can choose 'Pump' to inflate or 'Cash Out' to stop.\n"
            "Respond ONLY with 'Pump' or 'Cash Out' at each step."
        )
        conversation.append({"role": "system", "content": system_message})

        # The user "introduces" the balloon
        conversation.append({
            "role": "user",
            "content": (
                f"Balloon #{balloon_id}. Current earnings: $0.00.\n"
                "Do you choose 'Pump' or 'Cash Out'?\n"
                "ONLY respond with 'Pump' or 'Cash Out'."
            )
        })

        pumps_attempted = 0
        choices = []
        burst = False
        full_responses = []
        error_flag = False
        error_message = ""

        while True:
            decision, response_text = self.get_decision_from_model(conversation)

            if not decision:
                # If we don't get a valid response, mark error and end
                error_flag = True
                error_message = "No valid LLM response after 3 attempts"
                logging.warning(f"[{self.model}] Balloon {balloon_id} ERROR: {error_message}")
                break

            choices.append(decision)
            full_responses.append(response_text)

            # Append the assistant message so the conversation has it
            conversation.append({"role": "assistant", "content": response_text})

            if decision == "PUMP":
                pumps_attempted += 1

                # Log a short line if debug is off, so we see what's happening
                if not self.debug:
                    logging.info(
                        f"[{self.model}] Balloon {balloon_id} => PUMP => "
                        f"total pumps={pumps_attempted}, threshold={threshold}, still safe"
                    )

                if pumps_attempted >= threshold:
                    # The balloon bursts
                    burst = True
                    conversation.append({
                        "role": "user",
                        "content": "The balloon has BURST! You lost all earnings for this balloon ($0)."
                    })
                    if not self.debug:
                        logging.info(
                            f"[{self.model}] Balloon {balloon_id} => BURST => "
                            f"total pumps={pumps_attempted} (final earn=$0.00)"
                        )
                    break
                else:
                    current_earnings = pumps_attempted * self.reward_per_pump
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"The balloon did NOT burst. Current balloon earnings: ${current_earnings:.2f}.\n"
                            "Do you choose to 'Pump' again or 'Cash Out'?\n"
                            "Respond ONLY with 'Pump' or 'Cash Out'."
                        )
                    })

            else:  # "CASH OUT"
                # The user chooses to stop
                if not self.debug:
                    final_earn = pumps_attempted * self.reward_per_pump
                    logging.info(
                        f"[{self.model}] Balloon {balloon_id} => CASH OUT => "
                        f"total pumps={pumps_attempted} (final earn=${final_earn:.2f})"
                    )
                break

        # Compute final data for this balloon
        if error_flag:
            return {
                "balloon_id": balloon_id,
                "threshold_pumps": threshold,
                "pumps_attempted": 0,
                "burst": False,
                "earnings": 0.0,
                "choices": [],
                "full_responses": [f"ERROR: {error_message}"]
            }
        else:
            if burst:
                earnings = 0.0
            else:
                earnings = pumps_attempted * self.reward_per_pump

            return {
                "balloon_id": balloon_id,
                "threshold_pumps": threshold,
                "pumps_attempted": pumps_attempted,
                "burst": burst,
                "earnings": round(earnings, 2),
                "choices": choices,
                "full_responses": full_responses
            }

    def get_decision_from_model(self, conversation):
        """
        Send the conversation to the model, parse the response, up to 3 tries
        to get a valid "PUMP" or "CASH OUT".

        Returns (decision, raw_response_text) or (None, "") on error.
        """
        for attempt in range(3):
            response_text = self.api.send_message(conversation)
            if not response_text:
                logging.warning(f"[{self.model}] Empty/invalid response. Attempt={attempt+1}")
                continue

            parsed = self.extract_decision(response_text)
            if parsed in ("PUMP", "CASH OUT"):
                return (parsed, response_text.strip())
            else:
                logging.warning(f"[{self.model}] Invalid choice '{parsed}'. Attempt={attempt+1}")

        # If we never got a valid decision
        return (None, "")

    def extract_decision(self, raw_response):
        text = raw_response.lower()
        if "pump" in text and "cash" not in text:
            return "PUMP"
        elif "cash out" in text or "cashout" in text:
            return "CASH OUT"
        return "INVALID"
