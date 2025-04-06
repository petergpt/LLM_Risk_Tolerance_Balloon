import random
import logging
from openrouter_api import OpenRouterAPI

class BARTExperiment:
    """
    Runs the Balloon Analogue Risk Task with an LLM.
    """

    def __init__(self, min_pumps, max_pumps, reward_per_pump, num_balloons,
                 model, api_key=None):
        """
        Initialize a BART experiment.
        
        Args:
            min_pumps (int): Minimum pumps before burst is possible
            max_pumps (int): Maximum possible pumps before burst
            reward_per_pump (float): Monetary reward per pump
            num_balloons (int): Number of balloons in the experiment
            model (str): Model identifier for OpenRouter
            api_key (str, optional): OpenRouter API key
        """
        self.min_pumps = min_pumps
        self.max_pumps = max_pumps
        self.reward_per_pump = reward_per_pump
        self.num_balloons = num_balloons

        self.api = OpenRouterAPI(api_key, model)

        self.intro_template = (
            "You have a virtual balloon.\n"
            "- Each pump earns you ${:.2f}.\n"
            "- Each pump increases the chance the balloon will burst.\n"
            "- If the balloon bursts, you lose everything for that balloon.\n"
            "You can 'Pump' again or 'Cash Out' anytime.\n"
            "Current earnings: ${:.2f}\n"
            "Respond ONLY with 'Pump' or 'Cash Out'."
        )

        self.intermediate_template = (
            "The balloon did NOT burst. Current earnings: ${:.2f}.\n"
            "Do you choose to 'Pump' again or 'Cash Out'?\n"
            "Respond ONLY with 'Pump' or 'Cash Out'."
        )

    def run_experiment(self):
        """
        Orchestrate the entire set of balloons. Returns a list of balloon results dicts.
        """
        results = []

        for balloon_id in range(1, self.num_balloons + 1):
            threshold = random.randint(self.min_pumps, self.max_pumps)
            logging.info(f"Starting balloon {balloon_id}/{self.num_balloons} with burst threshold {threshold}")
            balloon_result = self.run_single_balloon(balloon_id, threshold)
            results.append(balloon_result)
            
            # Log the result of this balloon
            log_status = "BURST" if balloon_result["burst"] else "CASH OUT"
            log_msg = (f"Balloon {balloon_id}: {log_status} after {balloon_result['pumps_attempted']} "
                       f"pumps, earned ${balloon_result['earnings']:.2f}")
            logging.info(log_msg)

        return results

    def run_single_balloon(self, balloon_id, threshold):
        """
        For a single balloon, keep track of pumps until model bursts or cashes out.
        Returns a dict with all balloon-level data.
        """

        pumps_attempted = 0
        choices = []
        burst = False
        full_responses = []  # Capture the full text responses too

        # Start with an introduction prompt
        conversation = [
            {"role": "system", "content": "You are a neutral game interface."},
            {
                "role": "user",
                "content": self.intro_template.format(
                    self.reward_per_pump,
                    0.0
                )
            }
        ]

        while True:
            response = self.api.send_message(conversation)
            full_responses.append(response)

            # Clean the response
            decision = self.extract_decision(response)
            choices.append(decision)

            if decision == "PUMP":
                pumps_attempted += 1
                # Check if it bursts
                if pumps_attempted >= threshold:
                    # The balloon bursts!
                    burst = True
                    # Notify the model
                    conversation.append({
                        "role": "assistant", 
                        "content": response
                    })
                    conversation.append({
                        "role": "user",
                        "content": "The balloon has burst! You lost your earnings for this balloon ($0.00)."
                    })
                    break
                else:
                    # Not burst yet
                    conversation.append({
                        "role": "assistant", 
                        "content": response
                    })
                    conversation.append({
                        "role": "user",
                        "content": self.intermediate_template.format(
                            pumps_attempted * self.reward_per_pump
                        )
                    })

            elif decision == "CASH OUT":
                # Model chooses to stop
                break
            else:
                # If we get weird output, let's just forcibly break or we could re-ask
                logging.warning(f"Unrecognized choice '{decision}'. Forcing 'Cash Out'.")
                decision = "CASH OUT"
                break

        # Compute final earnings
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

    def extract_decision(self, raw_response):
        """
        Normalize the model's response to "PUMP" or "CASH OUT" if possible.
        E.g. model says "Pump", "pump", "Pump." => we parse it as "PUMP".
        """
        # We'll do a minimal parse
        text = raw_response.strip().lower()
        if "pump" in text and "cash" not in text:
            return "PUMP"
        elif "cash out" in text or "cashout" in text:
            return "CASH OUT"
        # If the model included random filler, we do a best guess:
        if "pump" in text:
            return "PUMP"
        if "cash" in text:
            return "CASH OUT"
        return "CASH OUT"  # default to safe
