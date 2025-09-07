import os
import logging

from tinytroupe.agent import TinyPerson


def main():
    # Ensure DEBUG logs
    logger = logging.getLogger("tinytroupe")
    logger.setLevel(logging.DEBUG)

    # Require API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")

    # Create a simple agent and act once to force an action generation
    agent = TinyPerson(name="DemoAgent")
    agent.listen("You're in a coffee shop. Order a cappuccino politely.")

    # Act once; structured output is enforced by ActionGenerator with Pydantic models
    outputs = agent.act(until_done=False, n=1, return_actions=True, communication_display=False)

    # Print the raw structured response for inspection
    print("\n=== Structured Output ===")
    print(outputs[-1])


if __name__ == "__main__":
    main()


