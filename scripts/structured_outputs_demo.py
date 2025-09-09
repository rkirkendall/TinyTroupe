import os
import logging
from pathlib import Path

from tinytroupe.agent import TinyPerson


def _load_api_key_from_dotenv_if_missing():
    if os.getenv("OPENAI_API_KEY"):
        return

    # Try to read from .env in TinyTroupe/ and project root without overwriting env
    candidate_paths = [
        Path(__file__).resolve().parent.parent / ".env",        # project_root/TinyTroupe/../.env
        Path(__file__).resolve().parent / ".env",               # TinyTroupe/.env
        Path.cwd() / ".env",                                    # current working dir .env
    ]

    api_key = None
    for dotenv_path in candidate_paths:
        try:
            if dotenv_path.exists():
                with open(dotenv_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k == "OPENAI_API_KEY" and v:
                                api_key = v
                                break
            if api_key:
                break
        except Exception:
            continue

    if api_key and not os.getenv("OPENAI_API_KEY"):
        # Set only for this process; do not overwrite existing env
        os.environ["OPENAI_API_KEY"] = api_key


def main():
    # Ensure DEBUG logs
    logger = logging.getLogger("tinytroupe")
    logger.setLevel(logging.DEBUG)

    # Load from .env if needed (non-destructive)
    _load_api_key_from_dotenv_if_missing()

    # Require API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the environment or in a local .env file.")

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


