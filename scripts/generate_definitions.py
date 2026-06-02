import argparse
import json
from pathlib import Path

from def_gen.definition_generation import (
    GenerateDefinitionsConfig,
    run_generation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the generation workflow.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate ANW definition datasets from a JSON config file."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def load_json_config(config_path: Path) -> dict[str, object]:
    """Load configuration values from a JSON file.

    Args:
        config_path: Path to a JSON config file.

    Returns:
        Configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as fh:
        raw_config = json.load(fh)
    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a JSON object at the root.")
    return raw_config


def main() -> None:
    """Run the generation workflow from a required config file."""
    args = parse_args()
    config_data = load_json_config(args.config)
    config = GenerateDefinitionsConfig.model_validate(config_data)
    run_generation(config)


if __name__ == "__main__":
    main()
