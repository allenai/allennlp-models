"""
Run this script to update the list of pre-trained models in the README based on the current model cards.
"""

from typing import List
import json
import glob


AUTO_GENERATED_SECTION_START = "<!-- This section is automatically generated"
AUTO_GENERATED_SECTION_END = "<!-- End automatically generated section -->"


def main():
    with open("README.md") as readme_file:
        readme_lines = readme_file.readlines()

    section_start_idx = next(
        (i for i, l in enumerate(readme_lines) if l.startswith(AUTO_GENERATED_SECTION_START))
    )
    section_end_idx = next(
        (i for i, l in enumerate(readme_lines) if l.startswith(AUTO_GENERATED_SECTION_END))
    )

    model_list: List[str] = ["\n"]
    for model_card_path in sorted(glob.glob("allennlp_models/modelcards/*.json")):
        if model_card_path.endswith("modelcard-template.json"):
            continue
        with open(model_card_path) as model_card_file:
            model_card = json.load(model_card_file)
            model_id = model_card["id"]
            description = model_card["model_details"]["short_description"]
        model_list.append(
            f"- [`{model_id}`](https://github.com/allenai/allennlp-models/tree/main/"
            f"{model_card_path}) - {description}\n"
        )
    model_list.append("\n")

    readme_lines = (
        readme_lines[: section_start_idx + 1] + model_list + readme_lines[section_end_idx:]
    )

    with open("README.md", "w") as readme_file:
        readme_file.writelines(readme_lines)


if __name__ == "__main__":
    main()
