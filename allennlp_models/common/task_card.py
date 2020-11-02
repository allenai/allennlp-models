"""
A specification for defining task cards (derived from model cards).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from allennlp.common.from_params import FromParams


@dataclass(frozen=True)
class TaskCard(FromParams):
    """
    The `TaskCard` stores information about the task. It is modeled after the
    `ModelCard`.

    # Parameters

    id : `str`
        The task id.
        Example: `"rc"` for reading comprehension.
    name : `str`, optional
        The (display) name of the task.
    description : `str`, optional
        Description of the task.
        Example: "Textual Entailment (TE) is the task of predicting whether,
                 for a pair of sentences, the facts in the first sentence necessarily
                 imply the facts in the second."
    expected_inputs : `str`, optional
        All expected inputs and their format.
        Example: (For a reading comprehension task)
                 Passage (text string), Question (text string)
    expected_outputs : `str`, optional
        All expected outputs and their format.
        Example: (For a reading comprehension task)
                 Answer span (start token position and end token position).
    examples : `List[Dict[str, str]]`, optional
        List of examples for the task. Each dict should contain as keys the `expected_inputs`
        and `expected_outputs`.
        Example: (For textual entailment)
                 [{"premise": "A handmade djembe was on display at the Smithsonian.",
                   "hypothesis": "Visitors could see the djembe.",
                   "entails": "Yes"}]
                !!! Note
                    A model that attempts to solve the task may not produce the output
                    in exactly the same form as in the example. For instance, a model
                    may return probabilities instead of absolute booleans.
    """

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    expected_inputs: Optional[str] = None
    expected_outputs: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None
    # TODO: Add this to model cards.
