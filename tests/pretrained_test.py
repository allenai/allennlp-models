from typing import Dict, List
import pytest
import spacy

from allennlp.common.testing import AllenNlpTestCase
from allennlp_models.pretrained import get_pretrained_models, get_tasks, load_predictor


# But default we don't run these tests
@pytest.mark.pretrained_model_test
class TestAllenNlpPretrainedModels(AllenNlpTestCase):
    def test_machine_comprehension(self):
        predictor = load_predictor("rc-bidaf")

        passage = """The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called "the Matrix", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer Neo" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the "dream world". """
        question = "Who stars in The Matrix?"

        result = predictor.predict_json({"passage": passage, "question": question})

        correct = (
            "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano"
        )

        assert correct == result["best_span_str"]

    def test_semantic_role_labeling(self):
        predictor = load_predictor("structured-prediction-srl-bert")

        sentence = "If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"

        result = predictor.predict_json({"sentence": sentence})

        assert result["words"] == [
            "If",
            "you",
            "liked",
            "the",
            "music",
            "we",
            "were",
            "playing",
            "last",
            "night",
            ",",
            "you",
            "will",
            "absolutely",
            "love",
            "what",
            "we",
            "'re",
            "playing",
            "tomorrow",
            "!",
        ]

        assert result["verbs"] == [
            {
                "verb": "liked",
                "description": "If [ARG0: you] [V: liked] [ARG1: the music we were playing last night] , you will absolutely love what we 're playing tomorrow !",
                "tags": [
                    "O",
                    "B-ARG0",
                    "B-V",
                    "B-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "were",
                "description": "If you liked the music we [V: were] playing last night , you will absolutely love what we 're playing tomorrow !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-V",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "playing",
                "description": "If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last night] , you will absolutely love what we 're playing tomorrow !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "B-ARG1",
                    "I-ARG1",
                    "B-ARG0",
                    "O",
                    "B-V",
                    "B-ARGM-TMP",
                    "I-ARGM-TMP",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "will",
                "description": "If you liked the music we were playing last night , you [V: will] absolutely love what we 're playing tomorrow !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-V",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "love",
                "description": "[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [ARGM-MOD: will] [ARGM-ADV: absolutely] [V: love] [ARG1: what we 're playing tomorrow] !",
                "tags": [
                    "B-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "O",
                    "B-ARG0",
                    "B-ARGM-MOD",
                    "B-ARGM-ADV",
                    "B-V",
                    "B-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "O",
                ],
            },
            {
                "verb": "'re",
                "description": "If you liked the music we were playing last night , you will absolutely love what we [V: 're] playing tomorrow !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-V",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "playing",
                "description": "If you liked the music we were playing last night , you will absolutely love [ARG1: what] [ARG0: we] 're [V: playing] [ARGM-TMP: tomorrow] !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-ARG1",
                    "B-ARG0",
                    "O",
                    "B-V",
                    "B-ARGM-TMP",
                    "O",
                ],
            },
        ]

    def test_textual_entailment(self):
        predictor = load_predictor("pair-classification-decomposable-attention-elmo")

        result = predictor.predict_json(
            {
                "premise": "An interplanetary spacecraft is in orbit around a gas giant's icy moon.",
                "hypothesis": "The spacecraft has the ability to travel between planets.",
            }
        )

        assert result["label_probs"][0] > 0.7  # entailment

        result = predictor.predict_json(
            {
                "premise": "Two women are wandering along the shore drinking iced tea.",
                "hypothesis": "Two women are sitting on a blanket near some rocks talking about politics.",
            }
        )

        assert result["label_probs"][1] > 0.8  # contradiction

        result = predictor.predict_json(
            {
                "premise": "A large, gray elephant walked beside a herd of zebras.",
                "hypothesis": "The elephant was lost.",
            }
        )

        assert result["label_probs"][2] > 0.6  # neutral

    def test_coreference_resolution(self):
        predictor = load_predictor("coref-spanbert")

        document = "We 're not going to skimp on quality , but we are very focused to make next year . The only problem is that some of the fabrics are wearing out - since I was a newbie I skimped on some of the fabric and the poor quality ones are developing holes ."

        result = predictor.predict_json({"document": document})
        print(result)
        assert result["clusters"] == [
            [[0, 0], [10, 10]],
            [[33, 33], [37, 37]],
            # [[26, 27], [42, 43]],  # Unfortunately the model misses this one.
        ]
        assert result["document"] == [
            "We",
            "'re",
            "not",
            "going",
            "to",
            "skimp",
            "on",
            "quality",
            ",",
            "but",
            "we",  # 10
            "are",
            "very",
            "focused",
            "to",
            "make",
            "next",
            "year",
            ".",
            "The",
            "only",  # 20
            "problem",
            "is",
            "that",
            "some",
            "of",
            "the",
            "fabrics",
            "are",
            "wearing",
            "out",  # 30
            "-",
            "since",
            "I",
            "was",
            "a",
            "newbie",
            "I",
            "skimped",
            "on",
            "some",  # 40
            "of",
            "the",
            "fabric",
            "and",
            "the",
            "poor",
            "quality",
            "ones",
            "are",
            "developing",  # 50
            "holes",
            ".",
        ]

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_ner(self):
        predictor = load_predictor("tagging-elmo-crf-tagger")

        sentence = """Michael Jordan is a professor at Berkeley."""

        result = predictor.predict_json({"sentence": sentence})

        assert result["words"] == [
            "Michael",
            "Jordan",
            "is",
            "a",
            "professor",
            "at",
            "Berkeley",
            ".",
        ]
        assert result["tags"] == ["B-PER", "L-PER", "O", "O", "O", "O", "U-LOC", "O"]

    @pytest.mark.skipif(
        not ("2.1" <= spacy.__version__ < "2.3"),
        reason="this model changed before and after 2.1 and 2.2",
    )
    def test_constituency_parsing(self):
        predictor = load_predictor("structured-prediction-constituency-parser")

        sentence = """Pierre Vinken died aged 81; immortalised aged 61."""

        result = predictor.predict_json({"sentence": sentence})

        assert result["tokens"] == [
            "Pierre",
            "Vinken",
            "died",
            "aged",
            "81",
            ";",
            "immortalised",
            "aged",
            "61",
            ".",
        ]
        assert (
            result["trees"]
            == "(S (NP (NNP Pierre) (NNP Vinken)) (VP (VP (VBD died) (NP (JJ aged) (CD 81))) (, ;) (VP (VBN immortalised) (S (ADJP (JJ aged) (CD 61))))) (. .))"
        )

    def test_dependency_parsing(self):
        predictor = load_predictor("structured-prediction-biaffine-parser")
        sentence = """He ate spaghetti with chopsticks."""
        result = predictor.predict_json({"sentence": sentence})
        # Note that this tree is incorrect. We are checking here that the decoded
        # tree is _actually a tree_ - in greedy decoding versions of the dependency
        # parser, this sentence has multiple heads. This test shouldn't really live here,
        # but it's very difficult to re-create a concrete example of this behaviour without
        # a trained dependency parser.
        assert result["words"] == ["He", "ate", "spaghetti", "with", "chopsticks", "."]
        assert result["pos"] == ["PRON", "VERB", "NOUN", "ADP", "NOUN", "PUNCT"]
        assert result["predicted_dependencies"] == [
            "nsubj",
            "root",
            "dobj",
            "prep",
            "pobj",
            "punct",
        ]
        assert result["predicted_heads"] == [2, 0, 2, 2, 4, 2]

    def test_sentiment_analysis(self):
        predictor = load_predictor("roberta-sst")
        result = predictor.predict_json({"sentence": "This is a positive review."})
        assert result["label"] == "1"

    def test_openie(self):
        predictor = load_predictor("structured-prediction-srl")
        result = predictor.predict_json(
            {"sentence": "I'm against picketing, but I don't know how to show it."}
        )
        assert "verbs" in result
        assert "words" in result

    @pytest.mark.parametrize(
        "get_model_arg",
        ["tagging-fine-grained-crf-tagger", "tagging-fine-grained-transformer-crf-tagger"],
    )
    def test_fine_grained_ner(self, get_model_arg):
        predictor = load_predictor(get_model_arg)
        text = """Dwayne Haskins passed for 251 yards and three touchdowns, and Urban Meyer finished his coaching career at Ohio State with a 28-23 victory after the Buckeyes held off Washington’s thrilling fourth-quarter comeback in the 105th Rose Bowl on Tuesday. Parris Campbell, Johnnie Dixon and Rashod Berry caught TD passes in the first half for the fifth-ranked Buckeyes (13-1), who took a 25-point lead into the fourth. But Myles Gaskin threw a touchdown pass and rushed for two more scores for the No. 9 Huskies (10-4), scoring from 2 yards out with 42 seconds left. The Buckeyes intercepted Jake Browning’s pass on the 2-point conversion attempt and then recovered the Huskies’ onside kick to wrap up the final game of Meyer’s seven-year tenure. “I’m a very blessed man,” Meyer said. “I’m blessed because of my family, [but] this team, this year, I love this group as much as any I’ve ever had.”"""
        result = predictor.predict_json({"sentence": text})
        # Just assert that we predicted something better than all-O.
        assert len(frozenset(result["tags"])) > 1

    def test_transformer_qa(self):
        predictor = load_predictor("rc-transformer-qa")

        passage = (
            "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were "
            "the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. "
            'They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, '
            "Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. "
            "Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, "
            "their descendants would gradually merge with the Carolingian-based cultures of West Francia. "
            "The distinct cultural and ethnic identity of the Normans emerged initially in the first half "
            "of the 10th century, and it continued to evolve over the succeeding centuries."
        )

        question = "In what country is Normandy located?"
        result = predictor.predict(question, passage)
        assert result["best_span_str"] == "France"

        question = "Who gave their name to Normandy in the 1000's and 1100's"
        result = predictor.predict(question, passage)
        assert result["best_span_str"] == ""

    def test_vilbert_vqa(self):
        predictor = load_predictor("vqa-vilbert")

        result = predictor.predict(
            question="What game are they playing?",
            image="https://storage.googleapis.com/allennlp-public-data/vqav2/baseball.jpg",
        )
        max_answer = max((prob, answer) for answer, prob in result["tokens"].items())[1]
        assert max_answer == "baseball"

    @pytest.mark.parametrize("model_id, model_card", get_pretrained_models().items())
    def test_pretrained_models(self, model_id, model_card):
        assert model_card.model_usage.archive_file is not None
        models_without_training_configs = {"rc-nmn", "lm-next-token-lm-gpt2", "evaluate_rc-lerc"}
        if model_id not in models_without_training_configs:
            assert model_card.model_usage.training_config is not None
        assert model_card.display_name is not None
        assert model_card.model_details.description is not None
        assert model_card.model_details.short_description is not None

        if model_id not in ["rc-nmn"]:
            assert model_card.evaluation_data.dataset is not None
            assert model_card.training_data.dataset is not None
            assert model_card.metrics.model_performance_measures is not None

            if not model_id.startswith("lm") and not model_id.startswith("generation"):
                assert (
                    model_card.evaluation_data.dataset.processed_url is not None
                    or model_card.evaluation_data.dataset.notes is not None
                )

    @pytest.mark.parametrize("task_id, task_card", get_tasks().items())
    def test_tasks(self, task_id, task_card):
        if task_card.examples is not None:
            assert isinstance(task_card.examples, List) or isinstance(task_card.examples, Dict)
