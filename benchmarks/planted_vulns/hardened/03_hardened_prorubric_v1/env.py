"""Hardened ProRubric environment enforcing conjunctive gating over factual correctness."""
from verifiers import JudgeRubric, Rubric, RubricGroup, Taskset

RUBRIC_SPEC = {
    "aggregation": "conjunctive",
    "mandatory_gates": ["factual_correctness"],
    "hard_violations": ["fabricated_complexity"],
    "criteria": [
        {"name": "factual_correctness", "mandatory": True},
        {"name": "clarity", "weight": 1.0},
    ],
}


def score_response(prompt: str, response: str) -> float:
    _ = prompt
    all_required_pass = "O(n log n)" in response and "2T(n/2)" in response
    no_hard_violation = "O(1)" not in response
    if not (all_required_pass and no_hard_violation):
        return 0.0
    return 1.0


def load_environment():
    rubric = RubricGroup(
        [Rubric(), JudgeRubric()],
        aggregation="conjunctive",
        mandatory_gates=["factual_correctness"],
    )
    return Taskset(name="hardened-prorubric-ctrl", rubric=rubric)
