from pydantic import ValidationError

# pyrefly: ignore [missing-import]
from src.retrieval.router import QueryPlan


def test_valid_plan():
    plan = QueryPlan(
        intent="story_qa",
        retrieval_strategy="evidence",
        story_titles=["The Chronicles of the Celestial Spoon"],
        story_ids=[],
        genre=None,
        semantic_query=None,
    )

    print("✓ Valid plan accepted")
    print(plan)


def test_invalid_intent():
    try:
        QueryPlan(
            intent="banana",
            retrieval_strategy="evidence",
            story_titles=[],
            story_ids=[],
            genre=None,
            semantic_query=None,
        )

        print("✗ Invalid intent was accepted")

    except ValidationError as exc:
        print("✓ Invalid intent rejected")
        print(exc)


def test_invalid_strategy():
    try:
        QueryPlan(
            intent="story_qa",
            retrieval_strategy="vector_magic",
            story_titles=[],
            story_ids=[],
            genre=None,
            semantic_query=None,
        )

        print("✗ Invalid strategy was accepted")

    except ValidationError as exc:
        print("✓ Invalid strategy rejected")
        print(exc)


def test_defaults():
    plan = QueryPlan(
        intent="story_discovery",
        retrieval_strategy="semantic",
        semantic_query="friendship and adventure",
    )

    assert plan.story_titles == []
    assert plan.story_ids == []
    assert plan.genre is None

    print("✓ Default values work")
    print(plan)


def test_json_validation():
    raw_json = """
    {
        "intent": "summarization",
        "retrieval_strategy": "full_story",
        "story_titles": [
            "The Chronicles of the Celestial Spoon"
        ],
        "story_ids": [],
        "genre": null,
        "semantic_query": null
    }
    """

    plan = QueryPlan.model_validate_json(raw_json)

    print("✓ JSON validation works")
    print(plan)


def test_invalid_json_plan():
    raw_json = """
    {
        "intent": "something_invalid",
        "retrieval_strategy": "full_story",
        "story_titles": [],
        "story_ids": []
    }
    """

    try:
        QueryPlan.model_validate_json(raw_json)

        print("✗ Invalid JSON plan was accepted")

    except ValidationError as exc:
        print("✓ Invalid JSON plan rejected")
        print(exc)


def main():
    print("\n--- TEST 1: VALID PLAN ---")
    test_valid_plan()

    print("\n--- TEST 2: INVALID INTENT ---")
    test_invalid_intent()

    print("\n--- TEST 3: INVALID STRATEGY ---")
    test_invalid_strategy()

    print("\n--- TEST 4: DEFAULT VALUES ---")
    test_defaults()

    print("\n--- TEST 5: JSON VALIDATION ---")
    test_json_validation()

    print("\n--- TEST 6: INVALID JSON PLAN ---")
    test_invalid_json_plan()


if __name__ == "__main__":
    main()
