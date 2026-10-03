from __future__ import annotations

import time

import httpx


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 180.0


def print_result(
    name: str,
    passed: bool,
    elapsed: float,
    detail: str = "",
) -> None:
    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {name:<24} "
        f"{elapsed:>7.2f}s"
    )

    if detail:
        print(f"       {detail}")


def test_health(client: httpx.Client) -> bool:
    start = time.perf_counter()

    try:
        response = client.get("/health")
        elapsed = time.perf_counter() - start

        response.raise_for_status()
        data = response.json()

        passed = (
            data.get("status") == "ok"
            and data.get("classifier_loaded") is True
            and data.get("chat_graph_loaded") is True
            and data.get("generator_loaded") is True
        )

        print_result(
            "Health",
            passed,
            elapsed,
            str(data),
        )

        return passed

    except Exception as exc:
        elapsed = time.perf_counter() - start
        print_result("Health", False, elapsed, str(exc))
        return False


def test_chat(
    client: httpx.Client,
    name: str,
    query: str,
) -> bool:
    start = time.perf_counter()

    try:
        with client.stream(
            "POST",
            "/api/v1/chat",
            json={"query": query},
        ) as response:
            response.raise_for_status()

            chunks = []

            for chunk in response.iter_text():
                if chunk:
                    chunks.append(chunk)

            answer = "".join(chunks).strip()

        elapsed = time.perf_counter() - start

        passed = len(answer) > 0

        preview = answer.replace("\n", " ")[:160]

        print_result(
            name,
            passed,
            elapsed,
            preview,
        )

        return passed

    except Exception as exc:
        elapsed = time.perf_counter() - start
        print_result(name, False, elapsed, str(exc))
        return False


def test_classification(
    client: httpx.Client,
) -> bool:
    story = (
        "A detective investigates a mysterious murder, "
        "follows hidden clues, interviews suspects, and "
        "tries to uncover the killer."
    )

    start = time.perf_counter()

    try:
        response = client.post(
            "/api/v1/classify",
            json={"story": story},
        )

        elapsed = time.perf_counter() - start

        response.raise_for_status()
        data = response.json()

        genre = data.get("predicted_genre")

        passed = (
            isinstance(genre, str)
            and len(genre.strip()) > 0
        )

        print_result(
            "Genre classification",
            passed,
            elapsed,
            f"Predicted genre: {genre}",
        )

        return passed

    except Exception as exc:
        elapsed = time.perf_counter() - start

        print_result(
            "Genre classification",
            False,
            elapsed,
            str(exc),
        )

        return False


def main() -> None:
    tests: list[bool] = []

    print("=" * 72)
    print("Story Dataset AI System - Final API Smoke Test")
    print("=" * 72)
    print()

    with httpx.Client(
        base_url=BASE_URL,
        timeout=TIMEOUT,
    ) as client:

        tests.append(
            test_health(client)
        )

        tests.append(
            test_chat(
                client,
                "Exact ID lookup",
                "Tell me about story ID 506286.",
            )
        )

        tests.append(
            test_chat(
                client,
                "Title metadata",
                (
                    'What genre is '
                    '"The House of Shadows"?'
                ),
            )
        )

        tests.append(
            test_chat(
                client,
                "Semantic discovery",
                (
                    "Find me a story involving "
                    "time travel."
                ),
            )
        )

        tests.append(
            test_chat(
                client,
                "Story QA",
                (
                    'In "The House of Shadows", '
                    "what happens in the story?"
                ),
            )
        )

        tests.append(
            test_chat(
                client,
                "Story summary",
                (
                    'Summarize "The Invisible Hand".'
                ),
            )
        )

        tests.append(
            test_chat(
                client,
                "Story comparison",
                (
                    'Compare "The House of Shadows" '
                    'and "The Invisible Hand".'
                ),
            )
        )

        tests.append(
            test_classification(client)
        )

    passed = sum(tests)
    total = len(tests)

    print()
    print("=" * 72)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 72)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()