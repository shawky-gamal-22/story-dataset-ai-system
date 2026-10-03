from pathlib import Path

import pandas as pd


INPUT_PATH = Path("experiments/task3/evidence_reranker_top10_results.csv")

QUERY_IDS = ["Q043", "Q044", "Q046"]


def main():
    df = pd.read_csv(INPUT_PATH)

    if df["supported"].dtype != bool:
        df["supported"] = (
            df["supported"].astype(str).str.lower().map({"true": True, "false": False})
        )

    for query_id in QUERY_IDS:
        group = df[df["query_id"] == query_id].copy()

        if group.empty:
            print(f"\n{query_id}: NOT FOUND")
            continue

        group = group.sort_values("reranker_rank")

        first = group.iloc[0]

        print("\n" + "=" * 100)
        print(f"{query_id}")
        print("=" * 100)

        print(f"\nQUESTION:\n{first['query']}")
        print(f"\nEXPECTED ANSWER:\n{first['expected_answer']}")

        print("\nRANKING:")
        print("-" * 100)

        for _, row in group.iterrows():
            marker = "✅ SUPPORT" if row["supported"] else "❌"

            text = str(row["evidence_text"]).replace("\n", " ")

            print(
                f"\nDense #{int(row['chunk_rank'])} | "
                f"Rerank #{int(row['reranker_rank'])} | "
                f"score={row['reranker_score']:.4f} | "
                f"{marker}"
            )

            print(
                f"tokens="
                f"{int(row['evidence_start_token'])}-"
                f"{int(row['evidence_end_token'])}"
            )

            print(text)


if __name__ == "__main__":
    main()
