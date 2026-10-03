from datasets import load_dataset
import pandas as pd


DATASET_NAME = "FareedKhan/1k_stories_100_genre"


def is_malformed_story(story: str) -> bool:
    """
    Detect samples that contain a generation/chat prompt instead of
    an actual generated story.

    We intentionally require multiple strong signals to avoid removing
    valid stories that happen to contain isolated chat-template tokens.
    """
    text = str(story).strip().lower()

    has_chat_template = "<|im_start|>" in text

    has_generation_instruction = (
        "write a very long story" in text
        or "write a story" in text
        or "generate a story" in text
    )

    has_assistant_marker = "<|im_start|> assistant" in text

    # A prompt-only sample is expected to be relatively short because
    # the requested story was never actually generated.
    is_short = len(text.split()) < 150

    return (
        has_chat_template
        and has_generation_instruction
        and has_assistant_marker
        and is_short
    )


def load_story_dataset(
    filter_malformed: bool = True,
) -> pd.DataFrame:
    """
    Load and validate the story dataset.

    Parameters
    ----------
    filter_malformed:
        Remove samples containing generation prompts instead of
        actual story content.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        id, title, genre, story
    """

    dataset = load_dataset(DATASET_NAME)

    # The dataset contains a single split.
    split_name = list(dataset.keys())[0]
    df = dataset[split_name].to_pandas()

    required_columns = {"id", "title", "genre", "story"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Keep only fields required by the application.
    df = df[["id", "title", "genre", "story"]].copy()

    # Standardize IDs for retrieval/filtering.
    df["id"] = df["id"].astype(str)

    if filter_malformed:
        malformed_mask = df["story"].apply(is_malformed_story)

        if malformed_mask.any():
            malformed_ids = df.loc[malformed_mask, "id"].tolist()

            print(
                f"Excluding {malformed_mask.sum()} malformed "
                f"story sample(s): {malformed_ids}"
            )

            df = df.loc[~malformed_mask].copy()

    return df.reset_index(drop=True)
