# Dataset Analysis

## Dataset Structure

- 1,000 stories
- 99 unique genre labels
- 98 genres contain 10 stories each
- Historical Adventure contains 20 stories
- No missing values
- No duplicate IDs
- No duplicate titles
- No duplicate story texts

## Story Length

- Mean: 985.86 words
- Median: 978.5 words
- P90: 1,424.6 words
- P95: 1,669.3 words
- P99: 2,268.5 words
- Maximum: 3,093 words

81.3% of stories contain fewer than 1,500 words,
and 98.3% contain fewer than 2,000 words.

## Initial Implications

The dataset is clean and nearly balanced across genres.

Most stories are relatively compact, while only a small number
of stories are longer than 2,000 words.

This suggests that story-level retrieval may be a viable baseline
for the retrieval system. However, the final decision between
story-level and chunk-level retrieval will depend on the token
distribution under the selected embedding and LLM models.