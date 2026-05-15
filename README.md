# INFO Agent Starter

Daily INFO agent that:

1. Searches the web for one fixed question using OpenAI web search.
2. Searches GitHub repositories/issues/code/commits using GitHub REST Search API.
3. Normalizes every result into a "concept point".
4. Compares new concept points against all old Markdown reports and `history/index.jsonl`.
5. Writes a daily Markdown report with only novel points, plus a duplicate log.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp config.example.yaml config.yaml
export OPENAI_API_KEY="..."
export GITHUB_TOKEN="..."   # optional for public repo search, required/recommended for code search
uv run python info_agent.py --config config.yaml
```

## Daily automation with cron

Run every morning at 08:00:

```cron
0 8 * * * cd /path/to/info-agent-starter && /path/to/info-agent-starter/.venv/bin/python info_agent.py --config config.yaml >> logs/cron.log 2>&1
```

## Daily automation with GitHub Actions

Create `.github/workflows/info-agent.yml`:

```yaml
name: INFO Agent
on:
  schedule:
    - cron: "0 23 * * *" # 08:00 Japan time
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: uv pip install -r requirements.txt
      - run: uv run python info_agent.py --config config.yaml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Commit report
        run: |
          git config user.name "info-agent"
          git config user.email "info-agent@example.com"
          git add reports raw history
          git commit -m "Daily INFO report" || echo "No changes"
          git push
```

## How de-duplication works

The agent uses four layers:

1. Exact URL match.
2. Stable canonical key from entity + concept + normalized title.
3. Embedding similarity against historical concept points.
4. Lower embedding threshold when the same company/project/entity is detected.

Each report embeds hidden metadata comments like:

```md
<!--INFO_ITEM {"canonical_key":"...","title":"..."}-->
```

That lets the agent rebuild history from Markdown files even if `history/index.jsonl` is deleted.


## Local embedding model (optional)

If you run embeddings on a local OpenAI-compatible endpoint (for example Ollama), set the embedding config values in `config.yaml`:

```yaml
embedding_model: "Qwen3-Embedding-8B-Q4_K_M"
embedding_base_url: "http://172.28.21.22:11434/v1"
embedding_api_key: ""
```

Notes:
- `embedding_base_url` only applies to embedding generation.
- Web search still uses the main OpenAI client (`OPENAI_API_KEY`) because it depends on hosted web-search tools.


## API key and ChatGPT subscription note

A ChatGPT Plus/Pro/Team subscription and OpenAI API billing are separate products.

- If you only have a ChatGPT subscription, `OPENAI_API_KEY` may be unavailable for API calls.
- In that case, set `web_enabled: false` and use GitHub search + local embeddings.
- Web search via Responses API requires an API key with API billing enabled.

Example for no API key mode:

```yaml
web_enabled: false
embedding_model: "Qwen3-Embedding-8B-Q4_K_M"
embedding_base_url: "http://172.28.21.22:11434/v1"
embedding_api_key: ""
```
