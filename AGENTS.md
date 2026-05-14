# AGENTS.md

## Project: Daily INFO Agent

This project builds a daily INFO agent that answers one fixed research question every day, searches both the public web and GitHub, saves the results to Markdown, and avoids repeating concept points that have already appeared in earlier reports.

The agent's job is not only to collect links. Its main job is to detect whether a piece of information is genuinely new compared with historical `.md` reports.

## Target behavior

Every daily run should follow this pipeline:

```text
Fixed question
  -> web search
  -> GitHub search
  -> normalize results into concept points
  -> compare against historical Markdown and JSONL history
  -> remove repeated concepts
  -> write today's new-only Markdown report
  -> save raw data and update the history index
```

The final report for each day should contain only new concepts, not repeated versions of the same news, company update, repository update, or product announcement.

## Fixed question

The agent is designed around one fixed question configured in `config.yaml`:

```yaml
question: "What new information appeared today about AI agents, web search, and GitHub automation?"
```

This can be replaced with any stable recurring question, for example:

```yaml
question: "What new information appeared today about robotics foundation models and open-source robot agents?"
```

Keep the question stable over time. If the question changes often, the history comparison becomes less reliable.

## Core concept

The most important object in the system is a **concept point**.

A concept point is a normalized representation of one piece of information:

```yaml
title: "OpenAI adds X to web search"
url: "https://example.com/article"
source_type: "web"
source_name: "Example News"
published_at: "2026-05-14"
company_or_project: "OpenAI"
concept: "OpenAI adds X"
summary: "OpenAI announced or released X for web search."
why_relevant: "This is relevant to the fixed question because it affects AI agents using web search."
canonical_key: "hash(company + concept + normalized title)"
embedding: [0.0123, -0.0456, ...]
```

The agent should compare concept points, not just links.

Two different URLs can describe the same concept. One URL can also be updated or syndicated across multiple publishers. Therefore, URL matching alone is not enough.

## De-duplication rules

The agent should mark a new item as a duplicate when any of these are true:

```text
same URL
OR same canonical concept key
OR embedding_similarity >= 0.88
OR same company/project AND embedding_similarity >= 0.82
```

Recommended default thresholds:

```yaml
embedding_similarity_threshold: 0.88
same_entity_similarity_threshold: 0.82
```

### Same company is not enough

Do not remove an item only because the company or project is the same.

These should be treated as duplicates:

```text
OpenAI launches new web search feature
OpenAI releases web_search tool update
```

These should not automatically be treated as duplicates:

```text
OpenAI launches new web search feature
OpenAI acquires a startup
```

Same company, different concept, so keep both unless the summaries and embeddings show they are actually the same story.

## Historical memory

The agent should maintain two forms of history:

```text
history/index.jsonl
reports/YYYY-MM-DD.md
```

`history/index.jsonl` is the fast machine-readable index.

The Markdown files are the durable human-readable history. Each Markdown report should include hidden metadata comments so the history can be rebuilt even if the JSONL index is missing.

Example hidden comment:

```md
<!--INFO_ITEM {"canonical_key":"...","title":"...","url":"...","company_or_project":"...","concept":"...","embedding":[...]}-->
```

The agent should parse these comments from all previous `.md` reports before deciding whether today's item is new.

## Output files

Each daily run should create or update:

```text
reports/YYYY-MM-DD.md
raw/YYYY-MM-DD.json
history/index.jsonl
logs/cron.log              # when run by cron
```

The daily Markdown report should include:

```md
# Daily INFO Report - YYYY-MM-DD

Fixed question: ...

## New concept points

### 1. Concept title

- Source: web or GitHub
- Company/project: ...
- Published/updated: ...
- URL: ...
- Summary: ...
- Why relevant: ...

<!--INFO_ITEM {...}-->

## Duplicate / repeated items skipped

- Skipped title: reason it matched old history
```

If there are no new concept points, still write a report explaining that nothing novel was found.

## Source collection

### Web search

Use OpenAI's hosted web-search tool through the Responses API when possible.

The web-search step should ask for structured JSON, not prose. Each web result should be normalized into a concept point with:

```text
title
url
source_name
published_at
company_or_project
concept
summary
why_relevant
```

Rules for web collection:

- Prefer primary sources and official announcements when available.
- Avoid several articles that report the same story.
- Do not invent missing dates.
- Keep only items that answer the fixed question.
- Prefer fresh information for the current daily run.

### GitHub search

Use GitHub REST Search API for repositories, issues, pull requests, code, and commits where applicable.

Suggested query pattern:

```yaml
github_queries:
  - type: repositories
    q: "AI agent web search pushed:>{since}"
    sort: "updated"
    order: "desc"

  - type: issues
    q: "\"web_search\" \"agent\" updated:>{since} is:issue"
    sort: "updated"
    order: "desc"

  - type: issues
    q: "\"Responses API\" \"web_search\" updated:>{since} is:issue"
    sort: "updated"
    order: "desc"
```

Use `{since}` and `{today}` placeholders in config. Fill them as ISO dates such as `2026-05-14`.

Rules for GitHub collection:

- Keep the number of daily queries small.
- Respect GitHub search API rate limits.
- Add a delay between requests.
- Prefer repositories, issues, or releases that are directly relevant to the fixed question.
- Normalize every GitHub result into the same concept-point format as web results.

## Project structure

Expected starter structure:

```text
info-agent-starter/
  AGENTS.md
  README.md
  info_agent.py
  config.example.yaml
  requirements.txt
  reports/
  raw/
  history/
  logs/
```

The starter code should create missing output folders automatically.

## Configuration

Example `config.yaml`:

```yaml
question: "What new information appeared today about AI agents, web search, and GitHub automation?"

reports_dir: "reports"
raw_dir: "raw"
history_index: "history/index.jsonl"

openai_model: "gpt-5.5"
embedding_model: "text-embedding-3-small"
web_max_items: 12

embedding_similarity_threshold: 0.88
same_entity_similarity_threshold: 0.82

github_max_items_per_query: 10
github_min_delay_seconds: 2.5

github_queries:
  - type: repositories
    q: "AI agent web search pushed:>{since}"
    sort: "updated"
    order: "desc"

  - type: issues
    q: "\"web_search\" \"agent\" updated:>{since} is:issue"
    sort: "updated"
    order: "desc"

  - type: issues
    q: "\"Responses API\" \"web_search\" updated:>{since} is:issue"
    sort: "updated"
    order: "desc"
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml

export OPENAI_API_KEY="your_openai_key"
export GITHUB_TOKEN="your_github_token"

python info_agent.py --config config.yaml
```

`GITHUB_TOKEN` is optional for some public searches, but it is strongly recommended and may be required for code search or higher reliability.

## Daily automation with cron

Run every morning at 08:00:

```cron
0 8 * * * cd /path/to/info-agent-starter && /path/to/info-agent-starter/.venv/bin/python info_agent.py --config config.yaml >> logs/cron.log 2>&1
```

Make sure the environment variables are available to cron. If needed, place them in a shell wrapper script instead of directly in the crontab.

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

      - run: pip install -r requirements.txt

      - run: python info_agent.py --config config.yaml
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

## Implementation requirements for coding agents

When modifying this project, follow these rules:

1. Preserve the concept-point format.
2. Preserve hidden `<!--INFO_ITEM ... -->` metadata comments in reports.
3. Never rely only on URL matching for de-duplication.
4. Never discard an item only because it has the same company or project as a previous item.
5. Keep raw data in `raw/YYYY-MM-DD.json` for auditability.
6. Keep `history/index.jsonl` append-only where practical.
7. Handle missing API keys with clear errors.
8. Handle empty search results gracefully.
9. Keep Markdown reports human-readable.
10. Do not commit `.env`, API keys, tokens, or other secrets.

## Quality checks

Before considering a run successful, verify:

```text
reports/YYYY-MM-DD.md exists
raw/YYYY-MM-DD.json exists
history/index.jsonl exists or was updated
no duplicate concept appears in today's report
hidden INFO_ITEM comments are valid JSON
all new report items have title, URL, source type, summary, and concept
```

## Suggested de-duplication algorithm

For each candidate concept point:

```text
1. Normalize title, company/project, concept, and summary.
2. Build canonical_key from normalized company/project + concept + title + source type.
3. Generate embedding from company/project + concept + title + summary + why_relevant.
4. Compare with historical items.
5. Mark duplicate if:
   - URL already exists, or
   - canonical_key already exists, or
   - cosine similarity is above embedding_similarity_threshold, or
   - same normalized entity and cosine similarity is above same_entity_similarity_threshold.
6. Otherwise keep as a new concept point.
```

## Example duplicate decisions

Duplicate:

```text
Old: OpenAI adds web_search to agent tools
New: OpenAI releases web search tool for agents
Reason: same entity and highly similar concept
```

Duplicate:

```text
Old: GitHub repo example/agent-search releases v1.2
New: example/agent-search v1.2 release notes published
Reason: same project and same release concept
```

Not automatically duplicate:

```text
Old: Anthropic launches a new web browsing feature
New: Anthropic publishes a safety policy update
Reason: same company but different concept
```

Not automatically duplicate:

```text
Old: LangChain adds integration for search provider A
New: LangChain adds integration for search provider B
Reason: same project but different integration concept
```

## Recommended references

Use current official documentation when updating the implementation:

- OpenAI web-search tool documentation
- OpenAI embeddings documentation
- GitHub REST Search API documentation
- GitHub Actions workflow syntax documentation

Do not hard-code assumptions about API behavior that may change. Prefer small, readable functions and config-driven settings.

## Current goal

The current goal is to have a reliable daily report that answers:

```text
What is new today that was not already covered before?
```

The output should help a human quickly review genuinely new developments without rereading the same news, the same company announcement, or the same GitHub project update every day.
