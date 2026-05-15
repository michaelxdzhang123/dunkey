# Daily INFO Agent

Daily INFO Agent runs one fixed recurring question, collects web + GitHub sources, normalizes each result into a concept point, removes repeated concepts via URL/canonical/embedding checks, and writes only novel items into the daily report.

## Quick start

1. Create environment:
   - `uv venv --python 3.13`
   - `source .venv/bin/activate`
   - `uv sync` (or `uv pip install -r requirements.txt`)
2. Copy config:
   - `cp config.example.yaml config.yaml`
3. Set environment variables:
   - `Qwen_KEY`
   - `Qwen_local_url`
   - `EMBEDDING_MODEL_NAME`
   - `EMBEDDING_URL`
   - `GITHUB_TOKEN` (recommended)
4. Run:
   - `uv run info_agent.py --config config.yaml`

## Outputs

- `reports/YYYY-MM-DD.md`
- `raw/YYYY-MM-DD.json`
- `history/index.jsonl`
- `logs/cron.log` (when run by cron)
