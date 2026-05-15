#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests
import yaml
from openai import OpenAI

INFO_ITEM_PATTERN = re.compile(r"<!--INFO_ITEM\s+(\{.*\})-->")


@dataclass
class ConceptPoint:
    title: str
    url: str
    source_type: str
    source_name: str
    published_at: str
    company_or_project: str
    concept: str
    summary: str
    why_relevant: str
    canonical_key: str
    embedding: list[float]


@dataclass
class DuplicateDecision:
    item: ConceptPoint
    is_duplicate: bool
    reason: str | None = None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def build_canonical_key(item: dict[str, Any]) -> str:
    payload = "|".join(
        [
            normalize_text(item.get("company_or_project", "")),
            normalize_text(item.get("concept", "")),
            normalize_text(item.get("title", "")),
            normalize_text(item.get("source_type", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def ensure_dirs(config: dict[str, Any]) -> None:
    Path(config["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["raw_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["history_index"]).parent.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)


def embedding_text(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            item.get("company_or_project", ""),
            item.get("concept", ""),
            item.get("title", ""),
            item.get("summary", ""),
            item.get("why_relevant", ""),
        ]
    )


def fetch_embeddings(texts: list[str]) -> list[list[float]]:
    model = os.getenv("EMBEDDING_MODEL_NAME")
    base_url = os.getenv("EMBEDDING_URL")
    if not model or not base_url:
        raise RuntimeError("Missing EMBEDDING_MODEL_NAME or EMBEDDING_URL environment variable.")

    response = requests.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        headers={"Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def web_search(config: dict[str, Any], today: str) -> list[dict[str, Any]]:
    key = os.getenv("Qwen_KEY")
    base_url = os.getenv("Qwen_local_url")
    if not key or not base_url:
        raise RuntimeError("Missing Qwen_KEY or Qwen_local_url environment variable.")

    client = OpenAI(api_key=key, base_url=base_url)
    prompt = {
        "question": config["question"],
        "today": today,
        "max_items": config.get("web_max_items", 12),
        "required_fields": [
            "title",
            "url",
            "source_name",
            "published_at",
            "company_or_project",
            "concept",
            "summary",
            "why_relevant",
        ],
    }
    resp = client.responses.create(
        model=config.get("openai_model", "Qwen3"),
        tools=[{"type": "web_search"}],
        input=[
            {
                "role": "system",
                "content": "Return JSON only. Do not include markdown.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt),
            },
        ],
    )
    raw = resp.output_text.strip()
    payload = json.loads(raw)
    return payload.get("items", [])


def github_search(config: dict[str, Any], since: str, today: str) -> list[dict[str, Any]]:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    out: list[dict[str, Any]] = []
    for query in config.get("github_queries", []):
        q = query["q"].replace("{since}", since).replace("{today}", today)
        q_type = query["type"]
        url = f"https://api.github.com/search/{q_type}"
        params = {
            "q": q,
            "sort": query.get("sort", "updated"),
            "order": query.get("order", "desc"),
            "per_page": config.get("github_max_items_per_query", 10),
        }
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        out.extend(normalize_github_items(q_type, items, config["question"]))
        time.sleep(float(config.get("github_min_delay_seconds", 2.5)))
    return out


def normalize_github_items(kind: str, items: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if kind == "repositories":
            title = item.get("full_name", "")
            published = item.get("updated_at", "")[:10]
            summary = item.get("description") or "Repository activity update."
            company = (item.get("owner") or {}).get("login", "")
            concept = f"Repository updated: {title}"
            url = item.get("html_url", "")
            source_name = "GitHub"
        else:
            title = item.get("title", "")
            published = (item.get("updated_at") or "")[:10]
            summary = item.get("body") or "Issue or pull request updated."
            summary = normalize_text(summary)[:500]
            repo_url = item.get("repository_url", "")
            company = repo_url.rsplit("/", 2)[-2] if repo_url else ""
            concept = f"GitHub discussion updated: {title}"
            url = item.get("html_url", "")
            source_name = "GitHub"

        normalized.append(
            {
                "title": title,
                "url": url,
                "source_type": "github",
                "source_name": source_name,
                "published_at": published,
                "company_or_project": company,
                "concept": concept,
                "summary": summary,
                "why_relevant": f"Related to fixed question: {question}",
            }
        )
    return normalized


def parse_report_history(reports_dir: Path) -> list[ConceptPoint]:
    history: list[ConceptPoint] = []
    for path in sorted(reports_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for match in INFO_ITEM_PATTERN.finditer(content):
            payload = json.loads(match.group(1))
            history.append(ConceptPoint(**payload))
    return history


def parse_jsonl_history(history_path: Path) -> list[ConceptPoint]:
    if not history_path.exists():
        return []
    out: list[ConceptPoint] = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ConceptPoint(**json.loads(line)))
    return out


def dedupe_candidates(candidates: list[ConceptPoint], history: list[ConceptPoint], config: dict[str, Any]) -> tuple[list[ConceptPoint], list[str]]:
    new_items: list[ConceptPoint] = []
    skips: list[str] = []
    threshold = float(config.get("embedding_similarity_threshold", 0.88))
    entity_threshold = float(config.get("same_entity_similarity_threshold", 0.82))

    all_seen = history.copy()
    for cand in candidates:
        reason = None
        for old in all_seen:
            if cand.url and cand.url == old.url:
                reason = f"same URL as {old.title}"
                break
            if cand.canonical_key == old.canonical_key:
                reason = f"same canonical concept as {old.title}"
                break
            sim = cosine_similarity(cand.embedding, old.embedding)
            if sim >= threshold:
                reason = f"embedding similarity {sim:.3f} with {old.title}"
                break
            if normalize_text(cand.company_or_project) == normalize_text(old.company_or_project) and sim >= entity_threshold:
                reason = f"same entity + high similarity {sim:.3f} with {old.title}"
                break

        if reason:
            skips.append(f"{cand.title}: {reason}")
        else:
            new_items.append(cand)
            all_seen.append(cand)
    return new_items, skips


def write_outputs(config: dict[str, Any], today: str, question: str, raw_payload: dict[str, Any], new_items: list[ConceptPoint], skips: list[str]) -> None:
    report_path = Path(config["reports_dir"]) / f"{today}.md"
    raw_path = Path(config["raw_dir"]) / f"{today}.json"
    history_path = Path(config["history_index"])

    raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Daily INFO Report - {today}", "", f"Fixed question: {question}", "", "## New concept points", ""]
    if not new_items:
        lines.extend(["No genuinely new concept points were found today.", ""])
    else:
        for idx, item in enumerate(new_items, start=1):
            lines.extend(
                [
                    f"### {idx}. {item.title}",
                    "",
                    f"- Source: {item.source_type}",
                    f"- Company/project: {item.company_or_project}",
                    f"- Published/updated: {item.published_at}",
                    f"- URL: {item.url}",
                    f"- Summary: {item.summary}",
                    f"- Why relevant: {item.why_relevant}",
                    "",
                    f"<!--INFO_ITEM {json.dumps(asdict(item), ensure_ascii=False)}-->",
                    "",
                ]
            )

    lines.extend(["## Duplicate / repeated items skipped", ""])
    if not skips:
        lines.append("- None")
    else:
        lines.extend([f"- Skipped title: {entry}" for entry in skips])
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    with history_path.open("a", encoding="utf-8") as f:
        for item in new_items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    ensure_dirs(config)

    today = dt.date.today().isoformat()
    since = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    web_items = web_search(config, today)
    gh_items = github_search(config, since, today)
    all_items = web_items + gh_items

    for item in all_items:
        item["source_type"] = item.get("source_type", "web")
        item["canonical_key"] = build_canonical_key(item)

    embeddings = fetch_embeddings([embedding_text(item) for item in all_items]) if all_items else []
    candidates = []
    for item, emb in zip(all_items, embeddings):
        item["embedding"] = emb
        candidates.append(ConceptPoint(**item))

    history = parse_jsonl_history(Path(config["history_index"]))
    history.extend(parse_report_history(Path(config["reports_dir"])))

    new_items, skips = dedupe_candidates(candidates, history, config)
    write_outputs(
        config=config,
        today=today,
        question=config["question"],
        raw_payload={"today": today, "web": web_items, "github": gh_items},
        new_items=new_items,
        skips=skips,
    )


if __name__ == "__main__":
    main()
