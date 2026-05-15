#!/usr/bin/env python3
"""
INFO Agent Starter

Daily collector that searches web + GitHub for one fixed question, writes Markdown,
and avoids repeating the same concept points across historical Markdown files.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
import yaml
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


INFO_COMMENT_RE = re.compile(r"<!--INFO_ITEM\s+(\{.*?\})\s*-->", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9一-龥ぁ-んァ-ン가-힣]+", re.IGNORECASE)


@dataclasses.dataclass
class ConceptPoint:
    title: str
    url: str
    source_type: str
    source_name: str = ""
    published_at: str = ""
    company_or_project: str = ""
    concept: str = ""
    summary: str = ""
    why_relevant: str = ""
    raw: Dict[str, Any] = dataclasses.field(default_factory=dict)
    canonical_key: str = ""
    embedding: Optional[List[float]] = None

    def text_for_embedding(self) -> str:
        parts = [
            self.company_or_project,
            self.concept,
            self.title,
            self.summary,
            self.why_relevant,
            self.source_type,
        ]
        return "\n".join([p for p in parts if p])[:6000]

    def to_history_record(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "published_at": self.published_at,
            "company_or_project": self.company_or_project,
            "concept": self.concept,
            "summary": self.summary,
            "why_relevant": self.why_relevant,
            "canonical_key": self.canonical_key,
            "embedding": self.embedding,
        }


@dataclasses.dataclass
class DuplicateDecision:
    is_duplicate: bool
    reason: str = ""
    matched_title: str = ""
    matched_url: str = ""
    similarity: float = 0.0


def today_utc() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def iso_date_days_ago(days: int) -> str:
    return (today_utc() - dt.timedelta(days=days)).isoformat()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    required = ["question"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    cfg.setdefault("reports_dir", "reports")
    cfg.setdefault("raw_dir", "raw")
    cfg.setdefault("history_index", "history/index.jsonl")
    cfg.setdefault("openai_model", "gpt-5.5")
    cfg.setdefault("embedding_model", "text-embedding-3-small")
    cfg.setdefault("embedding_base_url", "")
    cfg.setdefault("embedding_api_key", "")
    cfg.setdefault("web_max_items", 12)
    cfg.setdefault("github_max_items_per_query", 10)
    cfg.setdefault("github_min_delay_seconds", 2.5)
    cfg.setdefault("embedding_similarity_threshold", 0.88)
    cfg.setdefault("same_entity_similarity_threshold", 0.82)
    cfg.setdefault("github_queries", [])
    cfg.setdefault("web_enabled", True)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["raw_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["history_index"]).parent.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = NON_WORD_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_entity(text: str) -> str:
    text = normalize_text(text)
    for suffix in [" inc", " incorporated", " corp", " corporation", " ltd", " llc", " gmbh", " co"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip()


def stable_hash(text: str, n: int = 20) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def make_canonical_key(item: ConceptPoint) -> str:
    entity = normalize_entity(item.company_or_project)
    concept = normalize_text(item.concept or item.summary or item.title)
    title = normalize_text(item.title)
    # Do not rely only on URL because the same news may appear on different sites.
    basis = "|".join([entity, concept[:180], title[:180], item.source_type])
    return stable_hash(basis)


def cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def safe_json_loads(text: str) -> Any:
    text = text.strip()
    # Remove ```json fences if a model returns them despite instructions.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def require_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("OpenAI package is not installed. Install dependencies with `uv pip install -r requirements.txt`.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing.")
    return OpenAI()


def embed_items(items: List[ConceptPoint], cfg: Dict[str, Any]) -> None:
    missing = [item for item in items if item.embedding is None]
    if not missing:
        return

    if OpenAI is None:
        raise RuntimeError("OpenAI package is not installed. Install dependencies with `uv pip install -r requirements.txt`.")

    base_url = str(cfg.get("embedding_base_url", "")).strip()
    api_key = str(cfg.get("embedding_api_key", "")).strip()
    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed")
    else:
        client = require_openai_client()

    model = cfg["embedding_model"]
    batch_size = 64
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        inputs = [item.text_for_embedding() or item.title or item.url for item in batch]
        resp = client.embeddings.create(model=model, input=inputs)
        for item, emb_obj in zip(batch, resp.data):
            item.embedding = list(emb_obj.embedding)


def web_search_openai(question: str, cfg: Dict[str, Any]) -> List[ConceptPoint]:
    """Use OpenAI Responses API with hosted web search and ask for JSON concept points."""
    client = require_openai_client()
    max_items = int(cfg.get("web_max_items", 12))
    today = today_utc().isoformat()
    prompt = f"""
You are the collection step of a daily INFO agent.

Fixed question:
{question}

Today is {today} UTC. Search the public web for fresh and important information related to the fixed question.
Return ONLY a JSON array, no markdown, no prose. Max {max_items} items.

Each item must have these keys:
- title: concise title
- url: canonical source URL
- source_name: publisher/site name
- published_at: ISO date if available, otherwise empty string
- company_or_project: main company, project, repo, product, or organization; empty if not clear
- concept: one short reusable concept label, e.g. "OpenAI adds X", "Company Y raises funding", "Repo Z releases v1.2"
- summary: 1-2 sentence factual summary
- why_relevant: why it answers the fixed question

Rules:
- Prefer primary sources and official announcements when available.
- Avoid multiple items that are the same story from different publishers.
- Do not invent missing dates.
""".strip()

    resp = client.responses.create(
        model=cfg["openai_model"],
        tools=[{"type": "web_search"}],
        tool_choice="required",
        input=prompt,
    )
    text = getattr(resp, "output_text", "") or ""
    data = safe_json_loads(text)
    if not isinstance(data, list):
        raise ValueError("Web search response was not a JSON array.")

    items: List[ConceptPoint] = []
    for row in data[:max_items]:
        if not isinstance(row, dict):
            continue
        item = ConceptPoint(
            title=str(row.get("title", "")).strip(),
            url=str(row.get("url", "")).strip(),
            source_type="web",
            source_name=str(row.get("source_name", "")).strip(),
            published_at=str(row.get("published_at", "")).strip(),
            company_or_project=str(row.get("company_or_project", "")).strip(),
            concept=str(row.get("concept", "")).strip(),
            summary=str(row.get("summary", "")).strip(),
            why_relevant=str(row.get("why_relevant", "")).strip(),
            raw=row,
        )
        if item.title and item.url:
            item.canonical_key = make_canonical_key(item)
            items.append(item)
    return items


def github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_endpoint(search_type: str) -> str:
    mapping = {
        "repositories": "https://api.github.com/search/repositories",
        "repos": "https://api.github.com/search/repositories",
        "issues": "https://api.github.com/search/issues",
        "pull_requests": "https://api.github.com/search/issues",
        "prs": "https://api.github.com/search/issues",
        "code": "https://api.github.com/search/code",
        "commits": "https://api.github.com/search/commits",
    }
    if search_type not in mapping:
        raise ValueError(f"Unsupported GitHub search type: {search_type}")
    return mapping[search_type]


def extract_repo_from_github_url(url: str) -> str:
    m = re.search(r"github\.com/([^/]+/[^/#?]+)", url or "")
    return m.group(1) if m else ""


def github_item_to_concept(search_type: str, row: Dict[str, Any]) -> Optional[ConceptPoint]:
    if search_type in {"repositories", "repos"}:
        full_name = row.get("full_name", "")
        title = full_name or row.get("name", "")
        url = row.get("html_url", "")
        desc = row.get("description") or ""
        lang = row.get("language") or ""
        stars = row.get("stargazers_count")
        updated = row.get("updated_at", "")
        pushed = row.get("pushed_at", "")
        summary = desc
        if stars is not None:
            summary = f"{desc} Stars: {stars}. Language: {lang}.".strip()
        return ConceptPoint(
            title=title,
            url=url,
            source_type="github_repository",
            source_name="GitHub",
            published_at=pushed or updated,
            company_or_project=full_name,
            concept=f"Repository update: {full_name}",
            summary=summary,
            why_relevant="Repository matched the configured daily GitHub query.",
            raw=row,
        )

    if search_type in {"issues", "pull_requests", "prs"}:
        title = row.get("title", "")
        url = row.get("html_url", "")
        repo = extract_repo_from_github_url(url)
        is_pr = "pull_request" in row
        kind = "Pull request" if is_pr else "Issue"
        body = (row.get("body") or "").strip().replace("\n", " ")[:500]
        updated = row.get("updated_at", "")
        state = row.get("state", "")
        return ConceptPoint(
            title=f"{repo} #{row.get('number')}: {title}" if repo else title,
            url=url,
            source_type="github_pr" if is_pr else "github_issue",
            source_name="GitHub",
            published_at=updated,
            company_or_project=repo,
            concept=f"{kind}: {title}",
            summary=f"State: {state}. {body}".strip(),
            why_relevant="GitHub issue/PR matched the configured daily query.",
            raw=row,
        )

    if search_type == "code":
        repo = (row.get("repository") or {}).get("full_name", "")
        name = row.get("name", "")
        path = row.get("path", "")
        url = row.get("html_url", "")
        return ConceptPoint(
            title=f"{repo}: {path or name}",
            url=url,
            source_type="github_code",
            source_name="GitHub",
            published_at="",
            company_or_project=repo,
            concept=f"Code match in {repo}: {path or name}",
            summary="Code search match from GitHub Search API.",
            why_relevant="Code result matched the configured daily GitHub query.",
            raw=row,
        )

    if search_type == "commits":
        url = row.get("html_url", "")
        repo = extract_repo_from_github_url(url)
        commit = row.get("commit") or {}
        msg = (commit.get("message") or "").split("\n")[0]
        author_date = ((commit.get("author") or {}).get("date")) or ""
        return ConceptPoint(
            title=f"{repo}: {msg}" if repo else msg,
            url=url,
            source_type="github_commit",
            source_name="GitHub",
            published_at=author_date,
            company_or_project=repo,
            concept=f"Commit: {msg}",
            summary=msg,
            why_relevant="Commit matched the configured daily GitHub query.",
            raw=row,
        )

    return None


def github_search(cfg: Dict[str, Any]) -> List[ConceptPoint]:
    queries = cfg.get("github_queries", []) or []
    if not queries:
        return []

    today = today_utc().isoformat()
    since = iso_date_days_ago(int(cfg.get("since_days", 7)))
    max_items = int(cfg.get("github_max_items_per_query", 10))
    delay = float(cfg.get("github_min_delay_seconds", 2.5))

    results: List[ConceptPoint] = []
    headers = github_headers()

    for qcfg in queries:
        search_type = str(qcfg.get("type", "repositories"))
        endpoint = github_endpoint(search_type)
        q = str(qcfg.get("q", "")).format(today=today, since=since)
        params = {
            "q": q,
            "per_page": min(100, max_items),
        }
        if qcfg.get("sort"):
            params["sort"] = qcfg["sort"]
        if qcfg.get("order"):
            params["order"] = qcfg["order"]
        if qcfg.get("advanced_search"):
            params["advanced_search"] = qcfg["advanced_search"]
        if qcfg.get("search_type"):
            params["search_type"] = qcfg["search_type"]

        resp = requests.get(endpoint, headers=headers, params=params, timeout=30)
        if resp.status_code in {403, 429}:
            # Respect GitHub's reset/retry headers when present.
            retry_after = resp.headers.get("retry-after")
            reset = resp.headers.get("x-ratelimit-reset")
            if retry_after:
                wait = int(retry_after)
            elif reset:
                wait = max(0, int(reset) - int(time.time()))
            else:
                wait = 60
            print(f"GitHub rate limited; waiting {wait}s", file=sys.stderr)
            time.sleep(min(wait, 300))
            resp = requests.get(endpoint, headers=headers, params=params, timeout=30)

        resp.raise_for_status()
        payload = resp.json()
        for row in payload.get("items", [])[:max_items]:
            item = github_item_to_concept(search_type, row)
            if item and item.title and item.url:
                item.canonical_key = make_canonical_key(item)
                results.append(item)
        time.sleep(delay)

    return results


def load_history_from_index(index_path: Path) -> List[ConceptPoint]:
    if not index_path.exists():
        return []
    items: List[ConceptPoint] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                item = ConceptPoint(
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    source_type=row.get("source_type", ""),
                    source_name=row.get("source_name", ""),
                    published_at=row.get("published_at", ""),
                    company_or_project=row.get("company_or_project", ""),
                    concept=row.get("concept", ""),
                    summary=row.get("summary", ""),
                    why_relevant=row.get("why_relevant", ""),
                    canonical_key=row.get("canonical_key", ""),
                    embedding=row.get("embedding"),
                )
                if item.title or item.url:
                    items.append(item)
            except Exception:
                continue
    return items


def load_history_from_markdown(reports_dir: Path) -> List[ConceptPoint]:
    items: List[ConceptPoint] = []
    if not reports_dir.exists():
        return items
    for md_path in sorted(reports_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for match in INFO_COMMENT_RE.finditer(text):
            try:
                row = json.loads(match.group(1))
                item = ConceptPoint(
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    source_type=row.get("source_type", ""),
                    source_name=row.get("source_name", ""),
                    published_at=row.get("published_at", ""),
                    company_or_project=row.get("company_or_project", ""),
                    concept=row.get("concept", ""),
                    summary=row.get("summary", ""),
                    why_relevant=row.get("why_relevant", ""),
                    canonical_key=row.get("canonical_key", ""),
                    embedding=row.get("embedding"),
                )
                if item.title or item.url:
                    items.append(item)
            except Exception:
                continue
    return items


def merge_history(*groups: Iterable[ConceptPoint]) -> List[ConceptPoint]:
    seen: set[str] = set()
    merged: List[ConceptPoint] = []
    for group in groups:
        for item in group:
            key = item.canonical_key or item.url or normalize_text(item.title)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def duplicate_decision(item: ConceptPoint, history: List[ConceptPoint], cfg: Dict[str, Any]) -> DuplicateDecision:
    url_norm = item.url.strip().lower()
    key = item.canonical_key
    entity = normalize_entity(item.company_or_project)
    threshold = float(cfg["embedding_similarity_threshold"])
    same_entity_threshold = float(cfg["same_entity_similarity_threshold"])

    best: Tuple[float, Optional[ConceptPoint]] = (0.0, None)
    for old in history:
        if url_norm and old.url.strip().lower() == url_norm:
            return DuplicateDecision(True, "same URL", old.title, old.url, 1.0)
        if key and old.canonical_key == key:
            return DuplicateDecision(True, "same canonical concept key", old.title, old.url, 1.0)

        sim = cosine(item.embedding, old.embedding)
        if sim > best[0]:
            best = (sim, old)

        old_entity = normalize_entity(old.company_or_project)
        if entity and old_entity and entity == old_entity and sim >= same_entity_threshold:
            return DuplicateDecision(True, "same entity and similar concept", old.title, old.url, sim)
        if sim >= threshold:
            return DuplicateDecision(True, "similar concept embedding", old.title, old.url, sim)

    if best[1] is not None:
        return DuplicateDecision(False, "", best[1].title, best[1].url, best[0])
    return DuplicateDecision(False)


def dedupe_new_items(candidates: List[ConceptPoint], history: List[ConceptPoint], cfg: Dict[str, Any]) -> Tuple[List[ConceptPoint], List[Tuple[ConceptPoint, DuplicateDecision]]]:
    novel: List[ConceptPoint] = []
    duplicates: List[Tuple[ConceptPoint, DuplicateDecision]] = []
    rolling_history = list(history)

    for item in candidates:
        decision = duplicate_decision(item, rolling_history, cfg)
        if decision.is_duplicate:
            duplicates.append((item, decision))
        else:
            novel.append(item)
            rolling_history.append(item)
    return novel, duplicates


def write_raw(raw_dir: Path, run_date: str, all_items: List[ConceptPoint], duplicates: List[Tuple[ConceptPoint, DuplicateDecision]]) -> None:
    payload = {
        "run_date": run_date,
        "items": [item.to_history_record() | {"raw": item.raw} for item in all_items],
        "duplicates": [
            {
                "item": item.to_history_record(),
                "decision": dataclasses.asdict(decision),
            }
            for item, decision in duplicates
        ],
    }
    (raw_dir / f"{run_date}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def md_escape(text: str) -> str:
    return (text or "").replace("\n", " ").strip()


def item_to_markdown(item: ConceptPoint, idx: int) -> str:
    record_json = json.dumps(item.to_history_record(), ensure_ascii=False, separators=(",", ":"))
    lines = [
        f"### {idx}. {md_escape(item.title)}",
        f"<!--INFO_ITEM {record_json}-->",
        "",
        f"- **Source:** {md_escape(item.source_type)} / {md_escape(item.source_name)}",
        f"- **Entity:** {md_escape(item.company_or_project) or 'N/A'}",
        f"- **Date:** {md_escape(item.published_at) or 'N/A'}",
        f"- **Concept:** {md_escape(item.concept) or 'N/A'}",
        f"- **URL:** {item.url}",
        f"- **Summary:** {md_escape(item.summary)}",
    ]
    if item.why_relevant:
        lines.append(f"- **Why relevant:** {md_escape(item.why_relevant)}")
    return "\n".join(lines)


def write_report(reports_dir: Path, run_date: str, question: str, novel: List[ConceptPoint], duplicates: List[Tuple[ConceptPoint, DuplicateDecision]]) -> Path:
    path = reports_dir / f"{run_date}.md"
    lines: List[str] = [
        f"# INFO Agent Report — {run_date}",
        "",
        f"**Fixed question:** {question}",
        "",
        f"**Novel concept points:** {len(novel)}",
        f"**Skipped duplicates:** {len(duplicates)}",
        "",
        "## New concept points",
        "",
    ]

    if novel:
        for idx, item in enumerate(novel, 1):
            lines.append(item_to_markdown(item, idx))
            lines.append("")
    else:
        lines.append("No novel concept points found today.")
        lines.append("")

    lines.extend(["## Skipped duplicates", ""])
    if duplicates:
        for idx, (item, decision) in enumerate(duplicates, 1):
            lines.extend(
                [
                    f"### {idx}. {md_escape(item.title)}",
                    f"- **Reason:** {decision.reason}",
                    f"- **Similarity:** {decision.similarity:.3f}",
                    f"- **Matched old title:** {md_escape(decision.matched_title)}",
                    f"- **Matched old URL:** {decision.matched_url}",
                    f"- **Candidate URL:** {item.url}",
                    "",
                ]
            )
    else:
        lines.append("No duplicates skipped.")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def append_history(index_path: Path, novel: List[ConceptPoint]) -> None:
    if not novel:
        return
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        for item in novel:
            f.write(json.dumps(item.to_history_record(), ensure_ascii=False) + "\n")


def collect(cfg: Dict[str, Any], skip_web: bool = False, skip_github: bool = False) -> List[ConceptPoint]:
    items: List[ConceptPoint] = []
    web_enabled = bool(cfg.get("web_enabled", True))
    if not skip_web and web_enabled:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[WARN] OPENAI_API_KEY is missing. Skipping web search. Set web_enabled: false to silence this warning.", file=sys.stderr)
        else:
            try:
                items.extend(web_search_openai(cfg["question"], cfg))
            except Exception as e:
                print(f"[WARN] Web search failed: {e}", file=sys.stderr)
    if not skip_github:
        try:
            items.extend(github_search(cfg))
        except Exception as e:
            print(f"[WARN] GitHub search failed: {e}", file=sys.stderr)

    # Remove exact duplicate URLs inside the same run before embeddings.
    seen_urls: set[str] = set()
    unique: List[ConceptPoint] = []
    for item in items:
        item.canonical_key = item.canonical_key or make_canonical_key(item)
        u = item.url.strip().lower()
        if u in seen_urls:
            continue
        seen_urls.add(u)
        unique.append(item)
    return unique


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--date", default=today_utc().isoformat(), help="Run date for filenames, YYYY-MM-DD")
    parser.add_argument("--skip-web", action="store_true")
    parser.add_argument("--skip-github", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    ensure_dirs(cfg)

    reports_dir = Path(cfg["reports_dir"])
    raw_dir = Path(cfg["raw_dir"])
    index_path = Path(cfg["history_index"])

    print("Collecting candidates...")
    candidates = collect(cfg, skip_web=args.skip_web, skip_github=args.skip_github)
    print(f"Collected {len(candidates)} candidate items.")

    print("Loading history from index and Markdown reports...")
    history = merge_history(
        load_history_from_index(index_path),
        load_history_from_markdown(reports_dir),
    )
    print(f"Loaded {len(history)} historical concept points.")

    # Embed both candidates and history records that are missing embeddings.
    to_embed = [*candidates, *[h for h in history if h.embedding is None]]
    if to_embed:
        print(f"Embedding {len(to_embed)} items for semantic de-duplication...")
        embed_items(to_embed, cfg)

    novel, duplicates = dedupe_new_items(candidates, history, cfg)
    print(f"Novel: {len(novel)}. Duplicates: {len(duplicates)}.")

    write_raw(raw_dir, args.date, candidates, duplicates)
    report_path = write_report(reports_dir, args.date, cfg["question"], novel, duplicates)
    append_history(index_path, novel)

    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
