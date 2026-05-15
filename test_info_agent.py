from info_agent import ConceptPoint, embed_items, load_config


def test_load_config_defaults_include_embedding_endpoint_fields(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('question: "q"\n', encoding='utf-8')

    cfg = load_config(cfg_file)

    assert cfg["embedding_model"] == "text-embedding-3-small"
    assert cfg["embedding_base_url"] == ""
    assert cfg["embedding_api_key"] == ""


def test_embed_items_uses_local_base_url_without_openai_api_key(monkeypatch):
    calls = {}

    class DummyEmbData:
        def __init__(self, embedding):
            self.embedding = embedding

    class DummyEmbeddings:
        def create(self, model, input):
            calls["model"] = model
            calls["input"] = input
            return type("Resp", (), {"data": [DummyEmbData([0.1, 0.2]) for _ in input]})

    class DummyClient:
        def __init__(self, base_url=None, api_key=None):
            calls["base_url"] = base_url
            calls["api_key"] = api_key
            self.embeddings = DummyEmbeddings()

    monkeypatch.setattr("info_agent.OpenAI", DummyClient)

    items = [
        ConceptPoint(
            title="Title",
            url="https://example.com",
            source_type="web",
            company_or_project="OpenAI",
            concept="adds thing",
            summary="summary",
            why_relevant="relevant",
        )
    ]

    cfg = {
        "embedding_model": "Qwen3-Embedding-8B-Q4_K_M",
        "embedding_base_url": "http://172.28.21.22:11434/v1",
        "embedding_api_key": "",
    }

    embed_items(items, cfg)

    assert calls["base_url"] == "http://172.28.21.22:11434/v1"
    assert calls["api_key"] == "not-needed"
    assert calls["model"] == "Qwen3-Embedding-8B-Q4_K_M"
    assert isinstance(items[0].embedding, list)
    assert len(items[0].embedding) == 2
