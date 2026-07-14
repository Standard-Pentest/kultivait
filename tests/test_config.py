from kultivait.config import Config, TierSpec, detect, load_config, save_config


def test_config_roundtrip_llamacpp_runtime(tmp_path):
    config = Config(
        runtime="llamacpp",
        chat_base_url="http://localhost:8080",
        embed_base_url="http://localhost:8081",
        embed_model="nomic-embed-text-v1.5.Q8_0.gguf",
        distill_model="qwen2.5-14b-instruct-q4_k_m.gguf",
        tiers=[
            TierSpec(
                name="gemma-3-4b-it-Q4_K_M.gguf", role="simple",
                kind="llamacpp", model="gemma-3-4b-it-Q4_K_M.gguf",
            ),
        ],
    )
    path = tmp_path / "config.toml"
    save_config(config, path)
    assert load_config(path) == config


def test_embed_url_falls_back_to_chat_url():
    assert Config(chat_base_url="http://x:1").embed_url() == "http://x:1"
    assert (
        Config(chat_base_url="http://x:1", embed_base_url="http://y:2").embed_url()
        == "http://y:2"
    )


def test_detect_llamacpp_maps_gguf_names_to_tiers():
    config = detect(
        ollama_models=[
            "gemma-3-4b-it-Q4_K_M.gguf",
            "qwen2.5-14b-instruct-q4_k_m.gguf",
            "nomic-embed-text-v1.5.Q8_0.gguf",
        ],
        available_clis=[],
        runtime="llamacpp",
    )
    roles = {t.role: t for t in config.tiers}
    assert roles["simple"].model == "gemma-3-4b-it-Q4_K_M.gguf"
    assert roles["simple"].kind == "llamacpp"
    assert roles["reasoning"].model == "qwen2.5-14b-instruct-q4_k_m.gguf"
    assert config.embed_model == "nomic-embed-text-v1.5.Q8_0.gguf"
    assert config.runtime == "llamacpp"
    assert config.chat_base_url == "http://localhost:8080"


def test_detect_llamacpp_sizes_unparseable_gguf_by_bytes():
    # a GGUF whose filename carries no parameter count: disk size decides
    config = detect(
        ollama_models=["mystery-model.gguf", "tiny.gguf"],
        available_clis=[],
        sizes={"mystery-model.gguf": 9_000_000_000, "tiny.gguf": 500_000_000},
        runtime="llamacpp",
    )
    roles = {t.role: t for t in config.tiers}
    # 9 GB / 0.75 ≈ 12B: above the simple floor; 0.5 GB is sub-floor
    assert roles["reasoning"].model == "mystery-model.gguf"
    assert roles["simple"].model == "mystery-model.gguf"


def test_config_roundtrip(tmp_path):
    config = Config(
        num_ctx=16384,
        embed_model="nomic-embed-text",
        distill_model="qwen2.5:14b",
        tiers=[
            TierSpec(name="mistral:7b", role="simple", kind="ollama", model="mistral:7b"),
            TierSpec(name="qwen2.5:14b", role="reasoning", kind="ollama", model="qwen2.5:14b"),
            TierSpec(
                name="claude", role="architect", kind="cli",
                command=["claude"], price_in=3.0, price_out=15.0,
            ),
        ],
    )
    path = tmp_path / "config.toml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded == config
    assert loaded.capability_order() == ["mistral:7b", "qwen2.5:14b", "claude"]


def test_detect_maps_models_to_tiers_by_size():
    config = detect(
        ollama_models=["mistral:7b", "qwen2.5:14b", "nomic-embed-text:latest"],
        available_clis=["claude"],
    )
    roles = {t.role: t for t in config.tiers}
    assert roles["simple"].model == "mistral:7b"
    assert roles["reasoning"].model == "qwen2.5:14b"
    assert roles["architect"].command == ["claude"]
    assert config.embed_model == "nomic-embed-text:latest"
    assert config.distill_model == "qwen2.5:14b"  # largest general model distills


def test_detect_local_only_keeps_virtual_frontier_tier():
    # No cloud CLIs: classification must still recognize cloud-worthy prompts
    # so the escalation-brief path fires. The tier exists; no backend will.
    config = detect(
        ollama_models=["llama3.1:8b", "nomic-embed-text:latest"],
        available_clis=[],
    )
    kinds = {t.role: t.kind for t in config.tiers}
    assert kinds["architect"] == "virtual"
    assert kinds["docs"] == "virtual"


def test_detect_single_model_covers_both_local_roles():
    config = detect(
        ollama_models=["llama3.1:8b", "nomic-embed-text:latest"],
        available_clis=[],
    )
    roles = {t.role: t for t in config.tiers}
    assert roles["simple"].model == "llama3.1:8b"
    assert roles["reasoning"].model == "llama3.1:8b"


def test_unparseable_model_names_fall_back_to_byte_size():
    # "gemma4:latest" has no parameter count in its name; without a size
    # estimate it sorted as smallest and became the simple tier. Bytes tell
    # the truth: ~9.6GB q4 is a ~13B-class model.
    config = detect(
        ollama_models=["gemma4:latest", "llama3.1:8b", "nomic-embed-text:latest"],
        available_clis=[],
        sizes={"gemma4:latest": 9_600_000_000, "llama3.1:8b": 4_900_000_000},
    )
    roles = {t.role: t for t in config.tiers}
    assert roles["simple"].model == "llama3.1:8b"
    assert roles["reasoning"].model == "gemma4:latest"


def test_simple_tier_has_a_4b_floor():
    # A 1.7B model is too weak to be anyone's default; prefer the smallest
    # model at or above 4B when one exists.
    config = detect(
        ollama_models=["qwen3:1.7b", "llama3.1:8b", "qwen3:14b", "nomic-embed-text:latest"],
        available_clis=[],
    )
    roles = {t.role: t for t in config.tiers}
    assert roles["simple"].model == "llama3.1:8b"
    assert roles["reasoning"].model == "qwen3:14b"


def test_detect_flags_missing_embedding_model():
    config = detect(ollama_models=["llama3.1:8b"], available_clis=[])
    assert config.embed_model is None  # init must offer to pull one


def test_detect_ignores_embedding_models_for_generation():
    config = detect(
        ollama_models=["bge-m3:latest", "llama3.1:8b"], available_clis=[]
    )
    roles = {t.role: t for t in config.tiers}
    assert roles["simple"].model == "llama3.1:8b"
    assert config.embed_model == "bge-m3:latest"
