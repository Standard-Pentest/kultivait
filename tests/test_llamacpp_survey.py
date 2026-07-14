from kultivait.cli import _local_llamacpp_models, match_gguf_sizes


def _entry(model_id, args):
    return {"id": model_id, "status": {"value": "unloaded", "args": args}}


def test_matches_router_ids_to_cache_filenames():
    # router model ids and cache filenames drift: path prefixes flattened
    # to underscores, :quant suffixes, .gguf sometimes present, case varies
    names = ["ggml-org/gemma-3-4b-it-GGUF", "qwen2.5-14b-instruct-q4_k_m.gguf"]
    files = {
        "ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf": 3_000_000_000,
        "qwen2.5-14b-instruct-q4_k_m.gguf": 9_000_000_000,
    }
    sizes = match_gguf_sizes(names, files)
    assert sizes["qwen2.5-14b-instruct-q4_k_m.gguf"] == 9_000_000_000
    assert sizes["ggml-org/gemma-3-4b-it-GGUF"] == 3_000_000_000


def test_unmatched_names_are_omitted_not_zeroed():
    # missing size is fine: _param_billions still reads "14b" from the name
    sizes = match_gguf_sizes(["qwen2.5-14b.gguf"], {})
    assert sizes == {}


def test_survey_excludes_undownloaded_hf_suggestions(tmp_path):
    # a router /v1/models listing mixes on-disk models (--model <path>) with
    # downloadable HF suggestions (--hf-repo, nothing on disk); surveying a
    # phantom model would trigger a surprise multi-GB fetch on first route
    gguf = tmp_path / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    gguf.write_bytes(b"\0" * 1024)
    entries = [
        _entry(
            "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
            ["llama-server", "--hf-repo", "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"],
        ),
        _entry(
            "qwen2.5-0.5b-instruct-q4_k_m",
            ["llama-server", "--model", str(gguf)],
        ),
    ]
    names, sizes = _local_llamacpp_models(entries, cache_files={})
    assert names == ["qwen2.5-0.5b-instruct-q4_k_m"]
    assert sizes == {"qwen2.5-0.5b-instruct-q4_k_m": 1024}


def test_survey_includes_hf_models_already_in_cache():
    entries = [
        _entry(
            "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
            ["llama-server", "--hf-repo", "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"],
        ),
    ]
    cache = {
        "mistralai_Ministral-3-8B-Instruct-2512-GGUF_Ministral-3-8B-Instruct-2512-Q4_K_M.gguf": 5_000_000_000
    }
    names, sizes = _local_llamacpp_models(entries, cache_files=cache)
    assert names == ["mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"]
    assert sizes[names[0]] == 5_000_000_000
