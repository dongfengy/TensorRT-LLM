from types import SimpleNamespace

from tensorrt_llm._torch.attention_backend.fmha.phased import PhasedFmha
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2


def _get_total_num_blocks(manager: SimpleNamespace, kv_factor: int = 2) -> int:
    fmha = SimpleNamespace(kv_factor=kv_factor)
    return PhasedFmha._get_total_num_blocks(fmha, SimpleNamespace(kv_cache_manager=manager))


def test_v2_primary_pool_page_bound_is_not_rescaled() -> None:
    manager = SimpleNamespace(blocks_in_primary_pool=50_000_000)

    assert (
        KVCacheManagerV2.get_primary_pool_page_index_upper_bound(manager)
        == manager.blocks_in_primary_pool
    )


def test_phased_fmha_uses_manager_page_index_upper_bound() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=50_000_000,
        num_local_layers=36,
        get_primary_pool_page_index_upper_bound=lambda: 50_000_000,
    )
    assert _get_total_num_blocks(manager) == 50_000_000


def test_phased_fmha_preserves_legacy_pool_scaling() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=1024,
        num_local_layers=36,
    )
    assert _get_total_num_blocks(manager, kv_factor=2) == 1024 * 36 * 2
