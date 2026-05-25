import os
import platform
import traceback

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl() -> bool:
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled

def get_env_enable_pdl_for_kernel(disable_env_name: str) -> bool:
    enabled = get_env_enable_pdl()
    disabled = enabled and os.environ.get(disable_env_name, "0") == "1"
    printed_key = f"_printed_{disable_env_name}"
    if disabled and not getattr(get_env_enable_pdl_for_kernel, printed_key, False):
        logger.info("PDL disabled by %s", disable_env_name)
        setattr(get_env_enable_pdl_for_kernel, printed_key, True)
    return enabled and not disabled


if platform.system() != "Windows":
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )
