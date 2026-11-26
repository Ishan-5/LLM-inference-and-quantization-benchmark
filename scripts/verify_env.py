"""Print a compact environment report without loading any model."""

from llmbench.hardware import get_device_info


def main() -> None:
    info = get_device_info()
    for key in (
        "python",
        "platform",
        "torch",
        "cuda_available",
        "gpu_name",
        "gpu_vram_mib",
        "cuda_capability",
        "transformers",
        "vllm",
        "llama_cpp",
        "autoawq",
        "auto_gptq",
    ):
        print(f"{key:20}: {info.get(key)}")


if __name__ == "__main__":
    main()
