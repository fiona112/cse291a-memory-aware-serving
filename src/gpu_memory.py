try:
    import pynvml
except ImportError:
    pynvml = None


_nvml_ready = None


def _init_nvml():
    global _nvml_ready

    if _nvml_ready is None:
        try:
            pynvml.nvmlInit()
            _nvml_ready = True

        except Exception:
            _nvml_ready = False

    return _nvml_ready


def get_vram_mb(pid):
    """
    Returns (vram_mb, scope).

    scope is "process" when the driver reports the
    llama-server process's own VRAM (Linux), "device"
    when it only reports whole-GPU usage (Windows WDDM),
    and None when no NVIDIA GPU is available.
    """
    if pynvml is None or not _init_nvml():
        return None, None

    try:
        device_count = pynvml.nvmlDeviceGetCount()

        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(device_count)
        ]

        process_bytes = 0
        found_process = False

        for handle in handles:
            for process in (
                pynvml.nvmlDeviceGetComputeRunningProcesses(
                    handle
                )
            ):
                if (
                    process.pid == pid
                    and process.usedGpuMemory is not None
                ):
                    process_bytes += process.usedGpuMemory
                    found_process = True

        if found_process:
            return process_bytes / (1024 * 1024), "process"

        device_bytes = sum(
            pynvml.nvmlDeviceGetMemoryInfo(handle).used
            for handle in handles
        )

        return device_bytes / (1024 * 1024), "device"

    except Exception:
        return None, None
