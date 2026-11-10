"""Apply, reset or report the CMP 40HX tuning profile (power, clock lock, fans, P0 offsets).

Driver state resets on reboot, sleep/resume and driver reload; the `EmiliaGPUProfile`
scheduled task (see install-gpu-profile-task.ps1) runs `apply` as SYSTEM to restore it.
`apply` and `reset` require Administrator/SYSTEM; `status` does not.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import sys
from pathlib import Path
from typing import Any

import pynvml

POWER_LIMIT_MW = 220_000
LOCKED_CLOCKS_MHZ = (300, 1980)
FAN_PERCENT = 100
CORE_OFFSET_KHZ = 105_000
MEMORY_OFFSET_KHZ = 500_000

LOG_PATH = Path(__file__).resolve().parent / "temp" / "gpu_profile.log"

# NvAPI_GPU_GetPstates20/SetPstates20 V2 buffer (7416 bytes). Int32 indices verified on driver 582.78:
# [2] numPstates, [3] numClocks, [5] pstateId, [7]/[18] clock domain, [10]/[21] frequency delta (kHz).
_PSTATES20_BYTES = 7416
_PSTATES20_VERSION = _PSTATES20_BYTES | (2 << 16)
_DOMAIN_GRAPHICS = 0
_DOMAIN_MEMORY = 4
_NVAPI_INITIALIZE = 0x0150E828
_NVAPI_ENUM_PHYSICAL_GPUS = 0xE5AC921F
_NVAPI_GET_PSTATES20 = 0x6FF81213
_NVAPI_SET_PSTATES20 = 0x0F4DAE6B
_NVAPI_GET_ERROR_MESSAGE = 0x6C2D048C


class NvApi:
    """Minimal NvAPI binding for P0 clock offsets on the first physical GPU."""

    def __init__(self) -> None:
        dll = ctypes.WinDLL("nvapi64.dll")
        self._query = dll.nvapi_QueryInterface
        self._query.restype = ctypes.c_void_p
        self._query.argtypes = [ctypes.c_uint32]
        self._check(self._fn(_NVAPI_INITIALIZE)(), "NvAPI_Initialize")
        handles = (ctypes.c_void_p * 64)()
        count = ctypes.c_uint32(0)
        self._check(
            self._fn(_NVAPI_ENUM_PHYSICAL_GPUS, ctypes.c_void_p, ctypes.c_void_p)(
                ctypes.byref(handles), ctypes.byref(count)
            ),
            "NvAPI_EnumPhysicalGPUs",
        )
        if count.value < 1:
            raise RuntimeError("NvAPI found no physical NVIDIA GPU.")
        self._gpu = handles[0]

    def _fn(self, function_id: int, *argtypes: Any) -> Any:
        pointer = self._query(function_id)
        if not pointer:
            raise RuntimeError(f"NvAPI function 0x{function_id:08X} is unavailable.")
        return ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(pointer)

    def _check(self, status: int, name: str) -> None:
        if status == 0:
            return
        message = ctypes.create_string_buffer(256)
        self._fn(_NVAPI_GET_ERROR_MESSAGE, ctypes.c_int, ctypes.c_char_p)(status, message)
        raise RuntimeError(f"{name} failed: {status} {message.value.decode(errors='replace')}")

    def offsets_khz(self) -> tuple[int, int]:
        buffer = (ctypes.c_int32 * (_PSTATES20_BYTES // 4))()
        buffer[0] = _PSTATES20_VERSION
        self._check(
            self._fn(_NVAPI_GET_PSTATES20, ctypes.c_void_p, ctypes.c_void_p)(self._gpu, ctypes.byref(buffer)),
            "NvAPI_GPU_GetPstates20",
        )
        return buffer[10], buffer[21]

    def set_offsets_khz(self, core: int, memory: int) -> None:
        buffer = (ctypes.c_int32 * (_PSTATES20_BYTES // 4))()
        buffer[0] = _PSTATES20_VERSION
        buffer[2] = 1  # one P-state: P0
        buffer[3] = 2  # two clock entries
        buffer[5] = 0  # P0
        buffer[7] = _DOMAIN_GRAPHICS
        buffer[10] = core
        buffer[18] = _DOMAIN_MEMORY
        buffer[21] = memory
        self._check(
            self._fn(_NVAPI_SET_PSTATES20, ctypes.c_void_p, ctypes.c_void_p)(self._gpu, ctypes.byref(buffer)),
            "NvAPI_GPU_SetPstates20",
        )


def _state(handle: Any, nvapi: NvApi) -> dict[str, int]:
    core, memory = nvapi.offsets_khz()
    return {
        "power_limit_mw": pynvml.nvmlDeviceGetPowerManagementLimit(handle),
        "fan_targets": [
            pynvml.nvmlDeviceGetTargetFanSpeed(handle, fan) for fan in range(pynvml.nvmlDeviceGetNumFans(handle))
        ],
        "core_offset_khz": core,
        "memory_offset_khz": memory,
        "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
    }


def _apply(handle: Any, nvapi: NvApi) -> list[str]:
    """Write only settings that drifted; locked clocks have no NVML getter and are always rewritten."""
    changes: list[str] = []
    if pynvml.nvmlDeviceGetPowerManagementLimit(handle) != POWER_LIMIT_MW:
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, POWER_LIMIT_MW)
        changes.append(f"power_limit={POWER_LIMIT_MW // 1000}W")
    pynvml.nvmlDeviceSetGpuLockedClocks(handle, *LOCKED_CLOCKS_MHZ)
    for fan in range(pynvml.nvmlDeviceGetNumFans(handle)):
        if pynvml.nvmlDeviceGetTargetFanSpeed(handle, fan) != FAN_PERCENT:
            pynvml.nvmlDeviceSetFanSpeed_v2(handle, fan, FAN_PERCENT)
            changes.append(f"fan{fan}={FAN_PERCENT}%")
    if nvapi.offsets_khz() != (CORE_OFFSET_KHZ, MEMORY_OFFSET_KHZ):
        nvapi.set_offsets_khz(CORE_OFFSET_KHZ, MEMORY_OFFSET_KHZ)
        changes.append(f"offsets=core+{CORE_OFFSET_KHZ // 1000}/mem+{MEMORY_OFFSET_KHZ // 1000}MHz")
    if nvapi.offsets_khz() != (CORE_OFFSET_KHZ, MEMORY_OFFSET_KHZ):
        raise RuntimeError("P0 offsets did not read back after NvAPI_GPU_SetPstates20.")
    return changes


def _reset(handle: Any, nvapi: NvApi) -> None:
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle))
    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    for fan in range(pynvml.nvmlDeviceGetNumFans(handle)):
        pynvml.nvmlDeviceSetDefaultFanSpeed_v2(handle, fan)
    nvapi.set_offsets_khz(0, 0)


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"{stamp} {message}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("apply", "reset", "status"))
    command = parser.parse_args().command
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            nvapi = NvApi()
            if command == "apply":
                changes = _apply(handle, nvapi)
                if changes:
                    _log("apply " + ", ".join(changes))
            elif command == "reset":
                _reset(handle, nvapi)
                _log("reset to driver defaults")
            print(_state(handle, nvapi))
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        if command != "status":
            _log(f"{command} FAILED: {exc!r}")
        print(f"{command} failed: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
