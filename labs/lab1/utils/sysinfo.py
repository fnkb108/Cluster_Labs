"""
Сборщик системной информации и характеристик оборудования.
Работает «из коробки» на Linux и Windows.
Внешние зависимости не требуются (библиотека psutil используется опционально, если установлена).
"""

import os
import platform
import socket
import sys

def get_cpu_info():
    """Возвращает словарь с моделью процессора, количеством физических и логических ядер."""
    info = {
        "model": "Unknown CPU",
        "physical_cores": None,
        "logical_cores": os.cpu_count() or 1,
    }

    # Пробуем использовать psutil, если библиотека доступна
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
    except Exception:
        pass

    # Парсинг /proc/cpuinfo в Linux
    if sys.platform.startswith("linux") and os.path.exists("/proc/cpuinfo"):
        try:
            cores_set = set()
            curr_phys_id = 0
            curr_core_id = 0
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        k, v = [x.strip() for x in line.split(":", 1)]
                        if k == "model name" and info["model"] == "Unknown CPU":
                            info["model"] = v
                        elif k == "physical id":
                            curr_phys_id = v
                        elif k == "core id":
                            curr_core_id = v
                            cores_set.add((curr_phys_id, curr_core_id))
            if cores_set:
                info["physical_cores"] = len(cores_set)
        except Exception:
            pass

    # Определение параметров CPU в Windows через реестр
    elif sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val:
                info["model"] = val.strip()
        except Exception:
            info["model"] = platform.processor() or "x86_64 CPU"

    if info["physical_cores"] is None:
        # Резервная эвристика: логические ядра // 2 при наличии Hyper-Threading, иначе равно логическим
        info["physical_cores"] = max(1, info["logical_cores"] // 2) if info["logical_cores"] > 1 else 1

    return info


def get_ram_info():
    """Возвращает словарь с общим (total_gb), доступным (available_gb) объемом ОЗУ и размером в байтах (total_bytes)."""
    info = {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "total_bytes": 0,
    }

    # Пробуем использовать psutil, если библиотека присутствует
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_bytes"] = vm.total
        info["total_gb"] = round(vm.total / (1024 ** 3), 2)
        info["available_gb"] = round(vm.available / (1024 ** 3), 2)
        return info
    except Exception:
        pass

    # Чтение /proc/meminfo в Linux
    if sys.platform.startswith("linux") and os.path.exists("/proc/meminfo"):
        try:
            mem_total_kb = 0
            mem_avail_kb = 0
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        k = parts[0].rstrip(":")
                        if k == "MemTotal":
                            mem_total_kb = int(parts[1])
                        elif k == "MemAvailable":
                            mem_avail_kb = int(parts[1])
            info["total_bytes"] = mem_total_kb * 1024
            info["total_gb"] = round(mem_total_kb / (1024 ** 2), 2)
            info["available_gb"] = round(mem_avail_kb / (1024 ** 2), 2)
            return info
        except Exception:
            pass

    # Вызов Win32 API GlobalMemoryStatusEx через ctypes в Windows
    if sys.platform == "win32":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["total_bytes"] = stat.ullTotalPhys
            info["total_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 2)
            info["available_gb"] = round(stat.ullAvailPhys / (1024 ** 3), 2)
            return info
        except Exception:
            pass

    return info


def get_node_hardware_summary():
    """Собирает все аппаратные характеристики узла в единый словарь."""
    cpu = get_cpu_info()
    ram = get_ram_info()
    hostname = socket.gethostname()
    try:
        ip_addr = socket.gethostbyname(hostname)
    except Exception:
        ip_addr = "127.0.0.1"

    return {
        "hostname": hostname,
        "ip": ip_addr,
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": cpu["model"],
        "physical_cores": cpu["physical_cores"],
        "logical_cores": cpu["logical_cores"],
        "ram_total_gb": ram["total_gb"],
        "ram_avail_gb": ram["available_gb"],
    }


if __name__ == "__main__":
    import json

    if "--json" in sys.argv:
        print(json.dumps(get_node_hardware_summary(), indent=2, ensure_ascii=False))
    else:
        info = get_node_hardware_summary()
        print("=" * 55)
        print("         NODE HARDWARE SPECIFICATIONS")
        print("=" * 55)
        print(f"  Hostname:       {info['hostname']}")
        print(f"  IP Address:     {info['ip']}")
        print(f"  OS / Platform:  {info['os']}")
        print(f"  CPU Model:      {info['cpu_model']}")
        print(f"  Physical Cores: {info['physical_cores']}")
        print(f"  Logical Cores:  {info['logical_cores']}")
        print(f"  RAM Total:      {info['ram_total_gb']} GB")
        print(f"  RAM Available:  {info['ram_avail_gb']} GB")
        print("=" * 55)