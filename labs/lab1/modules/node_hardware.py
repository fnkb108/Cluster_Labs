"""
Модуль 2: Характеристики оборудования узлов и вычислительная производительность (GFlop/sec, RAM, ядра)
"""

import os
import sys
import time

# Добавляем корневую директорию проекта в sys.path при прямом (автономном) запуске модуля
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from mpi4py import MPI
from utils.sysinfo import get_node_hardware_summary
from utils.formatting import format_table

def benchmark_gemm_gflops(matrix_sizes=None, repetitions=3):
    """
    Измеряет производительность вычислений с плавающей точкой на примере
    матричного умножения двойной точности (GEMM) по минимальному времени выполнения.
    """
    if matrix_sizes is None:
        matrix_sizes = [500, 1000, 1500]

    results = []
    for N in matrix_sizes:
        A = np.random.randn(N, N).astype(np.float64)
        B = np.random.randn(N, N).astype(np.float64)
        _ = np.dot(A, B)

        times = []
        flops = 2.0 * (N ** 3)
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = np.dot(A, B)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        min_time = min(times)
        peak_gflops = (flops / min_time) / 1e9 if min_time > 0 else 0.0

        results.append({
            "matrix_dim": N,
            "min_time_sec": min_time,
            "peak_gflops": peak_gflops,
        })

    best_gemm = max(results, key=lambda x: x["peak_gflops"])
    return {
        "detailed": results,
        "best_gflops": best_gemm["peak_gflops"],
        "best_matrix_dim": best_gemm["matrix_dim"]
    }


def benchmark_synthetic_fma_gflops(num_elements=5_000_000, loops=20, repetitions=3):
    """
    Синтетический тест с плавающей точкой: векторное умножение со сложением (FMA-подобная арифметика)
    y = y * alpha + beta
    2 операции на каждый элемент за одну итерацию цикла.
    Общее число операций: FLOP = 2 * num_elements * loops
    """
    x = np.ones(num_elements, dtype=np.float64)
    alpha = 1.0000001
    beta = 0.0000001

    # Прогрев
    x = x * alpha + beta

    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        for _ in range(loops):
            x = x * alpha + beta
        t1 = time.perf_counter()
        times.append(t1 - t0)

    min_elapsed = min(times)
    total_flops = 2.0 * num_elements * loops
    gflops = (total_flops / min_elapsed) / 1e9 if min_elapsed > 0 else 0.0

    return {
        "elapsed_sec": min_elapsed,
        "gflops": gflops,
        "num_elements": num_elements,
        "loops": loops
    }


def run_node_hardware_benchmarks(comm):
    """
    Собирает аппаратные характеристики с каждого ранга, запускает вычислительные бенчмарки
    одновременно на всех процессах и выводит сводный отчет по кластеру.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    processor_name = MPI.Get_processor_name()

    if rank == 0:
        print(f"\n=======================================================")
        print(f"  ПУНКТ 2: Характеристики оборудования узлов и производительность")
        print(f"=======================================================")
        print(f"[*] Сбор данных об оборудовании и замер GFlop/s...")

    hw_info = get_node_hardware_summary()
    hw_info["mpi_rank"] = rank
    hw_info["mpi_proc_name"] = processor_name

    comm.Barrier()
    gemm_res = benchmark_gemm_gflops(matrix_sizes=[500, 1000, 1500], repetitions=3)
    fma_res = benchmark_synthetic_fma_gflops(num_elements=5_000_000, loops=20, repetitions=3)

    hw_info["gemm_peak_gflops"] = gemm_res["best_gflops"]
    hw_info["gemm_details"] = gemm_res["detailed"]
    hw_info["fma_gflops"] = fma_res["gflops"]

    all_ranks_data = comm.gather(hw_info, root=0)
    results_summary = None

    if rank == 0:
        nodes_dict = {}
        for d in all_ranks_data:
            h = d["hostname"]
            nodes_dict.setdefault(h, []).append(d)

        headers = [
            "Node Hostname",
            "MPI Ranks",
            "CPU Model",
            "Phys Cores",
            "Log Cores",
            "RAM Total",
            "RAM Avail",
            "Peak GEMM (GFlop/s)",
            "Vector FMA (GFlop/s)"
        ]

        rows = []
        total_cluster_phys_cores = 0
        total_cluster_log_cores = 0
        total_cluster_ram_gb = 0.0
        total_cluster_gemm_gflops = 0.0

        for h, rank_list in nodes_dict.items():
            first_rank_data = rank_list[0]
            ranks_str = ",".join(str(r["mpi_rank"]) for r in rank_list)
            if len(ranks_str) > 12:
                ranks_str = f"{rank_list[0]['mpi_rank']}..{rank_list[-1]['mpi_rank']} ({len(rank_list)})"

            node_gemm_gflops = sum(r["gemm_peak_gflops"] for r in rank_list)
            node_fma_gflops = sum(r["fma_gflops"] for r in rank_list)

            phys_cores = first_rank_data["physical_cores"]
            log_cores = first_rank_data["logical_cores"]
            ram_tot = first_rank_data["ram_total_gb"]
            ram_avail = first_rank_data["ram_avail_gb"]

            total_cluster_phys_cores += phys_cores
            total_cluster_log_cores += log_cores
            total_cluster_ram_gb += ram_tot
            total_cluster_gemm_gflops += node_gemm_gflops

            rows.append([
                h,
                ranks_str,
                first_rank_data["cpu_model"][:30],
                str(phys_cores),
                str(log_cores),
                f"{ram_tot:.1f} GB",
                f"{ram_avail:.1f} GB",
                f"{node_gemm_gflops:.2f}",
                f"{node_fma_gflops:.2f}"
            ])

        print("\n--- Per-Node Hardware & Performance Summary ---")
        print(format_table(headers, rows, ['<', '<', '<', '>', '>', '>', '>', '>', '>']))

        print("\n--- Cluster Totals ---")
        print(f"  * Total Physical Nodes:            {len(nodes_dict)}")
        print(f"  * Total MPI Processes (Ranks):     {size}")
        print(f"  * Total Physical Cores:            {total_cluster_phys_cores}")
        print(f"  * Total Logical Cores:             {total_cluster_log_cores}")
        print(f"  * Total Cluster RAM:               {total_cluster_ram_gb:.2f} GB")
        print(f"  * Aggregated Cluster Performance:  {total_cluster_gemm_gflops:.2f} GFlop/sec")

        results_summary = {
            "all_ranks_data": all_ranks_data,
            "unique_nodes": list(nodes_dict.keys()),
            "total_phys_cores": total_cluster_phys_cores,
            "total_log_cores": total_cluster_log_cores,
            "total_ram_gb": total_cluster_ram_gb,
            "total_cluster_gflops": total_cluster_gemm_gflops,
            "table_rows": rows
        }

    return results_summary


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    run_node_hardware_benchmarks(comm)