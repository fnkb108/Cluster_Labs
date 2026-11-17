"""
ПУНКТ 1. Сетевой бенчмарк
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from mpi4py import MPI
from utils.formatting import format_table, format_bytes


def _ping_pong_master(comm, target_rank, buffer_size, iterations, warmup=10):
    """
    Процесс-инициатор (отправляет буфер байтов, засекает время до возврата этого же буфера).
    Возвращает список из RTT в секундах (Round-Trip Time).
    """
    send_buf = np.ones(buffer_size, dtype=np.uint8) if buffer_size > 0 else np.empty(0, dtype=np.uint8)
    recv_buf = np.empty(buffer_size, dtype=np.uint8) if buffer_size > 0 else np.empty(0, dtype=np.uint8)

    for _ in range(warmup):
        comm.Send([send_buf, MPI.BYTE], dest=target_rank, tag=77)
        comm.Recv([recv_buf, MPI.BYTE], source=target_rank, tag=77)

    rtts = []
    for _ in range(iterations):
        t_start = MPI.Wtime()
        comm.Send([send_buf, MPI.BYTE], dest=target_rank, tag=88)
        comm.Recv([recv_buf, MPI.BYTE], source=target_rank, tag=88)
        t_end = MPI.Wtime()
        rtts.append(t_end - t_start)

    return rtts


def _ping_pong_worker(comm, master_rank, buffer_size, iterations, warmup=10):
    """
    Принимает сообщение и отправляет его обратно (Процесс-ответчик).
    """
    buf = np.empty(buffer_size, dtype=np.uint8) if buffer_size > 0 else np.empty(0, dtype=np.uint8)

    for _ in range(warmup):
        comm.Recv([buf, MPI.BYTE], source=master_rank, tag=77)
        comm.Send([buf, MPI.BYTE], dest=master_rank, tag=77)

    for _ in range(iterations):
        comm.Recv([buf, MPI.BYTE], source=master_rank, tag=88)
        comm.Send([buf, MPI.BYTE], dest=master_rank, tag=88)


def measure_small_packet_latency(comm, rank1, rank2, packet_sizes=None, iterations=1000):
    """
    Замеряет задержку на малых пакетах между процессами rank1 и rank2.
    Возвращает список словарей с результатами.
    """
    if packet_sizes is None:
        packet_sizes = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

    rank = comm.Get_rank()
    results = []

    for size in packet_sizes:
        if rank == rank1:
            rtts = _ping_pong_master(comm, target_rank=rank2, buffer_size=size, iterations=iterations, warmup=50)
            min_one_way_sec = float(np.min(rtts)) / 2.0
            min_lat_us = min_one_way_sec * 1e6
            mb_per_sec = (size / min_one_way_sec) / 1e6 if min_one_way_sec > 0 and size > 0 else 0.0

            results.append({
                "size_bytes": size,
                "min_us": min_lat_us,
                "mb_per_sec": mb_per_sec,
            })
        elif rank == rank2:
            _ping_pong_worker(comm, master_rank=rank1, buffer_size=size, iterations=iterations, warmup=50)

        return results if rank == rank1 else None

def measure_bandwidth_vs_size(comm, rank1, rank2, max_size_kb=128):
    """
    Замеряет скорость передачи данных как функцию размера пакета между rank1 и rank2.
    Изменяет размер пакета от 1 байта до max_size_kb КБ.
    Возвращает список словарей.
    """
    rank = comm.Get_rank()

    max_bytes = int(max_size_kb * 1024)
    sizes_set = set()
    s = 1
    while s <= max_bytes:
        sizes_set.add(s)
        s *= 2
    for extra_kb in (10, 100):
        extra_b = extra_kb * 1024
        if extra_b <= max_bytes:
            sizes_set.add(extra_b)
    if max_bytes > 0:
        sizes_set.add(max_bytes)
    sizes = sorted(sizes_set)

    results = []

    for size in sizes:
        if size <= 1024:
            iters, warmup = 500, 30
        else:
            iters, warmup = 250, 20

        if rank == rank1:
            rtts = _ping_pong_master(comm, target_rank=rank2, buffer_size=size, iterations=iters, warmup=warmup)
            min_rtt = float(np.min(rtts))
            one_way_sec = min_rtt / 2.0
            one_way_us = one_way_sec * 1e6
            mb_s = (size / one_way_sec) / 1e6 if one_way_sec > 0 else 0.0
            mib_s = (size / one_way_sec) / (1024 * 1024) if one_way_sec > 0 else 0.0
            mbit_s = (size * 8 / one_way_sec) / 1e6 if one_way_sec > 0 else 0.0

            results.append({
                "size_bytes": size,
                "size_fmt": format_bytes(size),
                "one_way_us": one_way_us,
                "mb_per_sec": mb_s,
                "mib_per_sec": mib_s,
                "mbit_per_sec": mbit_s,
                "iterations": iters
            })
        elif rank == rank2:
            _ping_pong_worker(comm, master_rank=rank1, buffer_size=size, iterations=iters, warmup=warmup)

    return results if rank == rank1 else None


def fit_hockney_model(bandwidth_results):
    """
    Строит математическую модель коммуникационной среды Хокни: T(s) = alpha + s / beta
    где:
      alpha: стартовое время задержки
      beta: асимптотическая пропускная способность
      n_1/2 = alpha * beta: размер сообщения половинной производительности (beta / 2)
    """
    if not bandwidth_results or len(bandwidth_results) < 4:
        return None

    sizes = np.array([r["size_bytes"] for r in bandwidth_results], dtype=float)
    times = np.array([r["one_way_us"] * 1e-6 for r in bandwidth_results], dtype=float)

    try:
        poly = np.polyfit(sizes, times, 1)
        slope = poly[0]  # 1 / beta
        intercept = poly[1]  # alpha

        if slope > 0:
            beta = 1.0 / slope
            alpha_fit = max(1e-8, intercept)
            n_half = alpha_fit * beta
            return {
                "alpha_us": alpha_fit * 1e6,
                "beta_mb_s": beta / 1e6,
                "beta_mib_s": beta / (1024 * 1024),
                "beta_gbit_s": (beta * 8) / 1e9,
                "n_half_bytes": int(n_half),
                "n_half_fmt": format_bytes(int(n_half))
            }
    except Exception:
        pass

    return None


def _run_pair_benchmark(comm, rank1, rank2, mode_label, max_packet_kb):
    """
    Выполняет полный цикл замеров (латентность, пропускная способность, модель Хокни)
    для заданной пары процессов (rank1, rank2) и передает результаты на rank 0.
    """
    rank = comm.Get_rank()

    if rank == 0:
        print(f"\n-------------------------------------------------------")
        print(f"  Режим: {mode_label}")
        print(f"-------------------------------------------------------")

    comm.Barrier()

    if rank == 0:
        print("\n[*] Измерение задержки на малых пакетах (ping-pong, 1000 итераций)...")
    small_results = measure_small_packet_latency(comm, rank1, rank2, iterations=1000)

    comm.Barrier()

    if rank == 0:
        print(f"[*] Измерение скорости и задержки от размера пакета (от 1 Б до {max_packet_kb} КБ)...")
    bandwidth_results = measure_bandwidth_vs_size(comm, rank1, rank2, max_size_kb=max_packet_kb)

    comm.Barrier()

    # Если замер проводился не на rank 0 (например, пара на втором узле), пересылаем результаты на rank 0
    if rank1 != 0:
        if rank == rank1:
            comm.send((small_results, bandwidth_results), dest=0, tag=99)
        elif rank == 0:
            small_results, bandwidth_results = comm.recv(source=rank1, tag=99)

    pair_data = None
    if rank == 0:
        small_headers = ["Size", "Min Lat (us)", "Throughput (MB/s)"]
        small_rows = [
            [
                format_bytes(r["size_bytes"]),
                f"{r['min_us']:.2f}",
                f"{r['mb_per_sec']:.3f}",
            ]
            for r in small_results
        ]
        print("\n" + format_table(small_headers, small_rows, ['<', '>', '>']))

        bw_headers = ["Packet Size", "Min One-way (us)", "Speed (MB/s)", "Speed (Mbit/s)", "Iters"]
        bw_rows = [
            [
                r["size_fmt"],
                f"{r['one_way_us']:.2f}",
                f"{r['mb_per_sec']:.2f}",
                f"{r['mbit_per_sec']:.2f}",
                str(r["iterations"]),
            ]
            for r in bandwidth_results
        ]
        print("\n" + format_table(bw_headers, bw_rows, ['<', '>', '>', '>', '>']))

        hockney = fit_hockney_model(bandwidth_results)
        if hockney:
            print("\n[+] Аналитическая модель сети Хокни ( T(s) = alpha + s / beta ):")
            print(f"    - Стартовая задержка (alpha):           {hockney['alpha_us']:.2f} us")
            print(
                f"    - Асимптотическая скорость (beta):      {hockney['beta_mb_s']:.2f} MB/s ({hockney['beta_gbit_s']:.2f} Gbit/s)")
            print(
                f"    - Размер половинной мощности (n_1/2):   {hockney['n_half_fmt']} ({hockney['n_half_bytes']} bytes)")

        pair_data = {
            "mode": mode_label,
            "rank1": rank1,
            "rank2": rank2,
            "small_packet_latency": small_results,
            "bandwidth_vs_size": bandwidth_results,
            "hockney_model": hockney
        }

    return pair_data


def _node_sort_key(hostname):
    """Сортирует узлы кластера в логическом порядке: mpi, opi1..opi6, rpi1..rpi4."""
    if hostname == "mpi":
        return (0, 0, hostname)
    elif hostname.startswith("opi"):
        num = int(hostname[3:]) if hostname[3:].isdigit() else 99
        return (1, num, hostname)
    elif hostname.startswith("rpi"):
        num = int(hostname[3:]) if hostname[3:].isdigit() else 99
        return (2, num, hostname)
    return (3, 0, hostname)


def measure_node_matrices(comm, host_to_ranks, bw_packet_kb=100, lat_iters=200, bw_iters=50):
    """
    Строит две симметричные матрицы N x N по всем узлам кластера (по минимальному времени):
    1) Матрица пропускной способности (MB/s) на пакетах bw_packet_kb КБ (по умолчанию 100 КБ).
    2) Матрица минимальной односторонней задержки (мкс) на малых пакетах (1 Б).
    На главной диагонали (i == j) измеряется скорость и латенси ВНУТРИ одного компьютера
    (если на узле запущено >= 2 процессов).
    """
    rank = comm.Get_rank()
    nodes = sorted(host_to_ranks.keys(), key=_node_sort_key)
    n_nodes = len(nodes)

    bw_size = int(bw_packet_kb * 1024)
    lat_size = 1

    local_bw = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    local_lat = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for i in range(n_nodes):
        for j in range(i, n_nodes):
            h_i, h_j = nodes[i], nodes[j]

            if i == j:
                if len(host_to_ranks[h_i]) < 2:
                    continue
                r1, r2 = host_to_ranks[h_i][0], host_to_ranks[h_i][1]
            else:
                r1, r2 = host_to_ranks[h_i][0], host_to_ranks[h_j][0]

            # 1. Минимальная односторонняя латентность (мкс)
            if rank == r1:
                rtts_lat = _ping_pong_master(comm, r2, lat_size, iterations=lat_iters, warmup=20)
                min_lat_us = (float(np.min(rtts_lat)) / 2.0) * 1e6
                local_lat[i, j] = min_lat_us
                local_lat[j, i] = min_lat_us
            elif rank == r2:
                _ping_pong_worker(comm, r1, lat_size, iterations=lat_iters, warmup=20)

            comm.Barrier()

            # 2. Максимальная скорость передачи по минимальному времени (MB/s)
            if rank == r1:
                rtts_bw = _ping_pong_master(comm, r2, bw_size, iterations=bw_iters, warmup=10)
                min_one_way_sec = float(np.min(rtts_bw)) / 2.0
                mb_s = (bw_size / min_one_way_sec) / 1e6 if min_one_way_sec > 0 else 0.0
                local_bw[i, j] = mb_s
                local_bw[j, i] = mb_s
            elif rank == r2:
                _ping_pong_worker(comm, r1, bw_size, iterations=bw_iters, warmup=10)

            comm.Barrier()

    full_bw = np.zeros_like(local_bw)
    full_lat = np.zeros_like(local_lat)
    comm.Reduce([local_bw, MPI.DOUBLE], [full_bw, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([local_lat, MPI.DOUBLE], [full_lat, MPI.DOUBLE], op=MPI.SUM, root=0)

    if rank == 0:
        return {
            "nodes": nodes,
            "bandwidth_matrix_mb_s": full_bw,
            "latency_matrix_us": full_lat,
            "bw_packet_kb": bw_packet_kb
        }
    return None


def run_network_benchmarks(comm, max_packet_kb=128):
    """
    Оркестрирует сетевой бенчмарк:
    1) Пересылка на одном компьютере (Intra-node / Shared Memory), если на узле >= 2 процессов.
    2) Пересылка между разными компьютерами (Inter-node / Network), если задействовано >= 2 узлов.
    3) Полные матрицы N x N по всем узлам кластера (скорость на 100 КБ и латенси на 1 Б).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    processor_name = MPI.Get_processor_name()

    if size < 2:
        if rank == 0:
            print("\n[!] ВНИМАНИЕ: Для тестирования необходимо хотя бы 2 MPI процесса.")
        return None

    all_hosts = comm.allgather((rank, processor_name))
    host_to_ranks = {}
    for r, h in all_hosts:
        host_to_ranks.setdefault(h, []).append(r)

    unique_hosts = sorted(host_to_ranks.keys(), key=_node_sort_key)

    # Поиск пары на одном компьютере (Intra-node)
    intra_pair = None
    for h in unique_hosts:
        if len(host_to_ranks[h]) >= 2:
            intra_pair = (host_to_ranks[h][0], host_to_ranks[h][1], h)
            break

    # Поиск пары на разных компьютерах (Inter-node)
    inter_pair = None
    if len(unique_hosts) > 1:
        h1, h2 = unique_hosts[0], unique_hosts[1]
        inter_pair = (host_to_ranks[h1][0], host_to_ranks[h2][0], h1, h2)

    if rank == 0:
        print(f"\n=======================================================")
        print(f"  ПУНКТ 1: Оценка сетевых характеристик (comm.Send/Recv)")
        print(f"  Узлов: {len(unique_hosts)} | Процессов: {size} | Макс. пакет: {max_packet_kb} КБ")
        print(f"=======================================================")

    intra_data = None
    inter_data = None

    # 1. Замер на одном компьютере (Intra-node)
    if intra_pair is not None:
        r1, r2, host = intra_pair
        label = f"Intra-node / На одном компьютере ({host}: Rank {r1} <---> Rank {r2})"
        intra_data = _run_pair_benchmark(comm, r1, r2, label, max_packet_kb)

    # 2. Замер между разными компьютерами (Inter-node)
    if inter_pair is not None:
        r1, r2, h1, h2 = inter_pair
        label = f"Inter-node / Между узлами по сети ({h1}:{r1} <---> {h2}:{r2})"
        inter_data = _run_pair_benchmark(comm, r1, r2, label, max_packet_kb)

    # 3. Полные матрицы N x N по всем узлам
    bw_mat_kb = min(100, max_packet_kb)
    if rank == 0:
        print(f"\n-------------------------------------------------------")
        print(f"  Построение матриц {len(unique_hosts)}x{len(unique_hosts)} между узлами (пакет {bw_mat_kb} КБ)")
        print(f"-------------------------------------------------------")

    matrix_data = measure_node_matrices(comm, host_to_ranks, bw_packet_kb=bw_mat_kb, lat_iters=200, bw_iters=50)

    if rank == 0 and matrix_data is not None:
        nodes = matrix_data["nodes"]
        n = len(nodes)
        bw_mat = matrix_data["bandwidth_matrix_mb_s"]
        lat_mat = matrix_data["latency_matrix_us"]

        headers = ["Узел"] + nodes
        bw_rows = [
            [nodes[i]] + [f"{bw_mat[i, j]:.2f}" if bw_mat[i, j] > 0 else "-" for j in range(n)]
            for i in range(n)
        ]
        lat_rows = [
            [nodes[i]] + [f"{lat_mat[i, j]:.1f}" if lat_mat[i, j] > 0 else "-" for j in range(n)]
            for i in range(n)
        ]

        print(f"\n--- Матрица пропускной способности (MB/s, пакет {bw_mat_kb} КБ) ---")
        print(format_table(headers, bw_rows, ['<'] + ['>'] * n))

        print("\n--- Матрица минимальной односторонней задержки (мкс, пакет 1 Б) ---")
        print(format_table(headers, lat_rows, ['<'] + ['>'] * n))

        primary = inter_data if inter_data is not None else intra_data
        output_data = dict(primary) if primary is not None else {}
        output_data["intra_node"] = intra_data
        output_data["inter_node"] = inter_data
        output_data["node_matrices"] = matrix_data
        return output_data

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    net_res = run_network_benchmarks(comm, max_packet_kb=128)
    if comm.Get_rank() == 0 and net_res is not None:
        from utils.plot_results import export_results_to_csv, generate_plots
        export_results_to_csv("benchmark_results", net_res, None, None, None)
        generate_plots("benchmark_results")