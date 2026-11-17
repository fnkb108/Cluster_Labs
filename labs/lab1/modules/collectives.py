"""
Модуль 3: Сравнение коллективных операций и коммуникаций «точка-точка» (Point-to-Point)
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from mpi4py import MPI
from utils.formatting import format_table, format_bytes


def custom_bcast_flat(buf, root, comm):
    """
    Наивная линейная широковещательная рассылка с использованием comm.Send и comm.Recv.
    Корневой процесс (root) последовательно отправляет буфер всем рангам от 0 до P-1.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        for dest in range(size):
            if dest != root:
                comm.Send([buf, MPI.BYTE], dest=dest, tag=101)
    else:
        comm.Recv([buf, MPI.BYTE], source=root, tag=101)


def custom_bcast_binomial_tree(buf, root, comm):
    """
    Широковещательная рассылка по биномиальному дереву с использованием comm.Send и comm.Recv.
    На шаге k (k = 0 .. ceil(log2(P))-1):
      Активные ранги отправляют данные процессам (rank + 2^k).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Сдвигаем ранги так, чтобы корневой процесс (root) стал виртуальным рангом 0
    vrank = (rank - root + size) % size
    mask = 1

    while mask < size:
        if vrank < mask:
            target_vrank = vrank + mask
            if target_vrank < size:
                target_actual = (target_vrank + root) % size
                comm.Send([buf, MPI.BYTE], dest=target_actual, tag=102)
        elif vrank < 2 * mask:
            source_vrank = vrank - mask
            source_actual = (source_vrank + root) % size
            comm.Recv([buf, MPI.BYTE], source=source_actual, tag=102)
        mask <<= 1


def custom_reduce_flat(sendbuf, recvbuf, root, comm):
    """
    Наивная линейная редукция: все ранги отправляют свои массивы корневому процессу; root суммирует их.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        # Инициализируем приемный буфер локальным вкладом корневого процесса
        recvbuf[:] = sendbuf[:]
        temp = np.empty_like(sendbuf)
        for src in range(size):
            if src != root:
                comm.Recv([temp, MPI.DOUBLE], source=src, tag=201)
                recvbuf += temp
    else:
        comm.Send([sendbuf, MPI.DOUBLE], dest=root, tag=201)


def custom_reduce_tree(sendbuf, recvbuf, root, comm):
    """
    Редукция по бинарному дереву с использованием comm.Send и comm.Recv.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Сдвигаем ранги так, чтобы корневой процесс (root) стал виртуальным рангом 0
    vrank = (rank - root + size) % size
    acc = sendbuf.copy()
    temp = np.empty_like(sendbuf)

    mask = 1
    while mask < size:
        if (vrank & mask) == 0:
            src_vrank = vrank | mask
            if src_vrank < size:
                src_actual = (src_vrank + root) % size
                comm.Recv([temp, MPI.DOUBLE], source=src_actual, tag=202)
                acc += temp
        else:
            dest_vrank = vrank & (~mask)
            dest_actual = (dest_vrank + root) % size
            comm.Send([acc, MPI.DOUBLE], dest=dest_actual, tag=202)
            break
        mask <<= 1

    if vrank == 0:
        recvbuf[:] = acc[:]


def _benchmark_op(comm, func, iters=30, warmup=3):
    """Измеряет максимальное время выполнения заданной коллективной функции среди всех рангов."""
    for _ in range(warmup):
        func()

    iter_times = []
    for _ in range(iters):
        comm.Barrier()
        t0 = MPI.Wtime()
        func()
        comm.Barrier()
        t1 = MPI.Wtime()
        iter_times.append(t1 - t0)

    min_elapsed = min(iter_times)
    global_min_elapsed = comm.reduce(min_elapsed, op=MPI.MAX, root=0)
    return global_min_elapsed

def run_collective_benchmarks(comm, sizes_kb=None):
    """
    Сравнивает встроенные коллективные операции и P2P-аналоги на различных размерах данных.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("\n[!] ВНИМАНИЕ: Для тестирования необходимо хотя бы 2 MPI процесса.")
        return None

    if sizes_kb is None:
        sizes_kb = [1, 2, 4, 8, 10, 16, 32, 64, 100, 128]

    if rank == 0:
        print(f"\n=======================================================")
        print(f"  ПУНКТ 3: Сравнение коллективных операций и P2P")
        print(f"  Total Ranks: {size}")
        print(f"=======================================================")

    # Часть А: Широковещательная рассылка (MPI.Bcast против Flat и Tree)
    if rank == 0:
        print("\n--- Broadcast (Bcast) Benchmark ---")

    bcast_results = []
    for sz_kb in sizes_kb:
        n_bytes = sz_kb * 1024
        buf = np.ones(n_bytes, dtype=np.uint8) if rank == 0 else np.zeros(n_bytes, dtype=np.uint8)
        iters = 50

        t_builtin = _benchmark_op(comm, lambda: comm.Bcast([buf, MPI.BYTE], root=0), iters=iters)
        t_flat = _benchmark_op(comm, lambda: custom_bcast_flat(buf, root=0, comm=comm), iters=iters)
        t_tree = _benchmark_op(comm, lambda: custom_bcast_binomial_tree(buf, root=0, comm=comm), iters=iters)

        if rank == 0:
            speedup_flat = (t_flat / t_builtin) if t_builtin > 0 else 1.0
            speedup_tree = (t_tree / t_builtin) if t_builtin > 0 else 1.0
            bw_builtin = (n_bytes / t_builtin) / 1e6 if t_builtin > 0 else 0.0

            bcast_results.append({
                "size_kb": sz_kb,
                "size_fmt": format_bytes(n_bytes),
                "t_builtin_ms": t_builtin * 1000,
                "t_flat_ms": t_flat * 1000,
                "t_tree_ms": t_tree * 1000,
                "speedup_vs_flat": speedup_flat,
                "speedup_vs_tree": speedup_tree,
                "bw_builtin_mb_s": bw_builtin
            })

    if rank == 0:
        headers = [
            "Data Size",
            "comm.Bcast (ms)",
            "P2P Flat (ms)",
            "P2P Tree (ms)",
            "Flat / Bcast",
            "Tree / Bcast",
            "Bcast Speed"
        ]
        rows = [
            [
                r["size_fmt"],
                f"{r['t_builtin_ms']:.4f}",
                f"{r['t_flat_ms']:.4f}",
                f"{r['t_tree_ms']:.4f}",
                f"{r['speedup_vs_flat']:.2f}x",
                f"{r['speedup_vs_tree']:.2f}x",
                f"{r['bw_builtin_mb_s']:.1f} MB/s"
            ]
            for r in bcast_results
        ]
        print(format_table(headers, rows, ['<', '>', '>', '>', '>', '>', '>']))

    # Часть Б: Reduce
    if rank == 0:
        print("\n--- Reduce (MPI_SUM) Benchmark ---")

    reduce_results = []
    for sz_kb in sizes_kb:
        n_elements = (sz_kb * 1024) // 8
        sendbuf = np.ones(n_elements, dtype=np.float64)
        recvbuf = np.zeros(n_elements, dtype=np.float64)
        n_bytes = n_elements * 8
        iters = 50

        t_builtin = _benchmark_op(
            comm,
            lambda: comm.Reduce([sendbuf, MPI.DOUBLE], [recvbuf, MPI.DOUBLE], op=MPI.SUM, root=0),
            iters=iters
        )
        t_flat = _benchmark_op(
            comm,
            lambda: custom_reduce_flat(sendbuf, recvbuf, root=0, comm=comm),
            iters=iters
        )
        t_tree = _benchmark_op(
            comm,
            lambda: custom_reduce_tree(sendbuf, recvbuf, root=0, comm=comm),
            iters=iters
        )

        if rank == 0:
            speedup_flat = (t_flat / t_builtin) if t_builtin > 0 else 1.0
            speedup_tree = (t_tree / t_builtin) if t_builtin > 0 else 1.0
            bw_builtin = (n_bytes / t_builtin) / 1e6 if t_builtin > 0 else 0.0

            reduce_results.append({
                "size_kb": sz_kb,
                "size_fmt": format_bytes(n_bytes),
                "t_builtin_ms": t_builtin * 1000,
                "t_flat_ms": t_flat * 1000,
                "t_tree_ms": t_tree * 1000,
                "speedup_vs_flat": speedup_flat,
                "speedup_vs_tree": speedup_tree,
                "bw_builtin_mb_s": bw_builtin
            })

    if rank == 0:
        headers = [
            "Data Size",
            "comm.Reduce (ms)",
            "P2P Flat (ms)",
            "P2P Tree (ms)",
            "Flat / Reduce",
            "Tree / Reduce",
            "Reduce Speed"
        ]
        rows = [
            [
                r["size_fmt"],
                f"{r['t_builtin_ms']:.4f}",
                f"{r['t_flat_ms']:.4f}",
                f"{r['t_tree_ms']:.4f}",
                f"{r['speedup_vs_flat']:.2f}x",
                f"{r['speedup_vs_tree']:.2f}x",
                f"{r['bw_builtin_mb_s']:.1f} MB/s"
            ]
            for r in reduce_results
        ]
        print(format_table(headers, rows, ['<', '>', '>', '>', '>', '>', '>']))

    return {
        "bcast": bcast_results,
        "reduce": reduce_results
    } if rank == 0 else None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    run_collective_benchmarks(comm)