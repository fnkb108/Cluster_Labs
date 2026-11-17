#!/usr/bin/env python3

import sys
import os
import argparse

# Добавляем текущую директорию скрипта в sys.path для корректного импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpi4py import MPI
from modules.network_p2p import run_network_benchmarks
from modules.node_hardware import run_node_hardware_benchmarks
from modules.collectives import run_collective_benchmarks
from utils.plot_results import export_results_to_csv, generate_plots


def main():
    # Инициализация глобального коммуникатора MPI, получение ранга процесса и общего числа процессов
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Настройка парсера аргументов командной строки (CLI)
    parser = argparse.ArgumentParser(
        description="Комплексный инструмент анализа и бенчмаркинга MPI-кластера (Лабораторная работа №1)"
    )
    parser.add_argument("--all", action="store_true", help="Запустить все модули бенчмарка (по умолчанию: True)")
    parser.add_argument("--network", action="store_true", help="Запустить только Модуль 1: Сеть P2P (Send/Recv)")
    parser.add_argument("--hardware", action="store_true",
                        help="Запустить только Модуль 2: Оборудование узлов и GFlop/s")
    parser.add_argument("--collectives", action="store_true",
                        help="Запустить только Модуль 3: Коллективные операции против P2P")
    parser.add_argument("--max-packet-kb", type=int, default=128,
                        help="Максимальный размер пакета в КБ (по умолчанию: 128)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Директория для сохранения CSV и графиков")
    parser.add_argument("--quick", action="store_true", help="Быстрый режим (ограничение пакетов до 16 КБ)")

    # Разбор аргументов командной строки на всех процессах
    args = parser.parse_args()

    # Определяем, какие именно модули необходимо выполнить
    run_all = args.all or not (args.network or args.hardware or args.collectives)
    do_net = run_all or args.network
    do_hw = run_all or args.hardware
    do_coll = run_all or args.collectives

    # В быстром режиме ограничиваем максимальный пакет 16 КБ вместо 128 КБ
    max_kb = 16 if args.quick else args.max_packet_kb

    # Корневой процесс (rank 0) выводит заголовок и общую конфигурацию запуска
    if rank == 0:
        print("=" * 70)
        print("   HPC COMPUTING CLUSTER BENCHMARK & ANALYSIS SUITE (LAB 1)")
        print(f"   MPI Processes (Ranks): {size}")
        print(f"   MPI Vendor/Version:    {MPI.get_vendor()}")
        print(f"   Max Packet Size:       {max_kb} KB")
        print(f"   Output Directory:      {args.output_dir}")
        print("=" * 70)

    # Синхронизация всех процессов перед стартом общего таймера
    comm.Barrier()
    t_suite_start = MPI.Wtime()

    net_data = None
    hw_data = None
    coll_data = None

    # Модуль 1: Тестирование сети точка-точка (латентность, пропускная способность, модель Хокни)
    if do_net:
        net_data = run_network_benchmarks(comm, max_packet_kb=max_kb)
        comm.Barrier()

    # Модуль 2: Сбор аппаратных характеристик узлов и замер производительности (GEMM и FMA GFlop/s)
    if do_hw:
        hw_data = run_node_hardware_benchmarks(comm)
        comm.Barrier()

    # Модуль 3: Сравнение штатных коллективных операций (Bcast, Reduce) с ручными P2P-алгоритмами
    if do_coll:
            coll_sizes = [1, 2, 4, 8, 10, 16] if args.quick else [1, 2, 4, 8, 10, 16, 32, 64, 100, 128]
            coll_data = run_collective_benchmarks(comm, sizes_kb=coll_sizes)
            comm.Barrier()

    # Фиксация времени завершения всего комплекса тестов
    t_suite_end = MPI.Wtime()

    # Экспорт собранных результатов в CSV и завершение работы (выполняется только на rank 0)
    if rank == 0:
        total_time = t_suite_end - t_suite_start
        print(f"\n=======================================================")
        print(f"  BENCHMARK SUITE COMPLETED IN {total_time:.2f} SECONDS")
        print(f"=======================================================")

        # Сохранение всех метрик в CSV-таблицы
        export_results_to_csv(args.output_dir, net_data, hw_data, coll_data, None)

        # Опциональная генерация графиков (отключена по умолчанию)
        # generate_plots(args.output_dir)

        print("\n[+] All tasks finished successfully.")
        print(f"[+] Output tables and raw CSV metrics saved to: '{args.output_dir}/'")


if __name__ == "__main__":
    main()