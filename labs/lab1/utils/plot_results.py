"""
Утилита для экспорта результатов бенчмарков в CSV-файлы и отрисовки графиков через matplotlib.
"""

import os
import csv

def _save_pair_csv(data_dir, pair_dict, prefix=""):
    """Сохраняет CSV по малым пакетам и кривой скорости для одной пары (inter-node или intra-node)."""
    if not pair_dict:
        return

    if pair_dict.get("bandwidth_vs_size"):
        path = os.path.join(data_dir, f"network_{prefix}bandwidth.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Size_Bytes", "Size_KB", "Size_Formatted", "Min_One_Way_Latency_us", "Bandwidth_MB_s", "Bandwidth_Mbit_s"])
            for r in pair_dict["bandwidth_vs_size"]:
                writer.writerow([
                    r["size_bytes"],
                    f"{r['size_bytes'] / 1024:.3f}",
                    r["size_fmt"],
                    f"{r['one_way_us']:.3f}",
                    f"{r['mb_per_sec']:.3f}",
                    f"{r['mbit_per_sec']:.3f}"
                ])
        print(f"[+] Saved network bandwidth data to {path}")

    if pair_dict.get("small_packet_latency"):
        path = os.path.join(data_dir, f"network_{prefix}small_packets.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Size_Bytes", "Min_Latency_us", "Throughput_MB_s"])
            for r in pair_dict["small_packet_latency"]:
                writer.writerow([
                    r["size_bytes"],
                    f"{r['min_us']:.3f}",
                    f"{r['mb_per_sec']:.3f}"
                ])
        print(f"[+] Saved small packet latency data to {path}")

def export_results_to_csv(data_dir, net_data, hw_data, coll_data, init_data):
    """Сохраняет все собранные метрики бенчмарков в CSV-файлы для построения графиков и отчетов."""
    os.makedirs(data_dir, exist_ok=True)

    # 1. Сетевые замеры (основной + отдельный intra-node на одном компьютере + матрицы N x N)
    if net_data:
        _save_pair_csv(data_dir, net_data, prefix="")

        if net_data.get("intra_node"):
            _save_pair_csv(data_dir, net_data["intra_node"], prefix="intra_")

        mat_data = net_data.get("node_matrices")
        if mat_data:
            nodes = mat_data["nodes"]
            n = len(nodes)
            bw_mat = mat_data["bandwidth_matrix_mb_s"]
            lat_mat = mat_data["latency_matrix_us"]

            bw_csv = os.path.join(data_dir, "matrix_node_bandwidth.csv")
            with open(bw_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Node"] + nodes)
                for i, h in enumerate(nodes):
                    w.writerow([h] + [f"{bw_mat[i, j]:.2f}" if bw_mat[i, j] > 0 else "-" for j in range(n)])
            print(f"[+] Saved node bandwidth matrix to {bw_csv}")

            lat_csv = os.path.join(data_dir, "matrix_node_latency.csv")
            with open(lat_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Node"] + nodes)
                for i, h in enumerate(nodes):
                    w.writerow([h] + [f"{lat_mat[i, j]:.1f}" if lat_mat[i, j] > 0 else "-" for j in range(n)])
            print(f"[+] Saved node latency matrix to {lat_csv}")

    # 2. Оборудование и производительность узлов
    if hw_data and "all_ranks_data" in hw_data:
        path = os.path.join(data_dir, "hardware_nodes.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["MPI_Rank", "Hostname", "CPU_Model", "Physical_Cores", "Logical_Cores", "RAM_Total_GB", "RAM_Avail_GB",
                 "GEMM_Peak_GFlops", "FMA_GFlops"])
            for r in hw_data["all_ranks_data"]:
                writer.writerow([
                    r.get("mpi_rank", ""),
                    r.get("hostname", ""),
                    r.get("cpu_model", ""),
                    r.get("physical_cores", ""),
                    r.get("logical_cores", ""),
                    r.get("ram_total_gb", ""),
                    r.get("ram_avail_gb", ""),
                    f"{r.get('gemm_peak_gflops', 0):.2f}",
                    f"{r.get('fma_gflops', 0):.2f}"
                ])
        print(f"[+] Saved hardware & GFlop/s data to {path}")

    # 3. Коллективные операции
    if coll_data:
        for op_name, fname in [("bcast", "collectives_bcast.csv"), ("reduce", "collectives_reduce.csv")]:
            if coll_data.get(op_name):
                path = os.path.join(data_dir, fname)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Size_KB", "Size_Formatted", "Min_Time_Builtin_ms", "Min_Time_Flat_P2P_ms",
                                     "Min_Time_Tree_P2P_ms", "Speedup_vs_Flat", "Speedup_vs_Tree",
                                     "Builtin_Speed_MB_s"])
                    for r in coll_data[op_name]:
                        writer.writerow([
                            r["size_kb"], r["size_fmt"],
                            f"{r['t_builtin_ms']:.4f}", f"{r['t_flat_ms']:.4f}", f"{r['t_tree_ms']:.4f}",
                            f"{r['speedup_vs_flat']:.2f}", f"{r['speedup_vs_tree']:.2f}",
                            f"{r['bw_builtin_mb_s']:.2f}"
                        ])
                print(f"[+] Saved {op_name} comparison data to {path}")

def _load_matrix_csv(csv_path):
    """Загружает матрицу узлов из CSV, заменяя '-' и 0 на np.nan."""
    import numpy as np
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    nodes = rows[0][1:]
    n = len(nodes)
    mat = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            val_str = rows[i + 1][j + 1].strip()
            if val_str != "-" and val_str != "":
                val = float(val_str)
                if val > 0:
                    mat[i, j] = val
    return nodes, mat

def _plot_node_matrices_heatmap(data_dir, plt):
    """
    Строит двойную тепловую карту матриц N x N (как на образце):
    - Слева: Пропускная способность (MB/s), палитра YlGnBu
    - Справа: Односторонняя задержка (мкс), палитра YlOrRd
    """
    import numpy as np

    bw_csv = os.path.join(data_dir, "matrix_node_bandwidth.csv")
    lat_csv = os.path.join(data_dir, "matrix_node_latency.csv")
    if not (os.path.exists(bw_csv) and os.path.exists(lat_csv)):
        return

    nodes, bw_mat = _load_matrix_csv(bw_csv)
    _, lat_mat = _load_matrix_csv(lat_csv)
    n = len(nodes)
    if n == 0:
        return

    has_diagonal = not np.all(np.isnan(np.diag(bw_mat)))
    diag_subtitle = (
        "на главной диагонали — обмен внутри одного компьютера"
        if has_diagonal else
        "диагональ не измерялась"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=300, facecolor="#eaeaea")
    fig.suptitle(
        f"Матрицы межузловых измерений • {n} узлов\n"
        f"Симметричное отображение результатов ping-pong; {diag_subtitle}",
        fontsize=13, fontweight="bold", y=0.97
    )

    # 1. Левая матрица: Пропускная способность (MB/s)
    masked_bw = np.ma.masked_invalid(bw_mat)
    cmap_bw = plt.get_cmap("YlGnBu").copy()
    cmap_bw.set_bad(color="#e2e2e2")

    # Цветовую шкалу настраиваем по межузловым значениям, чтобы высокая скорость на диагонали не сглаживала цвета сети
    off_diag_bw = bw_mat[~np.eye(n, dtype=bool)]
    valid_off_bw = off_diag_bw[~np.isnan(off_diag_bw)]
    if len(valid_off_bw) > 0:
        vmin_bw = float(np.min(valid_off_bw)) * 0.95
        vmax_bw = float(np.max(valid_off_bw)) * 1.02
    else:
        valid_all_bw = bw_mat[~np.isnan(bw_mat)]
        vmin_bw = float(np.min(valid_all_bw)) * 0.95 if len(valid_all_bw) > 0 else 0.0
        vmax_bw = float(np.max(valid_all_bw)) * 1.02 if len(valid_all_bw) > 0 else 100.0

    im1 = ax1.imshow(masked_bw, cmap=cmap_bw, vmin=vmin_bw, vmax=vmax_bw)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(nodes, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(nodes, fontsize=8)
    ax1.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax1.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
    ax1.tick_params(which="minor", bottom=False, left=False)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.72, pad=0.03)
    cb1.outline.set_visible(False)
    cb1.ax.tick_params(labelsize=8)

    for i in range(n):
        for j in range(n):
            val = bw_mat[i, j]
            if np.isnan(val):
                ax1.text(j, i, "—", ha="center", va="center", color="#444444", fontsize=7.5)
            else:
                txt = f"{val:.1f}" if val >= 1000 else f"{val:.2f}"
                txt_color = "white" if val > (vmin_bw + (vmax_bw - vmin_bw) * 0.55) else "black"
                ax1.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=6.8)

    # 2. Правая матрица: Задержка (мкс)
    masked_lat = np.ma.masked_invalid(lat_mat)
    cmap_lat = plt.get_cmap("YlOrRd").copy()
    cmap_lat.set_bad(color="#e2e2e2")

    valid_lat = lat_mat[~np.isnan(lat_mat)]
    vmin_lat = float(np.min(valid_lat)) * 0.95 if len(valid_lat) > 0 else 0.0
    vmax_lat = float(np.max(valid_lat)) * 1.02 if len(valid_lat) > 0 else 100.0

    im2 = ax2.imshow(masked_lat, cmap=cmap_lat, vmin=vmin_lat, vmax=vmax_lat)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(nodes, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(nodes, fontsize=8)
    ax2.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax2.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
    ax2.tick_params(which="minor", bottom=False, left=False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.72, pad=0.03)
    cb2.outline.set_visible(False)
    cb2.ax.tick_params(labelsize=8)

    for i in range(n):
        for j in range(n):
            val = lat_mat[i, j]
            if np.isnan(val):
                ax2.text(j, i, "—", ha="center", va="center", color="#444444", fontsize=7.5)
            else:
                txt_color = "white" if val > (vmin_lat + (vmax_lat - vmin_lat) * 0.62) else "black"
                ax2.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=6.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(data_dir, "plot_node_matrices.png")
    plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Generated node matrices heatmap: {out_png}")

def generate_plots(data_dir):
    """
    Строит все графики из экспортированных CSV-файлов через matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[*] Note: 'matplotlib' is not installed. Skipping plot generation.")
        print("    You can run 'python3 utils/plot_results.py --dir <data_dir>' locally to generate PNG charts.")
        return

    # 1. Тепловая карта матриц N x N (скорость и латентность)
    _plot_node_matrices_heatmap(data_dir, plt)

    # 2. Кривые скорости и задержки до 10/100/128 КБ (Inter-node и Intra-node на одном компьютере)
    bw_path = os.path.join(data_dir, "network_bandwidth.csv")
    intra_bw_path = os.path.join(data_dir, "network_intra_bandwidth.csv")

    if os.path.exists(bw_path):
        def _read_bw(p):
            sz_kb, bws, lats = [], [], []
            with open(p, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sz_kb.append(float(row["Size_KB"]))
                    bws.append(float(row["Bandwidth_MB_s"]))
                    lats.append(float(row["Min_One_Way_Latency_us"]))
            return sz_kb, bws, lats

        sz1, bw1, lat1 = _read_bw(bw_path)
        has_intra = os.path.exists(intra_bw_path)
        if has_intra:
            sz2, bw2, lat2 = _read_bw(intra_bw_path)

        if sz1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            ax1.plot(sz1, bw1, marker="o", linewidth=2, color="#1f77b4", label="Inter-node (Сеть) / Основной")
            if has_intra and sz2:
                ax1.plot(sz2, bw2, marker="s", linewidth=2, color="#2ca02c", label="Intra-node (Один компьютер)")
            ax1.set_title("Скорость передачи vs Размер пакета (КБ)", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Размер пакета (КБ)", fontsize=10)
            ax1.set_ylabel("Скорость (MB/s)", fontsize=10)
            ax1.grid(True, ls="--", alpha=0.6)
            ax1.legend()

            ax2.plot(sz1, lat1, marker="o", linewidth=2, color="#d62728", label="Inter-node (Сеть) / Основной")
            if has_intra and sz2:
                ax2.plot(sz2, lat2, marker="s", linewidth=2, color="#ff7f0e", label="Intra-node (Один компьютер)")
            ax2.set_title("Минимальная задержка vs Размер пакета (КБ)", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Размер пакета (КБ)", fontsize=10)
            ax2.set_ylabel("Односторонняя задержка (мкс)", fontsize=10)
            ax2.grid(True, ls="--", alpha=0.6)
            ax2.legend()

            plt.tight_layout()
            out_img = os.path.join(data_dir, "plot_bandwidth_vs_size.png")
            plt.savefig(out_img, dpi=300)
            plt.close()
            print(f"[+] Generated chart: {out_img}")

    # 3. Сравнение коллективных операций (Broadcast)
    bcast_path = os.path.join(data_dir, "collectives_bcast.csv")
    if os.path.exists(bcast_path):
        sizes_kb, t_builtin, t_flat, t_tree = [], [], [], []
        with open(bcast_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sizes_kb.append(float(row["Size_KB"]))
                t_builtin.append(float(row["Min_Time_Builtin_ms"]))
                t_flat.append(float(row["Min_Time_Flat_P2P_ms"]))
                t_tree.append(float(row["Min_Time_Tree_P2P_ms"]))

        if sizes_kb:
            plt.figure(figsize=(9, 5))
            plt.plot(sizes_kb, t_builtin, marker="o", label="Встроенный (comm.Bcast)", linewidth=2, color="green")
            plt.plot(sizes_kb, t_tree, marker="s", label="Биномиальное дерево (P2P)", linewidth=2, color="blue")
            plt.plot(sizes_kb, t_flat, marker="^", label="Линейная рассылка (Flat P2P)", linewidth=2, color="red")
            plt.title("Минимальное время Broadcast: встроенный и P2P-алгоритмы", fontsize=12, fontweight="bold")
            plt.xlabel("Размер данных (КБ)", fontsize=10)
            plt.ylabel("Минимальное время (мс)", fontsize=10)
            plt.legend(fontsize=10)
            plt.grid(True, ls="--", alpha=0.6)
            plt.tight_layout()
            out_img = os.path.join(data_dir, "plot_bcast_comparison.png")
            plt.savefig(out_img, dpi=300)
            plt.close()
            print(f"[+] Generated chart: {out_img}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Построение графиков из CSV-файлов бенчмарка")
    parser.add_argument("--dir", default="benchmark_results", help="Директория с CSV-файлами")
    args = parser.parse_args()
    generate_plots(args.dir)