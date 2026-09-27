"""Audit and report the frozen single-seed ResNet-18 non-IID study."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedmedai.non_iid_study import ALPHAS, ARMS, compare_cells, load_cell, read_lock

LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedprox_fp16": "FedProx + FP16"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_metrics(cell):
    """Independently recompute accuracy and macro-F1 from the saved confusion matrix."""
    result = cell["result"]
    report = result["test_evaluation"]
    cm = report["confusion_matrix"]
    if len(cm) != 8 or any(len(row) != 8 for row in cm):
        raise ValueError("Expected eight BloodMNIST classes")
    if any(not isinstance(x, int) or x < 0 for row in cm for x in row):
        raise ValueError("Invalid confusion matrix counts")
    total = sum(map(sum, cm))
    accuracy = sum(cm[i][i] for i in range(8)) / total
    f1 = sum(2 * cm[i][i] / denominator if denominator else 0
             for i in range(8)
             for denominator in [sum(cm[i]) + sum(row[i] for row in cm)]) / 8
    for key, expected in (("accuracy", accuracy), ("f1_macro", f1)):
        if not math.isclose(report[key], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Confusion matrix disagrees with test {key}")
        if not math.isclose(result[f"final_test_{key}"], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Final result disagrees with test {key}")


def validate_evidence(suite, index, cells, lock):
    """Recompute replay and verify snapshot linkage; do not trust a success label alone."""
    expected_names = {"baseline.lock.json", "protocol.md", "runner_snapshot.py"}
    hashes = read(suite / "snapshot_hashes.json")
    if set(hashes) != expected_names:
        raise ValueError("Incomplete snapshot hash manifest")
    for name, expected in hashes.items():
        if hashlib.sha256((suite / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot changed: {name}")
    execution = read(suite / "execution.json")
    replay = read(suite / "baseline_replay.json")
    rounds = lock["baseline_config"]["federation"]["num_rounds"]
    metric_keys = {"final_test_accuracy", "final_test_f1_macro", "final_test_loss",
                   "test_evaluation", "best_round", "total_model_payload_bytes", "partition_sha256"}
    required = metric_keys | {"checkpoint_tensors"} | {f"round_{r}_metrics" for r in range(1, rounds + 1)}
    if (index["status"] != "completed" or execution["status"] != "completed"
            or execution["exit_code"] != 0 or execution["supervisor_exit_code"] != 0
            or replay["exact_match"] is not True or set(replay["checks"]) != required
            or any(value is not True for value in replay["checks"].values())):
        raise ValueError("Incomplete execution or replay verification")
    baseline = next(c for c in cells if c["arm"] == "fedavg" and c["result"]["alpha"] == 0.3)
    if (Path(replay["reference"]).resolve() != Path(baseline["result"]["results_dir"]).resolve()
            or Path(replay["new_baseline"]).resolve() != Path(index["replay"]).resolve()):
        raise ValueError("Replay is not linked to the declared matrix baseline")
    if Path(index["replay"]).resolve() in {Path(path).resolve() for path in index["runs"]}:
        raise ValueError("Replay must be a separate run outside the matrix")
    repeated = load_cell(index["replay"], lock)
    if repeated["result"]["run_id"] in {cell["result"]["run_id"] for cell in cells}:
        raise ValueError("Replay must have a distinct run ID")
    for key in ("config", "source_sha256"):
        if baseline["manifest"][key] != repeated["manifest"][key]:
            raise ValueError(f"Replay changed {key}")
    if baseline["test_ids"] != repeated["test_ids"]:
        raise ValueError("Replay changed test IDs")
    for key in metric_keys:
        if baseline["result"][key] != repeated["result"][key]:
            raise ValueError(f"Replay changed {key}")
    import torch
    states = [torch.load(Path(c["manifest"]["artifact_paths"]["checkpoints"]) / "best_model.pt",
                         map_location="cpu", weights_only=True)["model_state_dict"]
              for c in (baseline, repeated)]
    if set(states[0]) != set(states[1]) or any(not torch.equal(states[0][k], states[1][k]) for k in states[0]):
        raise ValueError("Replay checkpoint tensors differ")
    for number in range(1, rounds + 1):
        records = [read(Path(c["result"]["results_dir"]) / f"rounds/round_{number:04d}.json")
                   for c in (baseline, repeated)]
        fields = [key for key in records[0] if key.startswith("global_val_") or key.endswith("model_bytes")]
        if any(records[0][key] != records[1][key] for key in fields):
            raise ValueError("Replay round metrics differ")
    for cell in cells + [repeated]:
        check_metrics(cell)
    return execution


def direct_codec_comparisons(cells):
    indexed = {(c["result"]["alpha"], c["arm"]): c["result"] for c in cells}
    rows = []
    for alpha in ALPHAS:
        a, b = indexed[alpha, "fedprox"], indexed[alpha, "fedprox_fp16"]
        rows.append({"alpha": alpha,
                     "accuracy_delta_pp": 100 * (b["final_test_accuracy"] - a["final_test_accuracy"]),
                     "f1_delta_pp": 100 * (b["final_test_f1_macro"] - a["final_test_f1_macro"]),
                     "total_saving_percent": 100 * (1 - b["total_model_payload_bytes"] / a["total_model_payload_bytes"])})
    return rows


def plot_results(cells, analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"fedavg": "#455A64", "fedprox": "#2166AC", "fedprox_fp16": "#B35806"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for arm in ARMS:
        values = [next(c["result"] for c in cells if c["arm"] == arm and c["result"]["alpha"] == alpha)
                  for alpha in ALPHAS]
        for ax, key in zip(axes[0], ("final_test_accuracy", "final_test_f1_macro")):
            ax.plot(range(3), [100 * r[key] for r in values], "o-", color=colors[arm], label=LABELS[arm])
        offset = (list(ARMS).index(arm) - 1) * 0.24
        axes[1, 1].bar([x + offset for x in range(3)],
                       [r["total_model_payload_bytes"] / 2**30 for r in values],
                       width=0.23, color=colors[arm], label=LABELS[arm])
    for arm in ("fedprox", "fedprox_fp16"):
        values = [r for r in analysis["comparisons"] if r["arm"] == arm]
        axes[1, 0].plot(range(3), [r["f1_delta_pp"] for r in values], "o-", color=colors[arm], label=LABELS[arm])
    axes[1, 0].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].axhline(1, color="gray", linestyle=":", linewidth=0.8, label="+1 pp practical threshold")
    for ax, title, ylabel in zip(axes.flat,
            ("Test accuracy", "Test macro-F1", "Macro-F1 change from fixed FedAvg", "Serialized model communication"),
            ("Accuracy (%)", "Macro-F1 (%)", "Difference (percentage points)", "Total payload (GiB)")):
        ax.set(title=title, ylabel=ylabel, xlabel="Dirichlet alpha")
        ax.set_xticks(range(3), [str(a) for a in ALPHAS])
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("BloodMNIST / ResNet-18 · 8 rounds · one seed (2026)\nConditions are not seed replicates; no uncertainty bars")
    fig.savefig(output / "non_iid_results.png", dpi=180)
    fig.savefig(output / "non_iid_results.svg")
    plt.close(fig)


def generate(suite):
    index = read(suite / "index.json")
    if index["status"] != "completed":
        raise ValueError("Cannot report an incomplete matrix")
    lock = read_lock(suite / "baseline.lock.json")
    cells = [load_cell(path, lock) for path in index["runs"]]
    analysis = compare_cells(cells, lock)
    if read(suite / "analysis.json") != analysis:
        raise ValueError("Recorded analysis differs from recomputed results")
    execution = validate_evidence(suite, index, cells, lock)
    direct = direct_codec_comparisons(cells)
    output = ROOT / "research"
    fields = ["alpha", "arm", "run_id", "model", "dataset", "partition_type", "completed_rounds",
              "train_samples", "test_samples", "algorithm", "uplink_codec", "random_seed", "final_test_accuracy",
              "final_test_f1_macro", "final_test_loss", "best_round", "total_fit_upload_model_bytes",
              "total_fit_download_model_bytes", "total_evaluate_download_model_bytes",
              "total_model_payload_bytes", "partition_sha256", "results_dir"]
    with (output / "non_iid_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**c["result"], "arm": c["arm"]} for c in cells)
    prox_gains = [str(r["alpha"]) for r in analysis["comparisons"]
                  if r["arm"] == "fedprox" and r["non_iid_practical_gain"]]
    prox_changes = "; ".join(f"α={r['alpha']}: {r['f1_delta_pp']:+.3f} điểm"
                             for r in analysis["comparisons"] if r["arm"] == "fedprox")
    passed = analysis["fedprox_fp16_communication_pass_all_alphas"]
    lines = ["## Material Passport", "", "- Origin Skill: ARS-Codex experiment-agent",
             "- Origin Mode: run / validate", f"- Origin Date: {datetime.now(timezone.utc).date()}",
             "- Verification Status: ANALYZED; exact FedAvg replay VERIFIED at α=0.3, seed=2026",
             "- Version Label: non_iid_report_v1", "", "## Kết quả", "",
             f"FedProx đạt ngưỡng cải thiện non-IID đã định trước tại α: **{', '.join(prox_gains) or 'không mức nào'}**.",
             "Ngưỡng: macro-F1 tăng ≥1 điểm phần trăm và accuracy không giảm so với FedAvg cùng α.",
             f"Δmacro-F1 của FedProx theo từng điều kiện: **{prox_changes}**.",
             "Kết quả không chứng minh FedProx cải thiện nhất quán ở cả ba mức non-IID.",
             f"FedProx+FP16 **{'ĐẠT' if passed else 'KHÔNG ĐẠT'}** yêu cầu đạt đồng thời tiêu chí truyền thông ở cả ba α:",
             "giảm ≥15% tổng payload, accuracy và macro-F1 giảm không quá 1 điểm phần trăm so với FedAvg.",
             "Đây là tiêu chí kỹ thuật mô tả với một seed; không phải kiểm định ý nghĩa thống kê hay chứng minh noninferiority.", "",
             "## Thiết kế đã khóa", "",
             "FedAvg không nén là baseline cố định. Hai phương pháp so sánh là FedProx μ=0.01",
             "và FedProx μ=0.01 kết hợp FP16 uplink. FedProx dùng proximal local objective thực sự;",
             "server vẫn tổng hợp trọng số theo số mẫu như định nghĩa phương pháp. FP16 chỉ nén",
             "uploads; training, aggregation và downloads giữ model dtype. Không chạy IID.", "",
             "ResNet-18 không pretrained, BloodMNIST 28×28 đầy đủ (11,959 train / 1,712 val / 3,421 test),",
             "3 clients tham gia mỗi round, 8 rounds, 1 local epoch, batch 32, AdamW lr=0.0001,",
             "weight decay=0.01, reset optimizer mỗi fit; deterministic sequential CPU với 8 threads.",
             "Seed 2026 chung cho mọi nhánh/α; chọn theo tính khả thi batch trước outcomes (seed 0/42",
             "có batch cuối một ảnh gây lỗi BatchNorm). Không bỏ ảnh, không sweep μ hoặc chọn seed theo test.",
             "Chọn checkpoint bằng pooled validation macro-F1, ưu tiên round sớm nhất khi bằng nhau;",
             "test chỉ đánh giá cuối run. Cả 9 ô hoàn tất; replay thứ 10 không được tính là lần lặp thống kê.", "",
             f"Suite: `{suite.resolve()}`.",
             f"Execution: `{execution['started_at_utc']}` → `{execution['finished_at_utc']}`, exit code 0.",
             "Chi tiết: [protocol](non_iid_protocol.md), [khóa baseline](non_iid_baseline.lock.json),",
             "[dữ liệu từng run](non_iid_runs.csv). Không dùng nghiên cứu BloodCNN cũ làm bằng chứng cho ResNet-18.", "",
             "## Số đo từng điều kiện", "",
             "Accuracy/macro-F1 là phần trăm; mỗi ô có một lần chạy, không có mean±SD hoặc CI qua seed.", "",
             "| α | Phương pháp | Accuracy | Macro-F1 | Test loss | Best round | Upload MiB | Tổng MiB | Worst-client val accuracy |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        r = c["result"]
        lines.append(f"| {r['alpha']} | {LABELS[c['arm']]} | {100*r['final_test_accuracy']:.3f} | "
                     f"{100*r['final_test_f1_macro']:.3f} | {r['final_test_loss']:.4f} | {r['best_round']} | "
                     f"{r['total_fit_upload_model_bytes']/2**20:.3f} | {r['total_model_payload_bytes']/2**20:.3f} | "
                     f"{100*r['last_round_fairness']['worst_client_accuracy']:.3f} |")
    lines += ["", "Worst-client accuracy dùng validation ở round cuối, không phải test theo client hoặc checkpoint đã chọn.",
              "Per-class metrics, confusion matrices, sample IDs và validation curves nằm trong thư mục từng run.", "",
              "### So với FedAvg cố định", "",
              "| α | Phương pháp | Δaccuracy (pp) | Δmacro-F1 (pp) | Giảm upload | Giảm tổng payload | Ngưỡng non-IID | Ngưỡng truyền thông |",
              "|---:|---|---:|---:|---:|---:|---|---|"]
    for r in analysis["comparisons"]:
        lines.append(f"| {r['alpha']} | {LABELS[r['arm']]} | {r['accuracy_delta_pp']:+.3f} | {r['f1_delta_pp']:+.3f} | "
                     f"{r['upload_saving_percent']:.3f}% | {r['total_saving_percent']:.3f}% | "
                     f"{'đạt' if r['non_iid_practical_gain'] else 'không đạt'} | "
                     f"{'đạt' if r['communication_quality_pass'] else 'không đạt'} |")
    lines += ["", "### Ảnh hưởng riêng của FP16 trong FedProx", "",
              "So sánh trực tiếp FedProx+FP16 trừ FedProx không nén; phần tiết kiệm byte thuộc codec FP16,",
              "không được quy cho proximal term. Thay đổi chất lượng giữa hai nhánh thể hiện ảnh hưởng codec",
              "trong quá trình training và lựa chọn checkpoint của lần chạy này.", "",
              "| α | Δaccuracy (pp) | Δmacro-F1 (pp) | Giảm tổng payload |", "|---:|---:|---:|---:|"]
    lines += [f"| {r['alpha']} | {r['accuracy_delta_pp']:+.3f} | {r['f1_delta_pp']:+.3f} | {r['total_saving_percent']:.3f}% |" for r in direct]
    lines += ["", "### Kiểm tra phân bố client", "",
              "Số ảnh train và tỷ trọng lớp lớn nhất của client 0 / 1 / 2. Cùng α dùng cùng partition ở mọi nhánh.", "",
              "| α | Train counts | Tỷ trọng lớp lớn nhất |", "|---:|---|---|"]
    for cell in cells:
        if cell["arm"] != "fedavg":
            continue
        parts = read(Path(cell["result"]["results_dir"]) / "partition_summary.json")
        clients = [parts["clients"][str(cid)] for cid in range(3)]
        counts = " / ".join(str(c["num_train_samples"]) for c in clients)
        shares = " / ".join(f"{100 * max(c['class_proportions'].values()):.1f}%" for c in clients)
        lines.append(f"| {cell['result']['alpha']} | {counts} | {shares} |")
    lines += ["", "![Kết quả theo mức độ non-IID](non_iid_results.png)", "", "## Kiểm tra tính toàn vẹn", "",
              "Đã kiểm tra mỗi sample ID xuất hiện đúng một lần trong partition của từng official split",
              "(training vẫn lặp qua 8 rounds), partition/test IDs ghép cặp,",
              "cấu hình và source hashes giống nhau ngoài treatment, μ hiệu dụng từng client-fit, đủ",
              "8 rounds/3 clients, không lỗi client, validation selection và accounting bytes khớp từng round.",
              "Tính lại accuracy/macro-F1 từ confusion matrices. Kiểm tra hash protocol/lock/runner snapshot.",
              "Replay FedAvg α=0.3 khớp chính xác tensors checkpoint, test report/loss, round được chọn,",
              "partition hash, validation và payload mỗi round. Replay không bao phủ cả 9 ô hoặc máy khác.", "",
              "## Diễn giải và giới hạn", "",
              "Mức tin cậy: **CAUTION**. Một seed cho mỗi điều kiện không đo được biến thiên do seed; ba α",
              "là ba điều kiện, không phải ba mẫu lặp. Không gộp chúng thành mean±SD hoặc kiểm định thống kê.",
              "Client partitions là label skew tổng hợp, không đại diện ba bệnh viện hoặc scanner/domain shift.",
              "Tám rounds không chứng minh hội tụ; μ=0.01 và lịch AdamW cố định không phải tối ưu mọi thuật toán.",
              "FedProx không thay cơ chế BatchNorm thành local BN; không gọi kết quả này là FedBN.",
              "Không đánh giá FedNova/SCAFFOLD hoặc client sampling; không xếp hạng các thuật toán chưa chạy.", "",
              "Payload gồm NumPy tensor headers, fit uploads/downloads và evaluation downloads; loại RPC",
              "envelopes, telemetry, bảo mật, retries. Không suy ra tiết kiệm thời gian mạng/năng lượng từ byte.",
              "Không dùng timing để tuyên bố speedup. BloodMNIST phân loại loại tế bào máu, không chứng minh",
              "chẩn đoán bệnh, bảo vệ riêng tư, độ tin cậy lâm sàng hoặc hiệu năng Jetson.",
              "Test set đã xuất hiện trong nghiên cứu BloodCNN trước; đây không phải holdout mới chưa quan sát.",
              "Thiết kế mới không dùng test để tune μ/seed. Phương pháp kết hợp là kiểm chứng kỹ thuật của",
              "thuật toán/codec có sẵn, không tuyên bố thuật toán mới hay novelty đã được thẩm định toàn diện.", "",
              "### ARS statistical fallacy scan — 11/11", "", "| Kiểm tra | Xử lý |", "|---|---|"]
    checks = [("Simpson's paradox", "Báo riêng từng α; không che điều kiện thất bại bằng trung bình."),
              ("Ecological fallacy", "Không suy luận bệnh nhân/bệnh viện từ virtual clients."),
              ("Berkson's paradox", "Giới hạn chọn benchmark; không suy rộng quần thể lâm sàng."),
              ("Collider bias", "Không điều chỉnh covariate theo outcomes; treatment cố định trước training."),
              ("Base rate neglect", "Dùng accuracy cùng macro-F1, giữ class supports/confusion matrices."),
              ("Regression to the mean", "Seed theo khả thi batch, không chọn kết quả cực trị; cần nhiều seed để khái quát."),
              ("Survivorship bias", "Bắt buộc đủ chín ô; giữ mọi kết quả âm."),
              ("Look-elsewhere effect", "Hai câu hỏi và ngưỡng định trước; không sweep test hay nhiều kiểm định."),
              ("Garden of forking paths", "Lưu protocol trước ma trận; đây là local protocol, không đăng ký preregistration."),
              ("Correlation/causation", "Đối chiếu có kiểm soát trong simulation; không suy ra tác động lâm sàng."),
              ("Reverse causality", "Treatment đặt trước kết quả; không có tuyên bố thời gian quan sát.")]
    lines += [f"| {name} | {detail} |" for name, detail in checks]
    notes_path = suite / "monitoring_notes.json"
    if notes_path.exists():
        lines += ["", "### Ghi nhận gián đoạn theo dõi", "",
                  "Có khoảng chờ thực tế dài hơn yêu cầu; nguyên nhân chưa xác minh. Không dùng",
                  "thời gian của suite này để suy luận speedup. Dữ liệu dưới đây chỉ ghi nhận quan sát:", ""]
        for note in read(notes_path)["observations"]:
            lines.append(f"- {note['observed_at']}: yêu cầu chờ {note['requested_wait_seconds']} s, "
                         f"tool ghi nhận {note['reported_wait_seconds']:.3f} s. {note['note']}")
    lines += ["", "## Tái lập và nguồn", "", "```powershell",
              "python -B -m unittest discover -s tests -v",
              "python -B -u scripts/run_non_iid_study.py",
              "python -B scripts/report_non_iid_study.py results/non_iid_study/<suite_directory>", "```", "",
              "Runner tạo suite mới, dừng khi lỗi hoặc quá deadline 4 giờ, kiểm tra lại khi máy resume.",
              "Không chạy lại lệnh training chỉ để xem báo cáo; lệnh report đọc artifacts hiện có.", "",
              "[FedAvg — McMahan et al.](https://proceedings.mlr.press/v54/mcmahan17a.html),",
              "[FedProx — Li et al.](https://arxiv.org/abs/1812.06127),",
              "[MedMNIST v2](https://www.nature.com/articles/s41597-022-01721-8),",
              "[nguồn BloodMNIST](https://data.mendeley.com/datasets/snkd93bnjr/1).", "",
              "ARS-Codex hỗ trợ thiết kế, code và báo cáo. Review kỹ thuật dùng agent cùng model family",
              "và shared context; không phải peer review bên ngoài hoặc xác minh cross-model."]
    (output / "non_iid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_results(cells, analysis, output)
    write_communication_report(cells, analysis, direct, suite, output)
    print(output / "non_iid_report.md")


def write_communication_report(cells, analysis, direct, suite, output):
    """Publish communication findings from the same audited Dirichlet-only matrix."""
    lines = ["# Communication research — ResNet-18 / BloodMNIST", "",
             "## Material Passport", "", "- Origin Skill: ARS-Codex experiment-agent",
             "- Origin Mode: validate / report refresh",
             f"- Origin Date: {datetime.now(timezone.utc).date()}",
             "- Verification Status: ANALYZED; FedAvg α=0.3 exact replay VERIFIED",
             "- Version Label: communication_resnet18_non_iid_v2", "",
             "Báo cáo này dùng ma trận mới: ResNet-18, toàn bộ BloodMNIST, 8 rounds,",
             "3 clients, một local epoch, seed 2026, chỉ Dirichlet α=0.1/0.3/1.0.",
             "FedAvg không nén là baseline cố định; hai nhánh đề xuất dùng FedProx μ=0.01",
             "và FedProx+FP16. Không có IID, INT8 hoặc BloodCNN trong các kết quả dưới đây.", "",
             f"Nguồn duy nhất: `{suite.resolve()}` (9 runs + 1 replay, replay không tính vào bảng).",
             "[Protocol](communication_protocol.md) · [Báo cáo đầy đủ](non_iid_report.md) · [CSV](communication_runs.csv)", "",
             "## Chất lượng và chi phí truyền thông", "",
             "| α | Phương pháp | Accuracy (%) | Macro-F1 (%) | Upload MiB | Tổng MiB |",
             "|---:|---|---:|---:|---:|---:|"]
    for cell in cells:
        r = cell["result"]
        lines.append(f"| {r['alpha']} | {LABELS[cell['arm']]} | {100*r['final_test_accuracy']:.3f} | "
                     f"{100*r['final_test_f1_macro']:.3f} | {r['total_fit_upload_model_bytes']/2**20:.3f} | "
                     f"{r['total_model_payload_bytes']/2**20:.3f} |")
    lines += ["", "## FedProx+FP16 so với FedAvg cố định", "",
              "Tiêu chí định trước ở từng α: giảm ≥15% tổng payload; accuracy và macro-F1 giảm không quá 1 điểm phần trăm.", "",
              "| α | Δaccuracy (pp) | Δmacro-F1 (pp) | Giảm upload | Giảm tổng payload | Tiêu chí |",
              "|---:|---:|---:|---:|---:|---|"]
    for r in analysis["comparisons"]:
        if r["arm"] == "fedprox_fp16":
            lines.append(f"| {r['alpha']} | {r['accuracy_delta_pp']:+.3f} | {r['f1_delta_pp']:+.3f} | "
                         f"{r['upload_saving_percent']:.3f}% | {r['total_saving_percent']:.3f}% | "
                         f"{'đạt' if r['communication_quality_pass'] else 'không đạt'} |")
    lines += ["", "Kết luận theo tiêu chí đồng thời cả ba α: **" +
              ("ĐẠT" if analysis["fedprox_fp16_communication_pass_all_alphas"] else "KHÔNG ĐẠT") + "**.",
              "Không che điều kiện thất bại bằng trung bình qua α. Một seed không cho phép kết luận độ ổn định qua seed.", "",
              "## Ảnh hưởng riêng của codec trong FedProx", "",
              "So sánh FedProx+FP16 với FedProx không nén tách ảnh hưởng của codec khỏi proximal objective.",
              "Tiết kiệm bytes thuộc FP16, không phải do μ. Training/aggregation/downloads giữ model dtype.", "",
              "| α | Δaccuracy (pp) | Δmacro-F1 (pp) | Giảm tổng payload |", "|---:|---:|---:|---:|"]
    lines += [f"| {r['alpha']} | {r['accuracy_delta_pp']:+.3f} | {r['f1_delta_pp']:+.3f} | {r['total_saving_percent']:.3f}% |" for r in direct]
    lines += ["", "![Chất lượng và payload theo α](communication_tradeoff.png)", "",
              "Payload là serialized model bytes: NumPy headers, fit uploads/downloads và evaluation downloads.",
              "Không gồm RPC envelopes, telemetry, mã hóa hoặc retries; không suy ra network speed hay năng lượng.",
              "FedProx cải thiện macro-F1 thực dụng chỉ tại α=1.0, giảm ở α=0.1/0.3; xem toàn bộ contrast và",
              "11 kiểm tra ARS trong [báo cáo chính](non_iid_report.md). Không tuyên bố cải thiện nhất quán hay hiệu quả lâm sàng.", "",
              "## Tái tạo từ artifacts đã có", "", "```powershell",
              f"python -B scripts/report_communication_study.py {suite.as_posix()}", "```", "",
              "Lệnh report chỉ đọc và kiểm tra suite mới, không training lại. Các kết quả cũ có IID đã được",
              "gỡ theo yêu cầu; không còn được dùng làm bằng chứng nghiên cứu hiện tại."]
    (output / "communication_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for source, target in (("non_iid_runs.csv", "communication_runs.csv"),
                           ("non_iid_results.png", "communication_tradeoff.png"),
                           ("non_iid_results.svg", "communication_tradeoff.svg")):
        shutil.copyfile(output / source, output / target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    generate(parser.parse_args().suite)
