# Communication research — ResNet-18 / BloodMNIST

## Material Passport

- Origin Skill: ARS-Codex experiment-agent
- Origin Mode: validate / report refresh
- Origin Date: 2026-09-26
- Verification Status: ANALYZED; FedAvg α=0.3 exact replay VERIFIED
- Version Label: communication_resnet18_non_iid_v2

Báo cáo này dùng ma trận mới: ResNet-18, toàn bộ BloodMNIST, 8 rounds,
3 clients, một local epoch, seed 2026, chỉ Dirichlet α=0.1/0.3/1.0.
FedAvg không nén là baseline cố định; hai nhánh đề xuất dùng FedProx μ=0.01
và FedProx+FP16. Không có IID, INT8 hoặc BloodCNN trong các kết quả dưới đây.

Nguồn duy nhất: `D:\FederatedLearning\fedmedai\results\non_iid_study\suite_20260926T103613Z` (9 runs + 1 replay, replay không tính vào bảng).
[Protocol](communication_protocol.md) · [Báo cáo đầy đủ](non_iid_report.md) · [CSV](communication_runs.csv)

## Chất lượng và chi phí truyền thông

| α | Phương pháp | Accuracy (%) | Macro-F1 (%) | Upload MiB | Tổng MiB |
|---:|---|---:|---:|---:|---:|
| 0.1 | FedAvg | 76.966 | 75.601 | 1024.856 | 3074.568 |
| 0.1 | FedProx | 76.703 | 72.474 | 1024.856 | 3074.568 |
| 0.1 | FedProx + FP16 | 76.849 | 73.388 | 512.609 | 2562.320 |
| 0.3 | FedAvg | 87.635 | 86.160 | 1024.856 | 3074.568 |
| 0.3 | FedProx | 86.349 | 84.758 | 1024.856 | 3074.568 |
| 0.3 | FedProx + FP16 | 87.255 | 85.737 | 512.609 | 2562.320 |
| 1.0 | FedAvg | 88.717 | 87.174 | 1024.856 | 3074.568 |
| 1.0 | FedProx | 89.652 | 88.350 | 1024.856 | 3074.568 |
| 1.0 | FedProx + FP16 | 89.272 | 87.829 | 512.609 | 2562.320 |

## FedProx+FP16 so với FedAvg cố định

Tiêu chí định trước ở từng α: giảm ≥15% tổng payload; accuracy và macro-F1 giảm không quá 1 điểm phần trăm.

| α | Δaccuracy (pp) | Δmacro-F1 (pp) | Giảm upload | Giảm tổng payload | Tiêu chí |
|---:|---:|---:|---:|---:|---|
| 0.1 | -0.117 | -2.214 | 49.982% | 16.661% | không đạt |
| 0.3 | -0.380 | -0.423 | 49.982% | 16.661% | đạt |
| 1.0 | +0.555 | +0.655 | 49.982% | 16.661% | đạt |

Kết luận theo tiêu chí đồng thời cả ba α: **KHÔNG ĐẠT**.
Không che điều kiện thất bại bằng trung bình qua α. Một seed không cho phép kết luận độ ổn định qua seed.

## Ảnh hưởng riêng của codec trong FedProx

So sánh FedProx+FP16 với FedProx không nén tách ảnh hưởng của codec khỏi proximal objective.
Tiết kiệm bytes thuộc FP16, không phải do μ. Training/aggregation/downloads giữ model dtype.

| α | Δaccuracy (pp) | Δmacro-F1 (pp) | Giảm tổng payload |
|---:|---:|---:|---:|
| 0.1 | +0.146 | +0.913 | 16.661% |
| 0.3 | +0.906 | +0.978 | 16.661% |
| 1.0 | -0.380 | -0.521 | 16.661% |

![Chất lượng và payload theo α](communication_tradeoff.png)

Payload là serialized model bytes: NumPy headers, fit uploads/downloads và evaluation downloads.
Không gồm RPC envelopes, telemetry, mã hóa hoặc retries; không suy ra network speed hay năng lượng.
FedProx cải thiện macro-F1 thực dụng chỉ tại α=1.0, giảm ở α=0.1/0.3; xem toàn bộ contrast và
11 kiểm tra ARS trong [báo cáo chính](non_iid_report.md). Không tuyên bố cải thiện nhất quán hay hiệu quả lâm sàng.

## Tái tạo từ artifacts đã có

```powershell
python -B scripts/report_communication_study.py results/non_iid_study/suite_20260926T103613Z
```

Lệnh report chỉ đọc và kiểm tra suite mới, không training lại. Các kết quả cũ có IID đã được
gỡ theo yêu cầu; không còn được dùng làm bằng chứng nghiên cứu hiện tại.
