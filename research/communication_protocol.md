# Communication protocol — nghiên cứu hiện hành

## Material Passport

- Origin Skill: ARS-Codex experiment-agent
- Origin Mode: report refresh
- Origin Date: 2026-09-26
- Verification Status: ANALYZED; protocol authority linked below
- Version Label: communication_resnet18_non_iid_v2

Protocol chính: [non_iid_protocol.md](non_iid_protocol.md).
Khóa baseline: [non_iid_baseline.lock.json](non_iid_baseline.lock.json).
Bản protocol được chụp trước khi chạy nằm trong suite, được kiểm tra bằng SHA256.
Tài liệu này tóm tắt thiết kế đã thực hiện, không phải một preregistration mới.

ResNet-18 không pretrained; toàn bộ BloodMNIST (11,959 train / 1,712 val / 3,421 test),
3 clients tham gia đầy đủ, 8 rounds, 1 local epoch, batch=32, AdamW lr=0.0001,
weight decay=0.01, seed=2026, deterministic sequential CPU, 8 threads.
Chỉ Dirichlet α=[0.1,0.3,1.0], không IID. Seed chọn trước outcomes để tránh
singleton BatchNorm batches; mọi mẫu được giữ lại.

| Nhánh | Local objective | Uplink |
|---|---|---|
| FedAvg cố định | Cross-entropy, μ hiệu dụng=0 | Không nén |
| FedProx | Cross-entropy + μ/2 × ||w−w_global||², μ=0.01 | Không nén |
| FedProx+FP16 | Cùng FedProx | FP16 |

9 runs (3 α × 3 nhánh) và 1 replay FedAvg α=0.3 cùng seed, không tính replay vào so sánh.
Checkpoint chọn bằng pooled validation macro-F1; test chỉ đánh giá cuối mỗi run.
Đối chiếu cấu hình, sample IDs, partition/source hashes, telemetry μ và model bytes.

Tiêu chí communication định trước cho mỗi α: giảm ít nhất 15% tổng payload,
accuracy và macro-F1 giảm không quá 1 điểm phần trăm so với FedAvg.
Chỉ gọi đạt đồng thời khi cả ba α đều đạt. So sánh trực tiếp FedProx+FP16 với
FedProx không nén để tách ảnh hưởng codec khỏi proximal objective.
Một seed: không mean±SD/CI qua α; báo đầy đủ điều kiện thất bại.

Payload gồm serialized tensor headers, fit uploads/downloads và evaluation downloads.
Không bao gồm RPC envelopes, telemetry, bảo mật, retries; không suy ra speedup hoặc năng lượng.
Không tuyên bố novelty thuật toán hoặc xác thực lâm sàng.

Kết quả và giới hạn: [communication_report.md](communication_report.md).
Bộ kết quả cũ có IID đã được gỡ theo yêu cầu; test set từng được quan sát trong
nghiên cứu trước, nên việc xóa artifacts không biến nó thành holdout mới.
