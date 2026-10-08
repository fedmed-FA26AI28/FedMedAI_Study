# FedMedAI

Hệ thống phân loại ảnh y tế 2D bằng PyTorch và Flower, dùng **CNN hai block**
(20.104 tham số với BloodMNIST 8 lớp). Hướng triển khai hiện tại: **PC/laptop
trong cùng mạng LAN**, sau đó mới mở rộng sang thiết bị khác.

Hai phương pháp chính cho đợt triển khai:

| Vai trò | ID trong cấu hình | Cơ chế |
|---|---|---|
| Baseline | `fedavg` | Tổng hợp model theo số mẫu train |
| Cải thiện chính tạm thời | `coverage_calibrated` | Train-count logit calibration + coverage-weighted head proximal; tổng hợp FedAvg |

Candidate đang được kiểm chứng tiếp; chưa có benchmark đa máy/Jetson trong bước
chuẩn bị này. FedProx và các ablation còn trong mã và kiểm thử, ngoài ma trận chính.
`proposed` và `fednova` là extension chưa triển khai.

## Bắt đầu

Chạy từ gốc repo bằng Python >=3.10:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -B -m unittest discover -s tests -v
```

Dependency hiện tại gồm Flower/Ray simulation. Việc tách Ray thành optional extra
cho máy client nằm trong kế hoạch; không cần chạy Ray khi dùng server/client CLI.

Smoke dùng cache BloodMNIST, không tự download:

```bash
python -c "import medmnist; [medmnist.BloodMNIST(split=s, download=True) for s in ('train','val','test')]"
python -B scripts/run_experiment.py --config configs/smoke.yaml --algorithm fedavg
python -B scripts/run_experiment.py --config configs/smoke.yaml --algorithm coverage_calibrated
```

Lệnh smoke chỉ xác minh pipeline với dữ liệu giới hạn. Chọn checkpoint bằng
validation; test chỉ đánh giá cuối run.

## Chạy trên các máy khác nhau

Server và client dùng cùng revision mã, Flower version, protocol và dữ liệu. Với
config mặc định cần **3 client**, logical IDs `0`, `1`, `2`:

```bash
# Máy server
python -B -m server.server --config configs/smoke.yaml --address 0.0.0.0:8080
# Trên từng máy client: thay SERVER_LAN_IP và CLIENT_ID bằng giá trị thật
python -B -m client.client --config configs/smoke.yaml --client-id CLIENT_ID --server-address SERVER_LAN_IP:8080 --max-wait-time 120
```

Để chạy candidate qua mạng, sao chép config vào `configs/local/`, đặt
`algorithm.name: coverage_calibrated` và dùng file đó trên mọi máy. Server truyền
loss coefficients cho client. Hướng dẫn cụ thể: [run.md](run.md).

Client hiện vẫn nạp toàn bộ train/val cache rồi chọn partition theo seed; server
cũng dựng partition và dùng union validation. Chế độ chỉ nạp shard riêng, handshake
protocol và round timeout cấu hình được là công việc tiếp theo.

## Cấu trúc

```text
algorithms/   FedAvg, Coverage + calibration, ablation và codecs
client/       Flower client, local training, evaluation
server/       Flower server
models/       CNN duy nhất
datasets/    MedMNIST, partition và data utilities (mã nguồn)
monitoring/   Metrics, resource và network utilities
experiments/  Simulation, centralized, artifacts và partition CLI
configs/      experiment.yaml, smoke.yaml, hardware presets
scripts/      CLI mô phỏng, ping và chuẩn bị SCP
tests/       Kiểm thử offline và integration checks riêng
docs/        Kế hoạch triển khai
```

## Tài liệu và phạm vi chia sẻ

- [Cách chạy và xử lý lỗi](run.md).
- [Kế hoạch client thực tế và điều kiện nghiệm thu](docs/real_clients_plan.md).
- [Quy tắc đóng gói repo](docs/repository_scope.md).

Nghiên cứu đã hoàn tất và đang tiếp tục được giữ local, ngoài cây mã chia sẻ qua
Git. Dataset, trọng số, outputs, địa chỉ máy và credentials cũng được ignore.
Core algorithm/training tests vẫn đi cùng repo. Kiểm thử loopback không thay thế
xác nhận trên nhiều máy thật.
