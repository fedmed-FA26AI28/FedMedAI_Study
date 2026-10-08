# Chạy FedMedAI trên PC/laptop cùng LAN

## Chuẩn bị mỗi máy

Dùng cùng source revision và Python >=3.10. Tạo/activate virtual environment rồi:

```bash
python -m pip install -e .
python -B -m unittest discover -s tests -v
python -c "import flwr, torch; print(flwr.__version__, torch.__version__, torch.cuda.is_available())"
python -c "import medmnist; [medmnist.BloodMNIST(split=s, download=True) for s in ('train','val','test')]"
```

Ở pilot hiện tại, các client cần cache train/val đầy đủ; server cần train/val/test.
Cùng dataset version, preprocessing, seed, num_clients, alpha và sample caps để
partition khớp nhau. SCP JSON partition chưa đủ: loader chưa đọc JSON/shard riêng.

## Cấu hình baseline và candidate

Tạo thư mục `configs/local/` (đã ignore). Copy `configs/smoke.yaml` thành
`configs/local/fedavg.yaml` và `configs/local/coverage_calibrated.yaml`.
Giữ các mục giống nhau, chỉ đổi `algorithm` và thư mục output theo method:

```yaml
# configs/local/coverage_calibrated.yaml: thay section algorithm
algorithm:
  name: coverage_calibrated
  head_mu: 1.0
  coverage_kappa: 32.0
  logit_tau: 1.0
  prior_smoothing: 1.0
communication:
  uplink_codec: none
```

File baseline dùng `algorithm.name: fedavg`, `communication.uplink_codec: none`.
Gửi cùng config method tới tất cả máy cho mỗi run; `dataset.root` và output paths
có thể sửa theo filesystem từng máy. Hiện chưa có flag hợp nhất local config.
Mặc định `num_clients: 3`; nếu chỉ có hai client, đổi thành `2` trên mọi máy.
Mỗi client có ID duy nhất trong `[0, num_clients-1]`. Dùng batch size/local epochs
và learning rate giống nhau cho hai method.

## Chạy LAN bằng CLI hiện có

Chọn IP LAN của server, kiểm tra client truy cập được server và cho phép inbound
TCP 8080 trong firewall của server trên mạng lab. Client chỉ cần kết nối outbound.
Start server trước, rồi đủ client trên các máy/terminal riêng:

```bash
python -B -m server.server --config configs/local/fedavg.yaml --address 0.0.0.0:8080
python -B -m client.client --config configs/local/fedavg.yaml --client-id 0 --server-address SERVER_LAN_IP:8080 --max-wait-time 120
python -B -m client.client --config configs/local/fedavg.yaml --client-id 1 --server-address SERVER_LAN_IP:8080 --max-wait-time 120
# Nếu num_clients=3, chạy thêm ID 2 trên terminal/máy còn lại.
```

`SERVER_LAN_IP` phải thay bằng IP thật, không dùng `0.0.0.0`. Chạy xong baseline,
khởi động lại server và các client với file `coverage_calibrated.yaml`. Server sẽ
truyền tên phương pháp và coefficients xuống client. Với pilot đầu dùng 1–2 rounds
và sample caps nhỏ; giữ config đầy đủ riêng cho benchmark sau smoke.

Hiện server chờ đủ client và chưa expose round timeout. Nếu thiếu client, kiểm tra
ID/count/IP/firewall trước; không coi việc đang chờ là đã chạy thành công. Nếu một
client lỗi giữa round, round bị từ chối. Chưa có exact resume giữa run; giữ logs
và chạy attempt mới. `--max-wait-time` là giới hạn client thử kết nối, không phải
thời gian tối đa local training.

## Xác minh

```bash
# Synthetic integration check hiện có: một client FedAvg qua loopback
python -B tests/check_network.py
# Flower/Ray simulation check, tách riêng khỏi network CLI
python -B tests/check_flower.py --algorithm coverage_calibrated
```

Cần kiểm tra `run_manifest.json` có `status: completed`, đủ client/round, cấu hình
loss trong telemetry, partition/sample IDs khớp, best checkpoint theo validation
và test report cuối run. `results/`, `checkpoints/`, `runs/` giữ local. Byte model
payload không bao gồm toàn bộ RPC traffic/retries; upload/download time chưa đo
được phải để `null`. CPU/GPU có thể khác sai số số thực; không yêu cầu bit-identical.

## Giới hạn và bước tiếp theo

Chưa xác nhận pilot LAN đa máy trong tác vụ chuẩn bị này. Client device label từ
CLI chưa phản ánh đúng mọi cấu hình hardware; shard riêng, handshake hashes,
timeout, optional Ray và config templates là backlog trong
[kế hoạch triển khai](docs/real_clients_plan.md). CLI transport hiện tại chưa expose
TLS certificates; kế hoạch ngoài LAN cần VPN/TLS và client identity.
