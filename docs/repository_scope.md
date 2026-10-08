# Phạm vi repository chia sẻ cho client

Repo phục vụ cài đặt và chạy FedMedAI: FedAvg baseline và `coverage_calibrated`
là candidate chính tạm thời. Tài liệu triển khai công khai ở `README.md`, `run.md`
và `docs/real_clients_plan.md`.

| Đưa vào cây mã chia sẻ | Giữ local và ignore |
|---|---|
| Root packages dùng cho training/server/client | `research/` toàn bộ: đã xong, đang pause và tiếp tục |
| `datasets/` mã nguồn, CNN và các loss/codec đang sống | `data/`, checkpoints, results, TensorBoard, model binaries |
| CLI thông thường, partition, ping, SCP helper | Study configs, study runners/reporters, research orchestration modules |
| Core tests và integration check độc lập | Tests phụ thuộc nghiên cứu/lock/protocol/runner riêng |
| Config mẫu không chứa địa chỉ thật | `configs/local/`, `.env`, certs, keys, logs |
| Kế hoạch triển khai hiện hành | AGENTS/STATUS local, kế hoạch nghiên cứu/cleanup dài |

Ignore phải tách theo phụ thuộc, không ẩn cả `tests/` hoặc toàn bộ phương pháp
coverage. Các test core vẫn kiểm tra loss/gradient/class counts/config/training,
artifact/partition/CLI/layout. Không sửa executable source chỉ để dọn cây Git.

File đã tracked cần bỏ khỏi index bằng `git rm --cached`; file trên máy vẫn giữ
nguyên. Việc này tạo staged deletions cho commit tiếp theo. Không dùng `git clean`
để dọn workspace nghiên cứu. Không tự dùng `git add -f` với nghiên cứu đã ignore.

Ignore và bỏ index chỉ thay cây ở commit tiếp theo, không xóa dữ liệu khỏi Git
history, không làm nhỏ các commit cũ. Bước này không rewrite history hoặc push.
Khi phát hành, review `git diff --cached --stat`, stage mã runtime cần thiết và
kiểm tra một clean export/clone. Source root packages hiện đang có thay đổi local
chưa commit; chỉ riêng bỏ index nghiên cứu chưa tạo ra một bản phát hành chạy được.

Kiểm tra public tree bằng `python -B -m unittest discover -s tests -v`. Workspace
local vẫn có test nghiên cứu nên discovery ở máy nghiên cứu có thể chạy nhiều
test hơn clean clone. Khi cần tiếp tục nghiên cứu, dùng nguồn/lock/snapshot local
đã thực thi; không diễn giải source hash của deployment như source hash lịch sử.

Hồ sơ backup, inventory và xác minh lần chuẩn bị 08/10/2026 lưu local tại
`research/repo_preparation_20261008/`; không phải dependency của client.

Lần chuẩn bị này đã bỏ index đúng 25 file nghiên cứu được người dùng duyệt.
AGENTS/STATUS và bytecode tracked từ trước vẫn còn trong index; rule ignore không
tự loại chúng. Bộ xét duyệt tự động từ chối thao tác bỏ index hàng loạt vì vượt
phạm vi nghiên cứu. Bản export kiểm thử là cây dự kiến theo ignore, chưa phải clone
của HEAD hay toàn bộ index hiện tại. Phát hành cần review riêng các mục còn tracked.
