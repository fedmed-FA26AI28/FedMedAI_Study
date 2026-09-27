## Material Passport

- Origin Skill: ARS-Codex experiment-agent
- Origin Mode: plan / run
- Origin Date: 2026-09-26
- Verification Status: UNVERIFIED — protocol trước khi chạy ma trận mới
- Version Label: non_iid_protocol_v2_resnet18_full

## Yêu cầu và câu hỏi nghiên cứu

Thiết kế này thay thế ma trận IID/nhiều seed của nghiên cứu compression trước.
Giữ FedAvg làm đối chứng cố định; phương pháp đề xuất dùng FedProx thực sự với
proximal coefficient khác 0. Chỉ đánh giá Dirichlet α ∈ {0.1, 0.3, 1.0}, seed=2026.

RQ1: FedProx có cải thiện phân loại khi non-IID so với FedAvg cố định không?
RQ2: FedProx + FP16 uplink có giảm payload với mức suy giảm chất lượng nhỏ không?

## Ma trận cố định: 9 runs

| Nhánh | Thuật toán local | μ | Uplink | Vai trò |
|---|---|---:|---|---|
| FedAvg | Cross-entropy | 0 hiệu dụng | FP32/none | Baseline cố định |
| FedProx | Cross-entropy + μ/2 × ||w − w_global||² | 0.01 | FP32/none | Đề xuất cho non-IID |
| FedProx + FP16 | Cùng FedProx | 0.01 | FP16 | Biến thể giảm truyền thông |

Mỗi nhánh chạy một lần tại mỗi α, cùng seed=2026. Không chạy IID, không thêm seed,
không thay thế FedAvg baseline bằng phiên bản nén. μ=0.01 là giá trị mặc định
đã có trong repository, cố định trước khi xem kết quả mới; không sweep theo test.
Không giả định FedProx chắc chắn cải thiện; kết quả âm vẫn được báo đầy đủ.

FedProx chia sẻ phép tổng hợp trọng số theo số mẫu với FedAvg theo định nghĩa
thuật toán. Khác biệt thực nằm ở local objective; kiểm tra telemetry từng client
để đảm bảo μ=0 ở baseline và μ=0.01 ở hai nhánh FedProx. Dùng chung lớp server
không có nghĩa là đổi tên FedAvg thành FedProx.

## Baseline được khóa

`non_iid_baseline.lock.json` khóa cấu hình ResNet-18 mặc định theo lựa chọn của
người dùng, 3 clients tham gia đầy đủ, 8 rounds (trong khoảng 8–10 được chấp thuận),
1 local epoch, batch=32, AdamW lr=0.0001, weight decay=0.01, reset
optimizer mỗi local fit. Dữ liệu BloodMNIST đầy đủ: 11,959 train / 1,712 val /
3,421 test, 28×28; không pretrained, không cắt subset. Sequential CPU simulation,
deterministic=true, 8 CPU threads; monitoring bật. Không dùng kết quả BloodCNN
cũ làm baseline cho ResNet-18.

Chỉ α thay đổi theo ba điều kiện đã yêu cầu; seed luôn bằng 2026. Output paths được
đổi để tách artifacts. Mã training/model/data/aggregation/codec được khóa bằng
SHA256 trước khi chạy ma trận. Không chỉnh learning rate, rounds, batch, model hoặc
sample policy để làm baseline yếu đi. Audit/report code được phép bổ sung.

Preflight chỉ dùng số lượng mẫu và timing dữ liệu tổng hợp, không xem kết quả
validation/test: seed 0 và 42 tạo một client có train_count mod 32 = 1 ở một α,
gây lỗi BatchNorm với batch cuối một ảnh. Chọn seed 2026 trước khi training để
giữ tất cả ảnh, không drop_last hoặc đổi kiến trúc. Train counts theo client:
α=0.1: [4497, 7108, 354]; α=0.3: [4779, 3864, 3316]; α=1: [3158, 4002, 4799].
Đây là lựa chọn seed theo khả thi kỹ thuật, không phải ngẫu nhiên hoàn toàn hay
bằng chứng ổn định qua seed. Timing 16 batches tổng hợp cho FedProx lần lượt
7.750/5.744/4.476/3.663 s ở 1/2/4/8 threads; chọn 8 threads cho tất cả nhánh.

Các nhánh cùng α phải có cùng partition hash, sample IDs, khởi tạo và lịch shuffle
local theo seed. Runner lưu baseline lock, protocol, code snapshot và config trước
khi training. Sau 9 runs, lặp lại FedAvg α=0.3 với cùng cấu hình và seed; checkpoint
tensors, test report, validation từng round và payload phải khớp chính xác lần
chạy ResNet-18 trong ma trận. Replay này chỉ kiểm tra tái lập, không thêm seed và
không được tính thành lần lặp thống kê. Không chạy lại nghiên cứu BloodCNN cũ.

## Đánh giá và tiêu chí

- Checkpoint: chọn macro-F1 cao nhất trên pooled validation, lấy round sớm nhất
  nếu bằng nhau. Test chính thức chỉ đánh giá sau khi kết thúc mỗi run.
- Báo accuracy, macro-F1, loss, per-class metrics, checkpoint round, worst-client
  validation accuracy ở round cuối, tổng upload/download/serialized model bytes.
- Báo Δaccuracy và Δmacro-F1 theo từng α; không gộp ba α thành ba lần lặp seed.
- RQ1: ghi nhận cải thiện thực dụng tại một α nếu macro-F1 tăng ≥1 điểm phần trăm
  và accuracy không giảm. Chỉ gọi cải thiện nhất quán nếu đạt ở cả ba α. Những
  thay đổi nhỏ hơn vẫn báo số thực, không diễn giải là khác biệt có ý nghĩa.
- RQ2: mỗi α cần giảm ≥15% tổng model payload, với accuracy và macro-F1 giảm
  không quá 1 điểm phần trăm so với FedAvg. Đồng thời so sánh FP16 với FedProx
  không nén để tách chi phí quantization khỏi tác động proximal term.
- Không đổi tiêu chí hoặc μ sau khi xem test. Không dùng p-value/CI/mean±SD của
  nhiều seed: nghiên cứu này chỉ có một seed theo yêu cầu.
- Round thất bại hoặc ô thiếu làm ma trận không đầy đủ; không âm thầm loại bỏ.
- Budget supervisor: 4 giờ theo đồng hồ, kiểm tra sau các khoảng chờ tối đa 30 s;
  máy sleep không thể chạy watchdog, kiểm tra lại khi resume. Không so sánh tốc độ
  từ các runs có khoảng gián đoạn. Không suy ra năng lượng hoặc network speed
  từ số byte mô phỏng.

## Lựa chọn phương pháp và giới hạn

[FedProx — Li et al., MLSys 2020](https://arxiv.org/abs/1812.06127) xử lý
heterogeneous federated optimization bằng proximal objective, phù hợp để thử
với label skew và đã có implementation kiểm thử trong repository.
[FedBN — Li et al., ICLR 2021](https://arxiv.org/abs/2102.07623) nhắm feature shift
và BatchNorm local; ResNet-18 có BatchNorm nhưng thí nghiệm này tập trung label
skew và chọn FedProx đã có implementation. BatchNorm vẫn dùng aggregation hiện
tại, không tuyên bố đây là FedBN. [SCAFFOLD — Karimireddy et al., ICML 2020](https://proceedings.mlr.press/v119/karimireddy20a.html)
là hướng khác dùng control variates; chưa triển khai/đánh giá trong thí nghiệm này.
Không tuyên bố đã so sánh hoặc xếp hạng tất cả thuật toán được người dùng nêu.

Nền tảng: [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html),
[MedMNIST v2](https://www.nature.com/articles/s41597-022-01721-8),
[nguồn ảnh BloodMNIST](https://data.mendeley.com/datasets/snkd93bnjr/1).
Đây là kiểm chứng kỹ thuật trên benchmark, không phải thuật toán mới, thử nghiệm
đa bệnh viện, bảo đảm riêng tư hoặc xác thực lâm sàng. Kết quả cũ đã được xem;
test set không phải một tập holdout mới chưa từng quan sát. Không dùng kết quả
cũ để tối ưu μ hay lựa chọn seed. Artifacts nghiên cứu cũ đã được gỡ theo yêu cầu;
việc xóa không thay đổi lịch sử test set đã được quan sát.

Ghi chú bảo trì sau thực nghiệm (2026-09-26): đoạn tình trạng lưu trữ phía trên
được cập nhật sau khi xóa bộ cũ. Protocol/config/source snapshots trong suite
đã chạy giữ nguyên để kiểm tra provenance; không thay đổi thiết kế hoặc ngưỡng.
