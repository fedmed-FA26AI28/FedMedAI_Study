# FedMedAI — Federated Learning for Medical Image Classification
## Kế hoạch nghiên cứu và triển khai trên Jetson Orin / Jetson Nano

## Cập nhật thực nghiệm đã hoàn tất — 2026-09-26

Phạm vi hiện tại đã chốt: ResNet-18 không pretrained, toàn bộ BloodMNIST,
3 clients, 8 rounds, 1 local epoch, AdamW lr=0.0001, batch=32, seed=2026.
Chỉ chạy Dirichlet α=[0.1, 0.3, 1.0]. FedAvg không nén là đối chứng cố định;
hai nhánh so sánh là FedProx μ=0.01 và FedProx+FP16.

Đã hoàn tất 9 runs và 1 replay FedAvg α=0.3 khớp chính xác. FedProx tăng
macro-F1 1.176 điểm tại α=1.0 nhưng giảm 3.127/1.401 điểm tại α=0.1/0.3.
FedProx+FP16 giảm 16.661% tổng payload; tiêu chí chất lượng đạt tại α=0.3/1.0,
không đạt tại α=0.1. Đây là kết quả một seed, không chứng minh cải thiện nhất quán.

Nguồn hiện hành: [protocol](research/non_iid_protocol.md),
[kết quả đầy đủ](research/non_iid_report.md),
[communication research](research/communication_report.md), [STATUS](STATUS.md).
Kết quả cũ có IID đã được gỡ theo yêu cầu. Những phần khảo sát, centralized,
DermaMNIST, nhiều clients và Jetson bên dưới là lộ trình mở rộng, không phải
thực nghiệm đã hoàn tất hoặc yêu cầu chạy thêm trong ma trận hiện tại.

> **Core direction:** Comprehensive Federated Learning survey → architecture selection → Server–Client FL implementation → lightweight 2D medical image classification → heterogeneous edge deployment.

---

# 1. Tổng quan đề tài

FedMedAI nghiên cứu và xây dựng một hệ thống **Federated Learning (FL)** cho bài toán **phân loại ảnh y tế 2D**.

Mục tiêu không chỉ là xây dựng một demo FL, mà yêu cầu sinh viên:

1. Khảo sát toàn diện Federated Learning.
2. Phân loại và so sánh các kiến trúc/setting/thuật toán FL.
3. Từ literature và constraints của đề tài, **tự lập luận để lựa chọn Server–Client FL**.
4. Xây dựng centralized baseline.
5. Xây dựng FedAvg baseline.
6. Chọn và triển khai một phương pháp cải tiến.
7. Mô phỏng statistical heterogeneity bằng Dirichlet.
8. Triển khai FL thực trên **5 Jetson Orin + 5 Jetson Nano**.
9. Đánh giá system/resource heterogeneity.
10. Thực hiện ablation, error analysis và resource/communication analysis.

Lộ trình mở rộng bao gồm centralized baseline, FedAvg, non-IID, một FL improvement,
evaluation, ablation và reproducibility; phạm vi thực nghiệm hiện hành được chốt ở đầu tài liệu.

---

# 2. Nguyên tắc nghiên cứu

## 2.1. Không mặc định thuật toán/kiến trúc ngay từ đầu

Sinh viên **không bắt đầu bằng việc mặc định**:

> Flower + FedAvg + Server–Client.

Thay vào đó:

```text
FL Survey
    ↓
FL Taxonomy
    ↓
Architecture Comparison
    ↓
Algorithm Comparison
    ↓
Medical / Edge FL Review
    ↓
Research Gap
    ↓
Architecture Selection
    ↓
Algorithm Selection
    ↓
Implementation
```

## 2.2. Nguyên tắc

> **Explore broadly → Compare systematically → Choose deliberately → Implement deeply.**

---

# 3. Research Problem

Medical image data thường phân tán giữa nhiều cơ sở và chịu các ràng buộc về:

- privacy;
- data ownership;
- governance;
- data sharing.

FL cho phép các client giữ raw data tại chỗ và cộng tác huấn luyện global model.

FedMedAI tập trung vào:

### 3.1. Statistical heterogeneity

Dữ liệu giữa các client có thể non-IID.

### 3.2. System/resource heterogeneity

Các edge clients có năng lực khác nhau:

- 5 Jetson Orin;
- 5 Jetson Nano.

### 3.3. Communication overhead

Model updates phải được truyền qua network.

### 3.4. Fairness

Các client hoặc minority classes có thể có performance thấp hơn.

### 3.5. Convergence

Non-IID và heterogeneous resources có thể ảnh hưởng đến convergence.

---

# 4. Research Questions

### RQ1 — FL architectures

**What are the major Federated Learning architectures and deployment settings, and what are their trade-offs for medical and edge AI?**

### RQ2 — Architecture selection

**Why is a centralized server–client FL architecture appropriate for this project compared with decentralized/serverless FL?**

### RQ3 — Non-IID

**How does statistical heterogeneity affect the performance and convergence of FedAvg in 2D medical image classification?**

### RQ4 — System heterogeneity

**How does the resource heterogeneity between Jetson Orin and Jetson Nano affect FL training time, convergence and communication efficiency?**

### RQ5 — Improved FL

**Can a selected FL improvement maintain or improve model performance under statistical and system heterogeneity?**

---

# 5. Phase 0 — Comprehensive Federated Learning Survey

Đây là **giai đoạn bắt buộc trước implementation**.

## 5.1. FL Fundamentals

Sinh viên phải hiểu:

- Centralized Learning;
- Distributed Learning;
- Federated Learning;
- sự khác nhau giữa Distributed Learning và Federated Learning.

---

# 6. FL Taxonomy

Sinh viên phải xây dựng taxonomy tối thiểu:

```text
Federated Learning
│
├── 1. Architecture
│   ├── Centralized / Server-based FL
│   └── Decentralized / Serverless / Peer-to-Peer FL
│
├── 2. Deployment Setting
│   ├── Cross-Silo FL
│   └── Cross-Device FL
│
├── 3. Data Heterogeneity
│   ├── IID
│   ├── Label Distribution Skew
│   ├── Quantity Skew
│   ├── Feature Shift
│   └── Concept Shift
│
├── 4. Optimization / Aggregation
│   ├── FedAvg
│   ├── FedProx
│   ├── FedBN
│   ├── FedNova
│   ├── SCAFFOLD
│   └── Other methods
│
├── 5. Communication Efficiency
│   ├── Client Selection
│   ├── Compression
│   ├── Quantization
│   ├── Sparsification
│   └── Federated Distillation
│
├── 6. Privacy / Security
│   ├── Differential Privacy
│   ├── Secure Aggregation
│   └── Other mechanisms
│
├── 7. Personalization
│   ├── Global model
│   ├── Local personalization
│   └── Clustered/personalized FL
│
└── 8. System Heterogeneity
    ├── Compute
    ├── Memory
    ├── Network
    ├── Availability
    └── Energy
```

---

# 7. Centralized FL vs Decentralized FL

## 7.1. Centralized / Server–Client FL

```text
                    FL Server
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Client 1     Client 2     Client N
```

Server:

- điều phối rounds;
- gửi global model;
- nhận local updates;
- aggregate;
- tạo global model mới.

**FedAvg** là ví dụ điển hình.

## 7.2. Decentralized FL

```text
        Client 1
        ↙     ↘
 Client 2 ←→ Client 3
        ↘     ↙
        Client 4
```

Không phụ thuộc vào một central aggregation server duy nhất.

Sinh viên phải tìm hiểu:

- peer-to-peer;
- gossip;
- serverless/distributed aggregation;
- topology;
- synchronization.

**Không triển khai decentralized FL trong core scope**, nhưng phải khảo sát và giải thích trade-off.

---

# 8. Cross-Silo vs Cross-Device

## Cross-Silo

Ví dụ:

```text
Hospital A ─┐
Hospital B ─┤
Hospital C ─┼──→ FL Server
Hospital D ─┤
Hospital E ─┘
```

Số client thường nhỏ và mỗi client có thể đại diện cho một tổ chức.

## Cross-Device

```text
Client 1
Client 2
Client 3
...
Client N
```

Client có thể là:

- smartphone;
- IoT;
- edge device;
- embedded device.

Đặc điểm:

- resource không đồng nhất;
- communication không ổn định;
- client availability khác nhau;
- data phân tán.

### FedMedAI

Sau literature review, project lựa chọn:

> **Centralized Server–Client + small-scale Cross-Device/Edge FL**

với:

> **5 Orin + 5 Nano**

---

# 9. Decision Gate — lựa chọn architecture

Sinh viên phải lập bảng so sánh trước khi coding.

| Tiêu chí | Server–Client | Decentralized |
|---|---|---|
| Medical FL | Phù hợp | Có thể |
| Implementation | Dễ hơn | Phức tạp hơn |
| Aggregation | Central server | Distributed |
| Monitoring | Dễ | Khó hơn |
| 10 Jetson | Phù hợp | Có thể |
| Flower | Phù hợp | Không phải core |
| 15 tuần | **Khả thi** | Rủi ro cao |
| Research extension | Có | Có |

### Decision

> **Centralized Server–Client FL được chọn làm architecture implementation chính.**

Lưu ý: đây là **kết quả của literature review + project constraints**, không phải assumption ban đầu.

---

# 10. Phase 1 — Algorithm Survey

Sau khi chọn architecture, sinh viên tiếp tục khảo sát:

- FedAvg;
- FedProx;
- FedBN;
- FedNova;
- SCAFFOLD;
- các hướng communication-efficient;
- imbalance-aware FL;
- personalized FL.

Mỗi phương pháp cần được mô tả:

| Field | Nội dung |
|---|---|
| Problem | Giải quyết vấn đề gì? |
| Main idea | Ý tưởng |
| Advantage | Ưu điểm |
| Limitation | Hạn chế |
| Computation | Computational cost |
| Communication | Communication cost |
| Non-IID | Khả năng xử lý |
| System heterogeneity | Khả năng xử lý |
| Medical FL | Phù hợp? |
| Jetson | Phù hợp? |

---

# 11. Phase 2 — Chọn Dataset

## Nguyên tắc

Chỉ dùng:

> **2D lightweight medical images**

Không dùng 3D trong core scope.

## Primary Dataset

### BloodMNIST

- 8 classes;
- 28×28 RGB;
- lightweight;
- phù hợp Jetson Nano;
- phù hợp Dirichlet non-IID;
- dễ phân tích minority classes và confusion matrix.

## Secondary Dataset

### DermaMNIST

Dùng để kiểm tra generalization sang domain medical image khác.

## Optional

- PathMNIST;
- PneumoniaMNIST;
- OrganAMNIST.

Không cần chạy full experimental matrix cho dataset optional.

---

# 12. Dataset không ưu tiên

Không dùng làm dataset chính:

- ChestX-ray14;
- CheXpert;
- MIMIC-CXR;
- BraTS;
- ISIC 2019;
- HAM10000.

Lý do:

- dataset lớn;
- training/preprocessing nặng hơn;
- không cần thiết cho mục tiêu lightweight edge FL;
- làm tăng rủi ro scope trong 15 tuần.

Proposal hiện tại cũng định hướng dùng MedMNIST và loại các dataset clinical lớn khỏi core scope.

---

# 13. Literature — Dataset và FL

## 13.1. Dataset

**Yang et al., 2023 — MedMNIST v2**

> “MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification”

Dùng làm reference chính cho:

- dataset;
- benchmark;
- centralized classification.

## 13.2. FL Foundation

- McMahan et al. — FedAvg.
- Yang et al. — Federated Machine Learning.

## 13.3. Non-IID / Heterogeneity

- FedProx.
- SCAFFOLD.
- FedNova.
- Survey về non-IID FL.
- FedCiR.

## 13.4. Medical FL

- Systematic review về FL trong medical image analysis.
- Benchmark về federated medical image classification.
- Class-imbalanced medical FL.

## 13.5. Communication

- Bandwidth-efficient FL.
- Compression.
- Federated distillation.

## 13.6. Edge / Heterogeneous FL

Tập trung vào:

- resource-constrained clients;
- heterogeneous hardware;
- stragglers;
- communication;
- energy.

---

# 14. Phase 3 — Centralized Baseline

Trước FL phải xây dựng centralized model.

```text
BloodMNIST
    ↓
Train / Validation / Test
    ↓
Lightweight CNN
    ↓
Centralized Training
    ↓
Accuracy / Precision / Recall / F1
```

Model khuyến nghị:

```text
Input 28×28×3
       ↓
Conv Block
       ↓
Conv Block
       ↓
Pooling
       ↓
Global Average Pooling
       ↓
Linear
       ↓
8 classes
```

Không dùng model quá lớn vì Nano là resource-constrained device.

---

# 15. Phase 4 — Non-IID Partition

Dùng Dirichlet distribution.

```text
BloodMNIST
     ↓
Dirichlet Partition
     ↓
Client 1 ... Client N
```

### α = 1.0

Dirichlet α=1.0; vẫn là non-IID, đọc histogram thực tế để xác định mức lệch.

### α = 0.3

Moderately non-IID.

### α = 0.1

Highly non-IID.

Proposal hiện tại đã yêu cầu các mức α này.

## Reproducibility

Phải lưu:

```text
dataset
alpha
random seed
client IDs
sample IDs
class distribution
```

Ví dụ:

```text
partition_seed_42_alpha_0.1.json
```

---

# 16. Phase 5 — FedAvg trên PC

Trước Jetson:

```text
PC
|
+-- Client 1
+-- Client 2
+-- Client 3
+-- ...
+-- Client N
```

Mỗi round:

```text
1. Server sends global model
2. Client receives model
3. Local training
4. Client sends update
5. Server aggregates
6. New global model
```

So sánh:

```text
Centralized
    vs
FedAvg
```

với:

- α = 1.0;
- α = 0.3;
- α = 0.1.

---

# 17. Phase 6 — Real FL trên Jetson Orin

Đầu tiên:

```text
PC Server
   |
   +-- Orin 1
   +-- Orin 2
   +-- Orin 3
```

Sau khi ổn định:

```text
PC Server
   |
   +-- Orin 1
   +-- Orin 2
   +-- Orin 3
   +-- Orin 4
   +-- Orin 5
```

Đo:

- local training time;
- communication time;
- round latency;
- CPU;
- GPU;
- RAM;
- GPU memory;
- temperature.

---

# 18. Phase 7 — Real FL trên Jetson Nano

Sau Orin:

```text
PC Server
   |
   +-- Nano 1
   +-- Nano 2
   +-- Nano 3
   +-- Nano 4
   +-- Nano 5
```

Giữ model/algorithm/protocol giống nhau nếu có thể.

Mục tiêu:

> Đo ảnh hưởng của resource-constrained clients.

---

# 19. Phase 8 — Heterogeneous FL: Orin + Nano

Experiment quan trọng:

```text
                 FL Server
                     |
       +-------------+-------------+
       |                           |
   5 × Orin                    5 × Nano
```

Cùng:

- dataset;
- model;
- FL algorithm;
- task.

Khác:

- compute;
- memory;
- training speed;
- power;
- energy.

---

# 20. Statistical vs System Heterogeneity

Phải tách hai yếu tố.

## Statistical heterogeneity

```text
Dirichlet α
    ↓
Different class distributions
    ↓
Client drift
```

## System heterogeneity

```text
Orin vs Nano
    ↓
Different compute/memory/speed
    ↓
Different local training time
```

Mục tiêu là xác định:

> Performance degradation đến từ **data** hay **hardware/resource**?

---

# 21. Experimental Control

Không thay đổi quá nhiều biến cùng lúc.

## Experiment A — Hardware effect

Giữ:

- data volume;
- data distribution;
- model;
- batch size;
- local epochs;
- algorithm.

Chỉ thay đổi:

> Orin vs Nano.

## Experiment B — Statistical heterogeneity

Giữ hardware:

> 5 Orin

Thay đổi:

> α = 1.0 / 0.3 / 0.1.

## Experiment C — Realistic heterogeneous FL

Thay đổi cả:

- data distribution;
- hardware.

---

# 22. Straggler Effect

Ví dụ:

```text
Orin:
12–14 seconds / round

Nano:
38–43 seconds / round
```

Trong synchronous FL, server có thể phải chờ client chậm nhất.

Đo:

- slowest client time;
- median client time;
- waiting time;
- straggler overhead.

Ví dụ:

```text
Straggler overhead
= slowest client time
  − typical fast-client time
```

Không giả định trước Nano chắc chắn là bottleneck; phải đo thực nghiệm.

---

# 23. Phase 9 — Proposed FL Method

Sau khi FedAvg baseline ổn định, chọn **một** phương pháp.

Ứng viên:

- FedProx;
- FedBN;
- FedNova;
- SCAFFOLD.

### Khuyến nghị ban đầu

**FedProx** nếu research gap tập trung vào client/system heterogeneity.

**FedBN** nếu research gap tập trung vào feature/data heterogeneity.

Không cần triển khai cả bốn.

Quyết định cuối cùng phải xuất phát từ literature review.

---

# 24. Phase 10 — Resource-aware Extension

Chỉ thực hiện sau khi đã xác định bottleneck.

Có thể nghiên cứu:

- adaptive local epochs;
- client selection;
- resource-aware scheduling;
- weighted participation.

Pipeline:

```text
FedAvg
   ↓
Measure heterogeneity
   ↓
Identify bottleneck
   ↓
Proposed improvement
   ↓
Resource-aware extension
```

Không làm resource-aware mechanism ngay từ đầu.

---

# 25. Metrics

## Model

- Accuracy;
- Precision;
- Recall;
- F1-score.

## FL

- communication rounds;
- communication bytes;
- rounds to target performance;
- training time;
- convergence speed.

## Client fairness

- mean client accuracy;
- variance;
- worst-client accuracy;
- per-client F1.

## Edge/resource

- CPU utilization;
- GPU utilization;
- RAM;
- GPU memory;
- temperature;
- local training time;
- upload/download time;
- round latency;
- power;
- energy/round;
- total energy.

---

# 26. Ablation Study

### A — Ba mức Dirichlet non-IID (phạm vi hiện tại)

```text
α = 1.0 / 0.3 / 0.1
```

### B — Algorithm

```text
FedAvg vs Proposed
```

### C — Number of clients

```text
3 vs 5 vs 10
```

### D — Hardware

```text
5 Orin
vs
5 Nano
vs
5 Orin + 5 Nano
```

### E — Optional

```text
Standard FL
vs
Resource-aware FL
```

---

# 27. Error Analysis

Phải phân tích:

- confusion matrix;
- per-class accuracy;
- minority classes;
- rare classes;
- class imbalance;
- client-specific performance;
- failure cases.

Câu hỏi chính:

> FedAvg thất bại ở đâu?

> Proposed method cải thiện ở đâu?

---

# 28. Dashboard

Prototype cuối kỳ:

```text
+--------------------------------------------+
|              FedMedAI Dashboard            |
+--------------------------------------------+
| Round: 25 / 50                             |
| Global Accuracy: XX.X%                     |
| Global F1:       XX.X%                     |
+--------------------------------------------+
| Client | Device | Loss | Train Time        |
| O1     | Orin   | ...  | ... sec           |
| O2     | Orin   | ...  | ... sec           |
| ...                                        |
| N1     | Nano   | ...  | ... sec           |
| ...                                        |
+--------------------------------------------+
| Communication: XXX MB                      |
| Energy:        XXX Wh                      |
+--------------------------------------------+
```

Dashboard cần thể hiện:

1. global round;
2. client status;
3. local training;
4. client latency;
5. model metrics;
6. communication;
7. resource/energy.

---

# 29. TensorRT

Không đưa TensorRT vào training pipeline chính.

Training:

```text
PyTorch + Flower
```

Optional deployment:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
   ↓
Jetson inference
```

TensorRT chỉ là optional optimization.

---

# 30. Software Architecture

```text
FedMedAI/
│
├── server/
│   ├── server.py
│   └── strategy/
│
├── client/
│   ├── client.py
│   ├── train.py
│   ├── evaluate.py
│   └── system_monitor.py
│
├── datasets/
│   ├── medmnist.py
│   └── partition.py
│
├── models/
│   └── cnn.py
│
├── algorithms/
│   ├── fedavg.py
│   └── proposed.py
│
├── experiments/
│   ├── non_iid_study/       # Artifacts hiện hành, không có nghiên cứu IID
│   ├── noniid_01/
│   ├── noniid_03/
│   └── noniid_10/
│
├── monitoring/
│   ├── metrics.py
│   ├── resource.py
│   ├── power.py
│   └── dashboard.py
│
└── configs/
    ├── experiment.yaml
    └── jetson.yaml
```

---

# 31. Phân công công việc

## SV1 — FL Architecture & Infrastructure

### Literature

- FL fundamentals;
- centralized vs decentralized;
- cross-silo vs cross-device.

### Implementation

- Flower;
- server;
- client;
- communication;
- experiment orchestration.

---

## SV2 — FL Algorithms

### Literature

- FedAvg;
- FedProx;
- FedBN;
- FedNova;
- SCAFFOLD;
- non-IID.

### Implementation

- FedAvg;
- selected proposed method;
- algorithm ablation.

---

## SV3 — Medical Dataset & Evaluation

### Literature

- medical FL;
- MedMNIST;
- class imbalance;
- fairness.

### Implementation

- BloodMNIST;
- preprocessing;
- Dirichlet partition;
- metrics;
- confusion matrix;
- error analysis.

---

## SV4 — Edge / Jetson

### Literature

- Edge FL;
- resource heterogeneity;
- communication;
- energy;
- straggler.

### Implementation

- 5 Orin;
- 5 Nano;
- resource monitoring;
- latency;
- power/energy;
- heterogeneous FL;
- dashboard.

### Lưu ý

Tất cả cùng xây dựng:

> **Một FedMedAI system + một research story thống nhất.**

---

# 32. Kế hoạch 15 tuần

| Tuần | Nội dung | Deliverable |
|---:|---|---|
| 1 | FL fundamentals + taxonomy | FL survey draft |
| 2 | Architecture/setting/algorithm survey | Comparison matrix |
| 3 | Medical FL + Edge FL + research gap | Architecture decision |
| 4 | Dataset + centralized model | Baseline |
| 5 | Dirichlet partition + reproducibility | Non-IID pipeline |
| 6 | Flower + FedAvg trên PC | FL baseline |
| 7 | α=1.0/0.3/0.1 | Non-IID results |
| 8 | 3 Orin | First real FL |
| 9 | 5 Orin | Orin benchmark |
| 10 | 5 Nano | Nano benchmark |
| 11 | 5 Orin + 5 Nano | Heterogeneous FL |
| 12 | Proposed FL method | Improved FL |
| 13 | Ablation + resource analysis | Full benchmark |
| 14 | Dashboard + error analysis | Prototype |
| 15 | Thesis + demo + defense | Final deliverables |

---

# 33. Decision Gate theo tuần

## Gate 1 — cuối tuần 2

Sinh viên phải trả lời:

- FL là gì?
- Có những architecture nào?
- Centralized vs decentralized?
- Cross-silo vs cross-device?

## Gate 2 — cuối tuần 3

Phải quyết định:

- architecture;
- dataset;
- baseline;
- candidate algorithms;
- research gap;
- experimental protocol.

## Gate 3 — cuối tuần 6

Phải có:

> Centralized + FedAvg trên PC.

## Gate 4 — cuối tuần 11

Phải có:

> Real FL trên Orin + Nano.

## Gate 5 — cuối tuần 13

Phải có:

> Proposed method + ablation + resource analysis.

---

# 34. Expected Contribution

Contribution dự kiến:

1. **Comprehensive FL study** cho bài toán medical/edge AI.
2. **Justified selection of Server–Client FL**.
3. Lightweight 2D medical image FL benchmark.
4. FedAvg baseline.
5. Một FL improvement được lựa chọn từ literature.
6. Statistical heterogeneity evaluation.
7. Real deployment trên 5 Jetson Orin + 5 Jetson Nano.
8. System/resource heterogeneity analysis.
9. Communication and energy evaluation.
10. Reproducible FL prototype.

---

# 35. Final Research Story

```text
                 Federated Learning Survey
                           |
                           v
                  FL Taxonomy & Review
                           |
                           v
             Architecture Comparison
                           |
                           v
                Server–Client FL
                           |
                           v
                    BloodMNIST
                           |
                           v
              Centralized Baseline
                           |
                           v
                       FedAvg
                           |
                           v
                 Dirichlet Non-IID
                           |
                 +---------+---------+
                 |                   |
                 v                   v
              5 Orin             5 Nano
                 |                   |
                 +---------+---------+
                           |
                           v
                    Orin + Nano
                 System Heterogeneity
                           |
                           v
                   Proposed FL
                           |
                           v
        Accuracy / F1 / Fairness / Convergence
        Communication / Latency / Resource / Energy
                           |
                           v
                Ablation + Error Analysis
                           |
                           v
                    FedMedAI Prototype
```

---

# 36. One-line Project Definition

> **FedMedAI is a lightweight 2D medical image classification study in which students comprehensively survey Federated Learning, justify the selection of a centralized server–client architecture, and investigate FL under statistical and system/resource heterogeneity using 5 NVIDIA Jetson Orin and 5 Jetson Nano edge clients.**

---

# 37. Guiding Principle

> **Students must first understand the FL landscape, then choose the architecture, then choose the method, and only afterward implement and evaluate it.**

Không làm:

> **FedAvg + Flower → code → demo.**

Mà làm:

> **Survey FL → taxonomy → compare → justify Server–Client → select algorithm → select dataset → centralized baseline → FedAvg → non-IID → Jetson Orin/Nano → proposed method → evaluation → analysis → prototype.**
