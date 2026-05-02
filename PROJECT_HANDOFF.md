# Opstimus Project Handoff

## 1. Mục đích dự án

### Định hướng hiện tại
- Dự án đang đi theo hướng **ứng dụng**.
- Mục tiêu là xây một **framework anomaly detection + root cause analysis (RCA)** có thể mở rộng cho nhiều dataset khác nhau, thay vì chỉ làm một notebook đơn lẻ cho một bộ dữ liệu.

### Bài toán
- Đầu vào:
  - dữ liệu bảng (`tabular`)
  - dữ liệu chuỗi thời gian đa biến (`multivariate time series`)
- Đầu ra:
  - anomaly score
  - dự đoán bất thường
  - metric đánh giá detection
  - RCA toàn cục
  - RCA theo segment/sự kiện nếu dataset hỗ trợ

### Mục tiêu thesis phù hợp
- Chứng minh hệ thống có thể:
  - chạy được trên nhiều dataset
  - hỗ trợ nhiều detector
  - tái lập bằng config
  - sinh artifact rõ ràng
  - có dashboard để demo kết quả

## 2. Trạng thái hiện tại

### Commit hiện tại
- `468e3a2`

### Nhánh
- `master` và `main` đã được đồng bộ ở các bước trước
- Remote hiện dùng:
  - `https://github.com/haiduongnguyen/Opstimus.git`

### Hướng đã chốt
- Trục chính: **framework ứng dụng**
- Nhánh nghiên cứu nhỏ sẽ phát triển sau, nhiều khả năng xoay quanh:
  - RCA cho time series
  - event-level evaluation
  - tradeoff detection vs RCA theo threshold

## 3. Cấu trúc thư mục hiện tại

### Root
- `main.py`
  - chạy 1 config đơn lẻ
- `run_batch.py`
  - chạy nhiều config và sinh leaderboard
- `run_dashboard_demo.bat`
  - mở dashboard local nhanh trên Windows
- `README.md`
  - hướng dẫn chạy chính
- `PROJECT_HANDOFF.md`
  - file handoff này

### `config/`
- Chứa schema config chuẩn hóa và các config theo dataset
- File chính:
  - `config/loader.py`
  - `config/credit_card/isolation_forest.json`
  - `config/sklearn_breast_cancer/isolation_forest.json`
  - `config/smd/machine_1_1/isolation_forest.json`
  - `config/smd/machine_1_1/isolation_forest_percentile_97.json`

### `datasets/`
- Adapter theo dataset
- File chính:
  - `base.py`
  - `registry.py`
  - `credit_card.py`
  - `smd.py`
  - `sklearn_breast_cancer.py`

### `preprocessing/`
- Tiền xử lý dùng chung
- Hiện chủ yếu là:
  - `scaling.py`

### `detection/`
- Detector abstraction + baseline detector
- File chính:
  - `detector_base.py`
  - `isolation_forest.py`
  - `lof.py`
  - `autoencoder.py`

### `thresholding/`
- Module threshold độc lập
- File chính:
  - `strategies.py`

### `evaluation/`
- Metric và reporting
- File chính:
  - `metrics.py`
  - `reporting.py`

### `rca/`
- Root cause analysis
- File chính:
  - `feature_ranking.py`

### `pipelines/`
- Orchestration
- File chính:
  - `runner.py`
  - `batch.py`

### `visualization/`
- Dashboard local
- File chính:
  - `dashboard.py`

### `artifacts/`
- Kết quả đầu ra của pipeline
- Hiện có:
  - `artifacts/sklearn_breast_cancer/...`
  - `artifacts/smd/machine-1-1/...`
  - `artifacts/batch_runs/...`
  - `artifacts/batch_runs_smd/...`

### `report_thesis/`
- Chứa bản thảo Word-compatible `.rtf`
- Đã được add vào `.gitignore`

## 4. Workflow chạy hiện tại

### Chạy 1 config
```bash
venv_opstimus\Scripts\python.exe main.py --config config/smd/machine_1_1/isolation_forest.json
```

### Chạy batch nhiều config
```bash
venv_opstimus\Scripts\python.exe run_batch.py --config-root config --output-dir artifacts/batch_runs
```

### Mở dashboard local
```bash
run_dashboard_demo.bat
```

Hoặc:
```bash
venv_opstimus\Scripts\python.exe visualization\dashboard.py --port 8765
```

### URL dashboard
- Run chi tiết:
  - `http://127.0.0.1:8765/?run=smd/machine-1-1/isolation_forest`
  - `http://127.0.0.1:8765/?run=smd/machine-1-1/isolation_forest_percentile_97`
  - `http://127.0.0.1:8765/?run=sklearn_breast_cancer/isolation_forest`

## 5. Schema config hiện tại

Mỗi config hiện hỗ trợ các section sau:

```json
{
  "experiment": {
    "name": "...",
    "task_type": "...",
    "tags": []
  },
  "dataset": {
    "name": "..."
  },
  "preprocessing": {
    "scaler": "standard"
  },
  "detector": {
    "name": "...",
    "params": {}
  },
  "threshold": {
    "strategy": "model_default"
  },
  "rca": {
    "top_k": 5
  },
  "output_dir": "..."
}
```

## 6. Các dataset đang hỗ trợ

### 6.1. `credit_card`
- Loại: `tabular`
- File config:
  - `config/credit_card/isolation_forest.json`
- Trạng thái:
  - adapter đã có
  - config đã có
  - hiện **không chạy được trên máy local** nếu thiếu file:
    - `data/raw/creditcard/creditcard.csv`

### 6.2. `smd`
- Loại: `time_series`
- File config:
  - `config/smd/machine_1_1/isolation_forest.json`
  - `config/smd/machine_1_1/isolation_forest_percentile_97.json`
- Trạng thái:
  - chạy được
  - hỗ trợ interpretation label
  - có RCA theo segment/event

### 6.3. `sklearn_breast_cancer`
- Loại: `tabular`
- File config:
  - `config/sklearn_breast_cancer/isolation_forest.json`
- Trạng thái:
  - **self-contained**
  - không cần file ngoài repo
  - rất hữu ích để smoke test framework

## 7. Detector hiện có

### 7.1. Isolation Forest
- Trạng thái: dùng ổn nhất hiện nay
- Dùng cho:
  - credit card
  - SMD
  - sklearn breast cancer

### 7.2. LOF
- Đã chỉnh `novelty=True`
- Có thể dùng cho dữ liệu test mới
- Chưa được mở rộng config nhiều như Isolation Forest

### 7.3. Autoencoder
- Đã có baseline implementation
- Chưa phải hướng chính hiện tại
- Vẫn cần hoàn thiện thêm nếu muốn đưa vào batch benchmark nghiêm túc

## 8. Thresholding hiện có

Module:
- `thresholding/strategies.py`

### Hỗ trợ
- `model_default`
  - dùng `detector.predict()` native
- `percentile`
- `stddev`
- `value`

### Ý nghĩa hiện tại
- Có thể benchmark tradeoff giữa:
  - precision / recall / f1
  - RCA hit@k
- Ví dụ SMD:
  - `model_default`: recall cao hơn, RCA hit@5 tốt hơn
  - `percentile_97`: precision cao hơn, recall thấp hơn, RCA hit@5 thấp hơn

## 9. RCA hiện có

Module:
- `rca/feature_ranking.py`

### Hỗ trợ
- Global ranking
  - top-k feature/channel bất thường toàn cục
- Segment-level RCA
  - gom anomaly liên tiếp thành segment
- Event matching
  - đối chiếu với `interpretation_label` của SMD
- RCA metrics
  - ví dụ `hit_at_5`

### Lưu ý
- Đây vẫn là **baseline contribution-based RCA**
- Chưa phải causal RCA
- Nhưng đủ tốt để demo và làm phần ứng dụng/thực nghiệm

## 10. Evaluation hiện có

### Detection metrics
- precision
- recall
- f1
- confusion matrix
- classification report
- roc_auc
- pr_auc

### RCA metrics
- hiện chủ yếu có:
  - `hit_at_5`
  - `matched_events`
  - `num_interpretation_events`

### Điểm còn thiếu
- event-level detection metrics cho time series
- detection delay
- event precision / recall / F1

## 11. Dashboard hiện có

Module:
- `visualization/dashboard.py`

### Hiển thị được
- metric cards
- anomaly score trend
- prediction vs ground truth
- global RCA
- segment-level RCA
- event-level RCA matching
- batch leaderboard comparison

### Mục đích
- Dùng để demo luận văn
- Dùng để xem nhanh nhiều run mà không phải mở từng file CSV/JSON

## 12. Batch runner hiện có

Module:
- `pipelines/batch.py`
- `run_batch.py`

### Hỗ trợ
- chạy cả thư mục config
- không chết cả lô nếu một config fail
- sinh:
  - `leaderboard.csv`
  - `batch_summary.json`

### Tình trạng thực tế
- Batch tổng thể hiện có thể chạy ra:
  - SMD model default
  - SMD percentile 97
  - sklearn breast cancer
- credit card sẽ fail nếu thiếu raw file local

## 13. Các feature đã hoàn thành

### Nền tảng pipeline
- [x] config-driven pipeline
- [x] dataset registry
- [x] config loader chuẩn hóa
- [x] artifact output chuẩn

### Detection
- [x] Isolation Forest
- [x] LOF
- [x] Autoencoder baseline

### RCA
- [x] feature/channel ranking
- [x] segment-level RCA
- [x] event matching cho SMD
- [x] RCA metric `hit_at_5`

### Extensibility
- [x] layout config theo dataset
- [x] batch runner
- [x] threshold module
- [x] thêm dataset tabular self-contained

### Demo / UI
- [x] local dashboard
- [x] leaderboard comparison trong dashboard

### Thesis support
- [x] `report_thesis/` với file `.rtf`
- [x] `report_thesis/` đã ignore khỏi git

## 14. Các feature đang thiếu hoặc nên làm tiếp

### Quan trọng nhất nếu đi theo hướng ứng dụng
1. Event-level evaluation cho time series
2. Thêm 1 dataset time-series nữa
3. Nâng dashboard filter/sort/best-run view
4. Dataset health checks
5. Auto-export figure/report cho luận văn

### Nếu muốn tăng chất nghiên cứu
1. RCA cho time series tốt hơn contribution-based
2. So sánh detector dưới góc nhìn RCA chứ không chỉ detection
3. Nghiên cứu threshold strategy ảnh hưởng thế nào tới RCA

## 15. Các điểm cần chú ý

### 15.1. Credit card dataset
- Config có sẵn nhưng local file có thể không còn
- Nếu muốn chạy lại credit card, cần chuẩn bị:
  - `data/raw/creditcard/creditcard.csv`

### 15.2. Autoencoder
- Chưa phải implementation production-grade
- Dùng được như baseline, nhưng chưa nên coi là trục chính của framework

### 15.3. Dashboard
- Route chi tiết theo `?run=...` là cách ổn định nhất để mở demo

### 15.4. Git
- Nhiều bước trước đã rewrite history để bỏ data/model lớn khỏi repo
- `.gitignore` hiện đã chặn:
  - raw datasets lớn
  - checkpoint models
  - artifacts
  - report_thesis

### 15.5. Artifacts
- Dashboard và batch comparison phụ thuộc trực tiếp vào artifact trong `artifacts/`
- Nếu muốn demo, nên chạy lại các config cần thiết trước

## 16. Các lệnh nên nhớ

### Chạy 1 config SMD
```bash
venv_opstimus\Scripts\python.exe main.py --config config/smd/machine_1_1/isolation_forest.json
```

### Chạy 1 config SMD với threshold khác
```bash
venv_opstimus\Scripts\python.exe main.py --config config/smd/machine_1_1/isolation_forest_percentile_97.json
```

### Chạy dataset self-contained
```bash
venv_opstimus\Scripts\python.exe main.py --config config/sklearn_breast_cancer/isolation_forest.json
```

### Chạy batch
```bash
venv_opstimus\Scripts\python.exe run_batch.py --config-root config --output-dir artifacts/batch_runs
```

### Mở dashboard
```bash
run_dashboard_demo.bat
```

## 17. Định hướng hợp lý cho chat mới

Nếu sang chat mới, nên mở đầu bằng một trong các hướng sau:

### Hướng ứng dụng
- "Tiếp tục hoàn thiện framework ứng dụng cho nhiều dataset hơn"

### Hướng thực nghiệm
- "Thêm event-level evaluation cho time series"

### Hướng demo / luận văn
- "Nâng dashboard và export report/figure cho luận văn"

### Hướng dataset
- "Thêm một dataset time-series mới vào registry, config, batch và dashboard"

## 18. Kết luận ngắn

Trạng thái hiện tại của dự án đã vượt qua mức notebook research prototype và đang ở mức:
- có kiến trúc framework
- có khả năng mở rộng dataset
- có batch evaluation
- có dashboard demo
- có RCA baseline chạy được

Điểm còn thiếu lớn nhất để tăng độ chín theo hướng ứng dụng là:
- event-level evaluation cho time series
- thêm dataset time-series mới
- cải thiện dashboard/leaderboard để phục vụ demo và báo cáo tốt hơn
