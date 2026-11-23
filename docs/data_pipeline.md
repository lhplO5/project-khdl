# Kiến trúc Pipeline Dữ liệu & Báo cáo Tiền xử lý (Tuần 1 & Tuần 2)

## 1. Kiến trúc tổng thể
Pipeline được tách thành các module độc lập trong `src/`, cấu hình tập trung qua `configs/pipeline.yaml` để đảm bảo tính tái lập (*reproducibility*) khi chuyển đổi giữa dữ liệu mô phỏng và dữ liệu Kaggle thật.

```text
Kaggle API (src/ingest.py)
   └─▶ data/raw/
         ├── data_global/res_train4_test8.npz
         └── data_stats/{data_stats.npz, mask2d.npy}
                 │
                 ├─▶ [Tuần 1] Kiểm toán chất lượng (src/profiling.py)
                 │        └─▶ data/interim/audit_*.json
                 │
                 └─▶ [Tuần 2] Chuẩn hóa Z-score (src/dataset.py)
                          └─▶ Làm sạch NaN/Inf & Winsorizing (src/clean.py)
                                   └─▶ Tạo đặc trưng 144 -> 7 (src/features.py)
                                            └─▶ data/processed/one_step_*.npz
```

---

## 2. Đặc tả chi tiết từng Module

| Module | Vai trò chính | Hàm cốt lõi |
| :--- | :--- | :--- |
| `src/ingest.py` | Tải tự động dữ liệu từ Kaggle CLI và giải nén vào `data/raw/`. | `download_kaggle_competition()` |
| `src/dataset.py` | Đọc `pipeline.yaml`, nạp `.npz`/`.npy` ở `float32`, chuẩn hóa/khôi phục Z-score và ánh xạ lưới `(360, 720)`. | `load_ml_ready_npz()`, `load_stats()`, `normalize_array()`, `denormalize_array()`, `to_global_grid()` |
| `src/profiling.py` | Kiểm toán shape, RAM (MB), số lượng NaN/Inf, min/max/mean/std và bảng phân vị `p50/p95` cho 7 biến `TREE_BAND`. | `audit_array()`, `audit_targets_summary()`, `save_audit_report()` |
| `src/clean.py` | Xử lý giá trị không hữu hạn (`NaN`, `±Inf`) và cắt ngưỡng ngoại lai cực đoan trong không gian chuẩn hóa. | `sanitize_non_finite()`, `clip_normalized_outliers()`, `clean_carbonglobe_arrays()` |
| `src/features.py` | Chuyển đổi mảng 4D/5D sang bảng đặc trưng 2D one-step-ahead (`144 -> 7`). | `build_one_step_dataset()` |
| `src/run_sprint1_2.py` | CLI điều phối toàn bộ luồng Tuần 1 + Tuần 2, hỗ trợ cờ `--synthetic` để kiểm chứng nhanh. | `create_synthetic_carbonglobe_files()`, `main()` |

---

## 3. Cơ sở toán học & Quy tắc làm sạch áp dụng

### 3.1. Chuẩn hóa Z-score an toàn
Với mọi kênh $j$, chuẩn hóa bằng thống kê toàn cục từ `data_stats.npz`:
$$z_j = \frac{x_j - \mu_j}{\max(\sigma_j,\, \varepsilon)}, \quad \varepsilon = 10^{-6}$$
Việc dùng `eps_std = 1e-6` giúp ngăn lỗi chia cho `0` ở các kênh có phương sai cực nhỏ (kênh hằng số hoặc gần hằng số).

### 3.2. Xử lý giá trị khuyết (`NaN`) và vô cực (`±Inf`)
* Thực hiện sau bước chuẩn hóa Z-score (hoặc thay bằng trung bình từng kênh). Trong không gian Z-score, giá trị `0.0` tương ứng đúng với giá trị trung bình toàn cục $\mu_j$, giúp bảo toàn kỳ vọng và không làm lệch phân phối khi huấn luyện cây quyết định hay mạng nơ-ron.

### 3.3. Kiểm soát ngoại lai (Winsorizing trong không gian chuẩn hóa)
* Cắt ngưỡng mọi giá trị $z$ nằm ngoài đoạn `[-10.0, 10.0]` (tức cách trung bình hơn $10\sigma$).
* **Lý do chọn ngưỡng $\pm 10\sigma$ thay vì IQR $1.5\times$:** Dữ liệu sinh thái toàn cầu có độ lệch tự nhiên rất lớn giữa vùng sa mạc/lãnh nguyên và rừng mưa nhiệt đới (sinh khối `agb` có đuôi phải dài). Nếu cắt chặt theo $1.5\times\text{IQR}$ sẽ làm mất tín hiệu sinh thái thật của rừng già; ngưỡng $\pm 10\sigma$ chỉ loại bỏ các lỗi số học/ngoại lai cực đoan (như điểm dị biệt `5000.0`).

### 3.4. Thiết kế vector đặc trưng 144 chiều (`src/features.py`)
1. **Trung bình năm của đầu vào (`136` chiều):** Lấy trung bình theo trục tháng (`axis=2`) từ `(S, 40, 12, 136)` $\rightarrow$ `(S, 40, 136)`.
2. **Trạng thái mục tiêu tháng 12 năm trước (`7` chiều):** Trích `y_norm[:, :, :-1, 11, :]` làm trạng thái khởi tạo $\mathbf{v}_{t-1}$.
3. **Tuổi rừng chuẩn hóa (`1` chiều):** Chia mức tuổi cho `500.0` để đưa về đoạn `(0, 1]`.
4. **Đầu ra mục tiêu (`7` chiều):** Trích `y_norm[:, :, 1:, 11, :]` (tháng 12 của năm $t$).

---

## 4. Kết quả kiểm thử & Vận hành

### 4.1. Kết quả kiểm thử tự động (`tests/test_pipeline.py`)
* Kiểm tra đọc/ghi `.npz`, phát hiện chính xác NaN và ngoại lai cài sẵn.
* Sau khi chạy `clean_carbonglobe_arrays`: `0` giá trị NaN/Inf còn lại, toàn bộ giá trị nằm trong `[-10.0, 10.0]`.
* Kích thước đầu ra `build_one_step_dataset` khớp chính xác $(15 \times S \times 40, 144)$ và $(15 \times S \times 40, 7)$.

### 4.2. Hướng dẫn chạy CLI
```powershell
# 1. Tải dữ liệu thật từ Kaggle (cần ~/.kaggle/kaggle.json)
python -m src.ingest

# 2. Chạy pipeline Tuần 1 & Tuần 2 trên dữ liệu thật
python -m src.run_sprint1_2

# 3. Hoặc chạy kiểm chứng end-to-end bằng dữ liệu mô phỏng chuẩn CarbonGlobe
python -m src.run_sprint1_2 --synthetic
```
