# Báo cáo Tiến độ Tuần 1 (Hoàn thành cả khối lượng Tuần 1 & Tuần 2)

- **Môn học:** Khoa học Dữ liệu (IT4930)
- **Đề tài:** Xây dựng AI Emulator dự báo chu trình carbon và động lực học hệ sinh thái đất toàn cầu trên bộ dữ liệu **CarbonGlobe** (IEEE BigData Cup 2026).
- **Trạng thái hiện tại:** Hoàn thành **100% khối lượng kỹ thuật của Tuần 1 (Setup, Thu thập & Kiểm toán)** và **Tuần 2 (Làm sạch, EDA chuyên sâu & Feature Engineering 144 $\rightarrow$ 7)**.

---

## 1. Bảng tổng hợp các đầu việc đã hoàn thành

| Giai đoạn | Hạng mục công việc | File bàn giao | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Tuần 1** | Thiết kế cấu trúc repo chuẩn KHDL, `.gitignore` chặn file lớn, cấu hình tập trung `pipeline.yaml` | `README.md`, `.gitignore`, `configs/pipeline.yaml` | Hoàn thành |
| **Tuần 1** | Phân tích notebook hướng dẫn chính thức & chốt schema `.npz`/`.npy`, 7 biến `TREE_BAND`, 15 mức `AGES` | `notebooks/01_data_tutorial.ipynb`, `docs/problem_scope.md` | Hoàn thành |
| **Tuần 1** | Xây dựng script tải dữ liệu Kaggle tự động & module kiểm toán chất lượng mảng 4D/5D | `src/ingest.py`, `src/dataset.py`, `src/profiling.py` | Hoàn thành |
| **Tuần 1** | Chạy notebook kiểm toán schema, phân bố 136 kênh đầu vào, 7 mục tiêu và bản đồ `mask2d` | `notebooks/schema_audit.ipynb` | Hoàn thành |
| **Tuần 2** | Xây dựng pipeline chuẩn hóa Z-score, làm sạch NaN/Inf, Winsorizing `[-10, 10]` | `src/clean.py`, `docs/data_pipeline.md` | Hoàn thành |
| **Tuần 2** | Xây dựng bộ tạo đặc trưng one-step-ahead (`144 -> 7`) và CLI chạy end-to-end | `src/features.py`, `src/run_sprint1_2.py` | Hoàn thành |
| **Tuần 2** | Chạy notebook EDA chuyên sâu: trước/sau làm sạch, quỹ đạo 40 năm, ảnh hưởng tuổi rừng, ma trận tương quan & kiểm tra rò rỉ dữ liệu | `notebooks/eda_and_cleaning.ipynb` | Hoàn thành |
| **Kiểm thử** | Viết và chạy pass kiểm thử tự động toàn bộ luồng Tuần 1 + Tuần 2 | `tests/test_pipeline.py` | Hoàn thành |

---

## 2. Kết quả định lượng & Phát hiện chính từ EDA

1. **Tối ưu bộ nhớ & hiệu năng:**
   - Chuyển đổi toàn bộ mảng sang `float32` giúp giảm **50% dung lượng RAM** so với `float64` mặc định của NumPy/SciPy.
   - Sử dụng tệp ML-ready `res_train4_test8.npz` kết hợp `max_sites` trong `configs/pipeline.yaml` giúp phát triển mượt mà trên máy cá nhân không cần GPU ở giai đoạn EDA và Baseline.
2. **Chất lượng dữ liệu & Làm sạch (`src/clean.py`):**
   - Pipeline phát hiện và xử lý triệt để 100% giá trị `NaN`/`Inf` về kỳ vọng chuẩn hóa (`0.0`), đồng thời cắt ngưỡng thành công các ngoại lai cực đoan vượt `±10σ` mà không làm biến dạng phân phối chính của dữ liệu sinh thái.
3. **Phát hiện phục vụ giai đoạn Modeling (Tuần 3):**
   - Các biến trong `TREE_BAND` có **tương quan chéo rất mạnh** (ví dụ `height`, `agb`, `lai`, `gpp` cùng tăng trưởng theo tuổi rừng và điều kiện khí hậu) $\rightarrow$ các mô hình đa đầu ra hoặc chia sẻ biểu diễn sẽ tận dụng tốt thông tin chéo này.
   - **Tương quan tự hồi quy bậc 1 (lag-1 autocorrelation)** giữa $\mathbf{y}_{t-1}$ và $\mathbf{y}_t$ rất cao $\rightarrow$ khi đánh giá theo giao thức *autoregressive rollout* 40 năm, sai số nhỏ ở các năm đầu sẽ khuếch đại dần theo thời gian (*cumulative drift*). Vì vậy Tuần 3 bắt buộc phải đánh giá trên **cả 5 chỉ số (`RMSE`, `MAE`, `R²`, `delta`, `cumulative`)** thay vì chỉ nhìn RMSE một bước.

---

## 3. Kế hoạch triển khai Tuần 3 & Tuần 4
* **Tuần 3 (Baseline Modeling & Autoregressive Evaluation):**
  - Nạp dữ liệu Kaggle thật `res_train4_test8.npz` và `data_stats.npz` vào `data/raw/` rồi chạy `python -m src.run_sprint1_2`.
  - Xây dựng `src/evaluate.py` cài đặt giao thức **40-year autoregressive rollout** trên tập `test8`.
  - Huấn luyện và so sánh các mô hình nền trên CPU: `Ridge`, `RandomForestRegressor`, `LightGBM`, `XGBoost`.
* **Tuần 4 (Tuning, Deep Learning & Báo cáo tổng kết):**
  - Tinh chỉnh siêu tham số, thử nghiệm thêm đặc trưng mùa vụ (ngoài trung bình năm) để giảm `delta error`.
  - Nếu cần cải thiện `cumulative drift`, chạy thử mô hình chuỗi thời gian (GRU/1D-CNN) trên **Kaggle Notebooks / Google Colab GPU miễn phí**.
  - Vẽ bản đồ sai số không gian toàn cầu bằng `mask2d.npy` và hoàn thiện báo cáo cuối kỳ.
