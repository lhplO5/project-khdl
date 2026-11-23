# Báo cáo tiến độ Tuần 1 (Hoàn thành cả phần việc Tuần 1 & Tuần 2)

- **Chủ đề:** Dự báo chu trình carbon & hệ sinh thái đất toàn cầu (CarbonGlobe — IEEE BigData Cup 2026).
- **Trạng thái:** Đã hoàn thiện trọn vẹn khối lượng kỹ thuật của **Tuần 1 (Setup & Kiểm toán)** và **Tuần 2 (Làm sạch, EDA & Feature Engineering)** theo đúng schema `.npz`/`.npy` thực tế từ notebook hướng dẫn chính thức.

## Các đầu việc đã hoàn thành
1. **Nắm vững cấu trúc dữ liệu thực tế CarbonGlobe (`docs/problem_scope.md`):**
   - Đồng bộ toàn bộ mã nguồn theo định dạng `res_train4_test8.npz`, `data_stats.npz`, `mask2d.npy`, 7 biến mục tiêu `TREE_BAND` (`height`, `agb`, `soil`, `lai`, `gpp`, `npp`, `rh`) và 15 mức tuổi rừng `AGES`.
2. **Hoàn thiện trọn bộ Pipeline Tuần 1 & Tuần 2 (`src/`):**
   - `src/ingest.py`: Tải và giải nén tự động bộ dữ liệu Kaggle vào `data/raw/`.
   - `src/dataset.py`: Nạp `.npz`/`.npy`, chuẩn hóa/khôi phục Z-score, ánh xạ lưới 2D `(360, 720)`.
   - `src/profiling.py`: Kiểm toán shape, RAM, NaN/Inf và thống kê phân vị của 7 biến mục tiêu.
   - `src/clean.py`: Xử lý giá trị không hữu hạn và giới hạn ngoại lai chuẩn hóa (`[-10, 10]`).
   - `src/features.py`: Xây dựng tập huấn luyện one-step-ahead `144 -> 7` (136 trung bình năm + 7 trạng thái năm trước + `age/500`).
   - `src/run_sprint1_2.py`: Script CLI chạy end-to-end toàn bộ luồng.
3. **Notebook trực quan hóa & Kiểm thử tự động:**
   - Hoàn thiện & chạy ra kết quả đầy đủ 2 notebook: `notebooks/schema_audit.ipynb` và `notebooks/eda_and_cleaning.ipynb` (kèm biểu đồ phân phối, ma trận tương quan 7 biến mục tiêu và bản đồ lưới 2D).
   - Viết và chạy pass kiểm thử `tests/test_pipeline.py`.

## Kế hoạch tuần tiếp theo (Tuần 3)
- Tải dữ liệu thật `res_train4_test8.npz` và `data_stats.npz` vào `data/raw/` rồi chạy `python -m src.run_sprint1_2`.
- Huấn luyện các mô hình nền (`RandomForest`, `LightGBM`, `XGBoost`) và đánh giá theo giao thức **40-year autoregressive rollout** (`RMSE`, `MAE`, `R2`, `delta`, `cumulative`).
