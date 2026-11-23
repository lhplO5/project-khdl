# Báo cáo tiến độ Tuần 1 (Sprint 1)

- **Chủ đề:** Dự báo chu trình carbon & hệ sinh thái đất toàn cầu (CarbonGlobe — IEEE BigData Cup 2026).
- **Mục tiêu tuần:** Hoàn thiện trọn vẹn hạ tầng kỹ thuật và pipeline **Thu thập $\rightarrow$ Kiểm toán $\rightarrow$ Làm sạch dữ liệu**.

## Các đầu việc đã hoàn thành
1. **Chốt bài toán & chiến lược tính toán:**
   - Bài toán hồi quy đa đầu ra theo chuỗi thời gian: dự báo 7 chỉ số hệ sinh thái từ 136 biến môi trường qua chuỗi 40 năm.
   - Đánh giá phần cứng: chạy CPU cho `LightGBM`/`XGBoost` với bộ nhớ tối ưu `float32`; dự phòng Kaggle/Colab GPU miễn phí nếu mở rộng sang Deep Learning.
2. **Xây dựng trọn bộ Pipeline Kỹ thuật Dữ liệu (`src/`):**
   - `src/ingest.py`: Tích hợp Kaggle CLI tải và giải nén tự động dữ liệu về `data/raw/`.
   - `src/dataset.py`: Đọc CSV/Parquet, ép kiểu `float32/int32` giảm 50% RAM, lấy mẫu phân tầng tái lập được.
   - `src/profiling.py`: Kiểm toán tự động missing values, cột hằng số, khóa trùng lặp và ngoại lai theo IQR, xuất JSON report.
   - `src/clean.py`: Chuẩn hóa tên cột, lọc cột vô ích, nội suy chuỗi thời gian theo từng site, winsorizing ngoại lai và xuất Parquet sạch sang `data/processed/`.
   - `src/run_sprint1.py`: Script CLI chạy end-to-end toàn bộ pipeline (hỗ trợ `--synthetic` mô phỏng 136 biến + 7 mục tiêu để kiểm thử trước).
3. **Kiểm thử tự động & Tài liệu:**
   - Viết bộ test `tests/test_sprint1.py` (`pytest`) kiểm tra schema, audit và logic làm sạch.
   - Hoàn thiện tài liệu kiến trúc pipeline (`docs/data_pipeline.md`) và đặc tả biến (`docs/problem_scope.md`).

## Kế hoạch Tuần 2
- Chạy `python -m src.ingest` để nạp bộ dữ liệu CarbonGlobe thật vào `data/raw/`.
- Chạy `python -m src.run_sprint1 --input-file ...` để xuất báo cáo kiểm toán thực tế và dữ liệu sạch sang `data/processed/`.
- Thực hiện EDA trực quan hóa (tương quan 136 biến, xu hướng 40 năm, phân bố không gian).
