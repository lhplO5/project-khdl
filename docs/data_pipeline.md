# Kiến trúc Pipeline Dữ liệu (Sprint 1)

## 1. Luồng xử lý
```text
Kaggle API (src/ingest.py)
   └─▶ data/raw/
         └─▶ Kiểm toán chất lượng (src/profiling.py) ──▶ data/interim/audit_*.json
         └─▶ Làm sạch & tối ưu bộ nhớ (src/clean.py) ──▶ data/processed/clean_*.parquet
```

## 2. Quy tắc làm sạch áp dụng
1. **Chuẩn hóa định danh cột:** Chuyển toàn bộ tên cột về `snake_case` chữ thường.
2. **Loại bỏ nhiễu cấu trúc:** Xóa bản ghi trùng lặp, xóa cột hằng số (chỉ có 1 giá trị) hoặc cột khuyết $>50\%$ dữ liệu (trừ các cột khóa bảo vệ như `site_id`, `year`).
3. **Điền khuyết theo chuỗi thời gian:** Sắp xếp theo `(site_id, year)` và nội suy tuyến tính theo từng điểm lưới; giá trị còn thiếu ở biên được lấp bằng median toàn cột.
4. **Kiểm soát ngoại lai (Winsorizing):** Cắt ngưỡng theo phân vị `[0.5%, 99.5%]` để tránh các điểm dị biệt cực đoan làm lệch mô hình hồi quy.
5. **Tối ưu RAM:** Ép kiểu `float64/int64` xuống `float32/int32` và lưu trữ ở định dạng cột Parquet.

## 3. Cách chạy kiểm thử
```powershell
# Chạy unit tests
python -m pytest tests/ -v

# Chạy thử toàn bộ pipeline với dữ liệu mô phỏng chuẩn CarbonGlobe (136 biến, 7 mục tiêu)
python -m src.run_sprint1 --synthetic
```
