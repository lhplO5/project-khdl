# Báo cáo tiến độ Tuần 1

- **Chủ đề:** Dự báo chu trình carbon & hệ sinh thái đất toàn cầu (CarbonGlobe — IEEE BigData Cup 2026).
- **Mục tiêu tuần:** Thiết lập nền tảng kỹ thuật và chốt phạm vi xử lý dữ liệu lớn.

## Kết quả đạt được
1. **Chốt bài toán & phương án phần cứng:**
   - Định hình bài toán hồi quy đa đầu ra theo chuỗi thời gian (136 biến đầu vào $\rightarrow$ 7 chỉ số hệ sinh thái trong 40 năm).
   - Xác định lộ trình chạy CPU cho các mô hình cây quyết định (`LightGBM`, `XGBoost`) kết hợp tối ưu kiểu `float32`; chỉ sử dụng GPU miễn phí (Kaggle/Colab) nếu mở rộng sang Deep Learning ở giai đoạn sau.
2. **Dựng khung dự án & package `carbonglobe`:**
   - Tổ chức repo tách biệt rõ: `carbonglobe/` (mã nguồn tái sử dụng), `exploration/` (notebook phân tích), `datasets/` (phân tầng `source` / `staging` / `ready` và chặn push file nặng qua `.gitignore`), `notes/`, `reports/`.
   - Hoàn thiện `carbonglobe/dataset.py` và `exploration/schema_audit.ipynb` phục vụ đo RAM, kiểm tra dữ liệu khuyết và lấy mẫu phân tầng tái lập được (`settings.yaml`).

## Kế hoạch Tuần 2
- Tải dữ liệu gốc vào `datasets/source/` và chạy `exploration/schema_audit.ipynb` để chốt danh sách tên biến vào `notes/problem_scope.md`.
- Thực hiện EDA: phân tích tương quan nhóm biến khí tượng/đất, biến động chuỗi 40 năm và xây dựng bước làm sạch dữ liệu.
