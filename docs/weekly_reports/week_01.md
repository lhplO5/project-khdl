# Báo cáo tiến độ Tuần 1

- **Đề tài:** Dự báo động lực học hệ sinh thái đất toàn cầu & chu trình carbon (bộ dữ liệu CarbonGlobe - IEEE BigData Cup 2026).
- **Thời gian:** Tuần 1

## 1. Các đầu việc đã hoàn thành
1. **Chốt phạm vi bài toán môn học:**
   - Xác định bài toán Multi-output Regression / Time-series Forecasting: dự báo 7 chỉ số hệ sinh thái từ 136 biến môi trường qua chuỗi 40 năm.
   - Đánh giá yêu cầu phần cứng: ưu tiên chạy CPU với các mô hình Tabular ML (`LightGBM`, `XGBoost`) kết hợp tối ưu bộ nhớ (`float32`) và sampling đại diện; dự phòng dùng Google Colab / Kaggle GPU miễn phí nếu thử nghiệm Deep Learning ở Tuần 3–4.
2. **Thiết kế & khởi tạo Repository chuẩn KHDL:**
   - Thiết lập cấu trúc `configs/`, `data/{raw,interim,processed}`, `docs/`, `notebooks/`, `src/`.
   - Cấu hình `.gitignore` tránh đẩy file dữ liệu lớn lên GitHub và tạo `requirements.txt`.
3. **Chuẩn bị khung code khảo sát dữ liệu (`src/data_loader.py` & `notebooks/01_data_understanding.ipynb`):**
   - Sẵn sàng hàm đọc cấu hình `config.yaml`, ép kiểu `float32` giảm 50% RAM, kiểm tra missing values và tạo tập sample đại diện lưu vào `data/interim/`.

## 2. Khó khăn & Hướng giải quyết
- **Dữ liệu số chiều lớn (136 biến) & dung lượng lớn (>54.152 điểm lưới $\times$ 40 năm):**
  - *Giải pháp:* Lưu dữ liệu dạng Parquet/float32, dùng tập sample có kiểm soát (`configs/config.yaml`) khi phát triển và trực quan hóa trên máy cá nhân.

## 3. Kế hoạch Tuần 2
- Tải đầy đủ dữ liệu từ Kaggle về `data/raw/` và cập nhật bảng mô tả chi tiết 136 biến vào `docs/01_problem_and_data.md`.
- Hoàn thành `notebooks/01_data_understanding.ipynb` và triển khai `notebooks/02_eda_visualization.ipynb` (phân tích phân phối, ma trận tương quan, xu hướng chuỗi thời gian 40 năm).
- Xây dựng pipeline làm sạch dữ liệu (`src/preprocessing.py`) xuất dữ liệu sạch ra `data/processed/`.
