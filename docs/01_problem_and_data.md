# Đặc tả bài toán & Dữ liệu (Data Understanding)

## 1. Tổng quan nguồn dữ liệu
- **Tên cuộc thi:** IEEE BigData Cup 2026: AI Emulation Challenge (dựa trên benchmark CarbonGlobe).
- **Bản chất bài toán:** Xây dựng bộ giả lập (AI Emulator) thay thế mô hình vật lý *Ecosystem Demography (ED)* có chi phí tính toán cao, dự báo chu trình carbon và hệ sinh thái đất toàn cầu.

## 2. Quy mô & Cấu trúc dữ liệu
- **Không gian:** $>54.152$ điểm lưới đất liền toàn cầu (độ phân giải $0.5^\circ$).
- **Điều kiện ban đầu:** $15$ trạng thái tuổi rừng khác nhau.
- **Thời gian:** Chuỗi dữ liệu mô phỏng $40$ năm.

## 3. Biến đầu vào ($X$ - 136 biến) & Biến mục tiêu ($Y$ - 7 biến)
*(Cập nhật bảng tên cột chi tiết sau khi tải file dữ liệu từ Kaggle và chạy `notebooks/01_data_understanding.ipynb`)*

| Nhóm biến | Số lượng | Mô tả sơ bộ |
| :--- | :---: | :--- |
| Khí tượng (Meteorological) | - | Nhiệt độ, lượng mưa, bức xạ, độ ẩm,... |
| Khí quyển & Đất | - | Nồng độ $\text{CO}_2$, tính chất lý-hóa của đất,... |
| Đặc trưng trạng thái | - | Tuổi rừng, tọa độ/vùng sinh thái,... |
| **Biến mục tiêu ($Y$)** | **7** | Trữ lượng carbon, thông lượng carbon, sinh khối trên mặt đất, chỉ số diện tích lá (LAI),... |
