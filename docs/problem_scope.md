# Phạm vi bài toán & Cấu trúc biến (CarbonGlobe)

## Bối cảnh
Thay vì chạy mô hình vật lý *Ecosystem Demography (ED)* rất tốn tài nguyên tính toán, nhóm sử dụng bộ dữ liệu **CarbonGlobe** (IEEE BigData Cup 2026) để huấn luyện mô hình học máy dự báo trực tiếp trạng thái hệ sinh thái đất.

## Quy mô dữ liệu
- **Không gian:** $>54.152$ điểm lưới đất liền toàn cầu (độ phân giải $0.5^\circ$).
- **Trạng thái rừng:** $15$ mức tuổi rừng.
- **Độ dài chuỗi:** $40$ năm.

## Phân nhóm biến
*(Sẽ cập nhật tên cột cụ thể sau khi chạy `notebooks/schema_audit.ipynb`)*

- **Đầu vào (136 biến):**
  - *Khí tượng:* nhiệt độ, lượng mưa, bức xạ mặt trời, độ ẩm,...
  - *Môi trường & Đất:* nồng độ $\text{CO}_2$, các đặc tính lý-hóa của đất.
  - *Ngữ cảnh:* tọa độ điểm lưới, tuổi rừng, chỉ số năm.
- **Đầu ra (7 biến mục tiêu):** trữ lượng carbon, thông lượng carbon, sinh khối trên mặt đất, chỉ số diện tích lá (LAI),...
