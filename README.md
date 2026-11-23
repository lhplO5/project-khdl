# CarbonGlobe Emulator — Đồ án Khoa học Dữ liệu

Bài tập lớn môn Khoa học dữ liệu: xây dựng mô hình học máy thay thế bộ mô phỏng vật lý *Ecosystem Demography (ED)* để dự báo chu trình carbon và động lực học hệ sinh thái đất toàn cầu trên bộ dữ liệu **CarbonGlobe** ([IEEE BigData Cup 2026](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/overview)).

## Bài toán đặt ra
- **Dữ liệu đầu vào:** 136 biến môi trường (khí tượng, $\text{CO}_2$, thổ nhưỡng, tuổi rừng, tọa độ) trải dài 40 năm trên hơn 54.152 điểm lưới toàn cầu ($0.5^\circ$).
- **Biến cần dự báo:** 7 chỉ số sinh thái đầu ra (trữ lượng & thông lượng carbon, sinh khối trên mặt đất, chỉ số diện tích lá,...).
- **Tiêu chí đánh giá:** RMSE, MAE và sai số biến thiên theo từng năm ($\Delta$ error).

## Tổ chức mã nguồn
```text
project-khdl/
├── carbonglobe/       # Thư viện xử lý dữ liệu & mô hình của nhóm
├── datasets/          # Dữ liệu cục bộ (source / staging / ready)
├── exploration/       # Notebook khảo sát, trực quan hóa & thử nghiệm
├── notes/             # Ghi chú bài toán, đặc tả biến
├── reports/           # Báo cáo tiến độ qua từng tuần
└── settings.yaml      # Tham số đường dẫn, seed và cấu hình lấy mẫu
```

## Cài đặt môi trường
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lộ trình thực hiện
- **Tuần 1:** Chốt đề tài, dựng khung package `carbonglobe`, cấu hình quy trình kiểm tra dữ liệu ([Xem báo cáo](reports/sprint_01_status.md)).
- **Tuần 2:** Tải dữ liệu Kaggle, phân tích phân phối 136 biến & chuỗi 40 năm, hoàn thiện bộ lọc làm sạch.
- **Tuần 3:** Tạo đặc trưng trễ theo thời gian, chạy các mô hình nền (LightGBM, XGBoost, Ridge).
- **Tuần 4:** Tối ưu siêu tham số, thử nghiệm mạng nơ-ron (trên Colab/Kaggle GPU nếu cần) và tổng hợp kết quả.
