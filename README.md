# Đồ án Khoa học Dữ liệu: Dự báo Hệ sinh thái Đất Toàn cầu (CarbonGlobe)

Dự án bài tập lớn môn Khoa học dữ liệu, sử dụng bộ dữ liệu **CarbonGlobe** từ cuộc thi [IEEE BigData Cup 2026: AI Emulation Challenge](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/overview) để xây dựng mô hình học máy mô phỏng chu trình carbon và động lực học hệ sinh thái đất.

## 1. Mục tiêu bài toán
- **Đầu vào ($X$):** 136 biến môi trường (khí tượng, nồng độ $\text{CO}_2$, tính chất đất, tuổi rừng, tọa độ) qua chuỗi thời gian 40 năm tại $>54.152$ điểm lưới toàn cầu (độ phân giải $0.5^\circ$).
- **Đầu ra ($Y$):** Dự báo 7 biến mục tiêu của mô hình vật lý *Ecosystem Demography (ED)* (trữ lượng carbon, thông lượng carbon, sinh khối trên mặt đất, chỉ số diện tích lá, v.v.).
- **Độ đo đánh giá:** RMSE, MAE và sai số quỹ đạo theo thời gian ($\Delta$ error).

## 2. Cấu trúc thư mục
```text
project-khdl/
├── configs/        # File cấu hình đường dẫn, random seed, tham số sampling
├── data/           # Dữ liệu (raw, interim, processed) - không push file nặng lên Git
├── docs/           # Tài liệu mô tả dữ liệu và báo cáo tiến độ hàng tuần
├── notebooks/      # Jupyter notebooks cho Data Understanding, EDA, thử nghiệm
└── src/            # Mã nguồn Python tái sử dụng (data loader, preprocessing, evaluate)
```

## 3. Hướng dẫn cài đặt
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Tiến độ dự án
- [x] **Tuần 1:** Chốt đề tài, thiết kế repo, thiết lập cấu hình và khung khảo sát dữ liệu ([Báo cáo Tuần 1](docs/weekly_reports/week_01.md)).
- [ ] **Tuần 2:** Thu thập dữ liệu hoàn chỉnh, EDA trực quan hóa (136 biến), Pipeline làm sạch dữ liệu.
- [ ] **Tuần 3:** Feature Engineering (lag/rolling features), Baseline Models (LightGBM/XGBoost/Ridge).
- [ ] **Tuần 4:** Tinh chỉnh mô hình, thử nghiệm Deep Learning trên Colab/Kaggle GPU (nếu cần), đánh giá & viết báo cáo tổng kết.
