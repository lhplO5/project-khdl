# CarbonGlobe Emulator — Đồ án Khoa học Dữ liệu

Bài tập lớn môn Khoa học dữ liệu: xây dựng mô hình học máy thay thế bộ mô phỏng vật lý *Ecosystem Demography (ED) v3.0* để dự báo chu trình carbon và động lực học hệ sinh thái đất toàn cầu trên bộ dữ liệu **CarbonGlobe** ([IEEE BigData Cup 2026](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/overview)).

## Bài toán đặt ra
- **Dữ liệu đầu vào:** 136 biến môi trường (12 tháng $\times$ 40 năm) trên 54.152 điểm lưới toàn cầu ($0.5^\circ$) và 15 mức tuổi rừng khởi tạo (`1..500`).
- **Biến cần dự báo (`TREE_BAND`):** `height`, `agb`, `soil`, `lai`, `gpp`, `npp`, `rh`.
- **Tiêu chí đánh giá:** Autoregressive rollout 40 năm với `RMSE`, `MAE`, `R2`, `delta`, `cumulative`.

## Tổ chức thư mục
```text
project-khdl/
├── configs/        # Tham số pipeline (pipeline.yaml)
├── data/           # Dữ liệu cục bộ (raw / interim / processed)
├── docs/           # Đặc tả dữ liệu, kiến trúc pipeline & báo cáo tiến độ
├── notebooks/      # Notebook khảo sát schema, EDA & làm sạch (có sẵn output/biểu đồ)
├── src/            # Mã nguồn: ingest, dataset, profiling, clean, features, run_sprint1_2
└── tests/          # Kiểm thử tự động pipeline dữ liệu
```

## Cài đặt & Chạy thử Pipeline (Tuần 1 & 2)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Chạy toàn bộ pipeline Tuần 1 & 2 trên dữ liệu mô phỏng
python -m src.run_sprint1_2 --synthetic
```

## Lộ trình thực hiện
- [x] **Tuần 1 & 2:** Đồng bộ schema `.npz`/`.npy`, dựng pipeline nạp/kiểm toán/làm sạch/tạo đặc trưng `144 -> 7`, chạy notebook EDA & kiểm thử ([Báo cáo tiến độ](docs/weekly_reports/sprint_01.md) | [Báo cáo EDA chuyên sâu](docs/eda_report.md)).
- [ ] **Tuần 3:** Huấn luyện mô hình nền (`LightGBM`, `XGBoost`, `RandomForest`) và đánh giá autoregressive rollout 40 năm.
- [ ] **Tuần 4:** Tối ưu siêu tham số, thử nghiệm Deep Learning trên Kaggle/Colab GPU (nếu cần) và tổng hợp báo cáo cuối kỳ.
