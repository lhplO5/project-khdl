# Intro to Data Science Project  - CarbonGlobe

Xây dựng mô hình học máy thay thế bộ mô phỏng vật lý *Ecosystem Demography (ED) v3.0* để dự báo chu trình carbon và động lực học hệ sinh thái đất toàn cầu trên bộ dữ liệu **CarbonGlobe** ([IEEE BigData Cup 2026](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/overview)).

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
```
or use uv:
```fish
uv venv .venv
uv pip install -r requirements.txt
uv run python -m src.run_sprint1_2 
```

Với dữ liệu Kaggle thật, đặt file vào `data/raw/` rồi chạy `python -m src.run_sprint1_2`. Pipeline sẽ tự chọn **tập con 200–500 site** theo `configs/pipeline.yaml` (xem [Giao thức thực nghiệm](docs/experiment_protocol.md)).

## Lộ trình thực hiện

- [X] **Tuần 1 & 2:** Đồng bộ schema `.npz`/`.npy`, dựng pipeline nạp/kiểm toán/làm sạch/tạo đặc trưng `144 -> 7`, chạy notebook EDA & kiểm thử ([Báo cáo tiến độ](docs/weekly_reports/sprint_01.md) | [Báo cáo EDA chuyên sâu](docs/eda_report.md)).
- [ ] **Tuần 3:** Trên **tập con dữ liệu thật 200–500 site (seeded, stratified)** — mục tiêu khoa học chính — chạy EDA, ablation, chốt đặc trưng/ngưỡng làm sạch và huấn luyện baseline (`LightGBM`, `XGBoost`, `RandomForest`); đánh giá autoregressive rollout 40 năm trên **toàn bộ test8**. Xem [Giao thức thực nghiệm](docs/experiment_protocol.md).
- [ ] **Tuần 4:** Tinh chỉnh siêu tham số từ kết quả subset, thử nghiệm Deep Learning trên Kaggle/Colab GPU (nếu cần), xác nhận lại trên full train/test8 và tổng hợp báo cáo cuối kỳ.
