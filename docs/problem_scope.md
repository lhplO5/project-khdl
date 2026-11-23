# Phạm vi bài toán & Cấu trúc dữ liệu CarbonGlobe

## 1. Bối cảnh
Bộ dữ liệu **CarbonGlobe** (IEEE BigData Cup 2026) mô phỏng mô hình vật lý *Ecosystem Demography (ED) v3.0* ở quy mô toàn cầu. Mục tiêu là huấn luyện mô hình học máy dự báo trạng thái chu trình carbon trong **40 năm** từ điều kiện môi trường và trạng thái rừng ban đầu.

## 2. Quy mô & Định dạng file thực tế
- **Lưới không gian:** $54.152$ điểm lưới đất liền toàn cầu (độ phân giải $0.5^\circ$, ánh xạ trên ma trận `mask2d.npy` kích thước $360 \times 720$).
- **Tuổi rừng khởi tạo (`AGES`):** $15$ mức (`1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500`).
- **Tập ML-ready (`data_global/res_train4_test8.npz`):**
  - `x_train`: `(3373, 40, 12, 136)` — 3.373 sites, 40 năm, 12 tháng, 136 biến đầu vào.
  - `y_train`: `(15, 3373, 41, 12, 7)` — 15 tuổi rừng, 41 năm (năm 0 là khởi tạo), 12 tháng, 7 biến mục tiêu.
  - `test8`: 852 sites đánh giá cố định (autoregressive rollout 40 năm).
- **Thống kê chuẩn hóa (`data_stats/data_stats.npz`):** `x_mean`, `x_std` `(136,)`; `y_mean`, `y_std` `(7,)`.

## 3. 7 biến mục tiêu (`TREE_BAND`)
1. `height`: Chiều cao tán rừng
2. `agb`: Sinh khối trên mặt đất (*Aboveground Biomass*)
3. `soil`: Trữ lượng carbon trong đất
4. `lai`: Chỉ số diện tích lá (*Leaf Area Index*)
5. `gpp`: Tổng năng suất sơ cấp (*Gross Primary Productivity*)
6. `npp`: Năng suất sơ cấp thuần (*Net Primary Productivity*)
7. `rh`: Hô hấp dị dưỡng (*Heterotrophic Respiration*)
