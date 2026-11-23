# Đặc tả Bài toán & Cấu trúc Dữ liệu CarbonGlobe

## 1. Bối cảnh khoa học & Mục tiêu dự án
Mô hình vật lý **Ecosystem Demography (ED) v3.0** là một trong những mô hình hàng đầu để mô phỏng động lực học rừng và chu trình carbon đất liền toàn cầu. Tuy nhiên, chi phí tính toán của ED v3.0 rất lớn khi chạy trên hàng chục nghìn điểm lưới qua nhiều thập kỷ.

Dự án này sử dụng bộ dữ liệu **CarbonGlobe** (từ cuộc thi [IEEE BigData Cup 2026: AI Emulation Challenge](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/overview)) để xây dựng một **AI Emulator** — mô hình học máy có khả năng dự báo nhanh 7 chỉ số hệ sinh thái rừng trong **40 năm liên tiếp** dựa trên điều kiện môi trường hàng tháng và trạng thái rừng khởi tạo.

---

## 2. Hai tầng dữ liệu của CarbonGlobe

| Tầng dữ liệu | Đường dẫn thực tế | Kích thước / Shape | Mục đích sử dụng trong đồ án |
| :--- | :--- | :--- | :--- |
| **Tầng 1: ML-ready** | `data_global/res_train4_test8.npz` | `x_train`: `(3373, 40, 12, 136)`<br>`y_train`: `(15, 3373, 41, 12, 7)`<br>`test8`: `852` sites cố định | Dùng cho huấn luyện, kiểm định chéo và đánh giá chuẩn của các mô hình ML mà không phải tải mảng 14 GB. |
| **Tầng 2: Toàn cầu (Global)** | `data_global/glob_X_fea.npy`<br>`data_global/glob_Y{age}.npy` | `X`: `(54152, 40, 12, 136)` (~14 GB)<br>`Y`: `(54152, 41, 12, 7)` mỗi mức tuổi | Mở bằng `np.load(..., mmap_mode='r')` để lập bản đồ sai số toàn cầu trên toàn bộ 50.779 điểm lưới `test.csv`. |
| **Thống kê & Mặt nạ** | `data_stats/data_stats.npz`<br>`data_stats/mask2d.npy`<br>`data_stats/age_triplet.npz` | `x_mean/std`: `(136,)`<br>`y_mean/std`: `(7,)`<br>`mask2d`: `(360, 720)` | Chuẩn hóa Z-score thống nhất giữa train/test và ánh xạ kết quả dự báo lên bản đồ không gian 2D ($0.5^\circ$). |

---

## 3. Chi tiết các chiều dữ liệu & 7 biến mục tiêu (`TREE_BAND`)

* **Không gian:** $54.152$ điểm lưới đất liền toàn cầu (độ phân giải $0.5^\circ$, ánh xạ trên `mask2d.npy` kích thước $360 \times 720$, trong đó `0` = đại dương, `1` = train `3.035` sites, `2` = val `338` sites, `3` = test `50.779` sites).
* **Tuổi rừng khởi tạo (`AGES`):** 15 mức: `[1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]` năm.
* **Thời gian:** 40 năm dự báo (mảng $Y$ có 41 năm, trong đó năm 0 là trạng thái ban đầu `y0` đã biết), mỗi năm gồm 12 tháng.
* **Đầu vào ($X$ — 136 kênh):** Bao gồm các nhóm biến khí tượng (nhiệt độ, lượng mưa, bức xạ sóng ngắn/dài, độ ẩm, tốc độ gió), nồng độ $\text{CO}_2$ khí quyển, đặc tính lý-hóa của đất (độ sâu, thành phần cát/sét, pH, độ dẫn thủy lực) và ngữ cảnh địa hình/sinh thái.

| STT | Tên biến | Tên tiếng Anh | Ý nghĩa sinh thái trong chu trình Carbon |
| :-: | :--- | :--- | :--- |
| 1 | `height` | Canopy Height | Chiều cao tán rừng, phản ánh cấu trúc thẳng đứng và giai đoạn phát triển. |
| 2 | `agb` | Aboveground Biomass | Sinh khối trên mặt đất — biến cốt lõi để ước lượng trữ lượng carbon thực vật. |
| 3 | `soil` | Soil Carbon | Trữ lượng carbon hữu cơ trong đất, biến đổi chậm nhưng chiếm tỷ trọng rất lớn. |
| 4 | `lai` | Leaf Area Index | Chỉ số diện tích lá, quyết định khả năng hấp thụ bức xạ và quang hợp. |
| 5 | `gpp` | Gross Primary Productivity | Tổng năng suất sơ cấp (tổng lượng carbon cây hấp thụ qua quang hợp). |
| 6 | `npp` | Net Primary Productivity | Năng suất sơ cấp thuần ($\text{GPP} - \text{hô hấp tự dưỡng}$). |
| 7 | `rh` | Heterotrophic Respiration | Hô hấp dị dưỡng (lượng carbon phát thải từ phân hủy vật chất hữu cơ trong đất). |

---

## 4. Bài toán học máy & Giao thức đánh giá

### 4.1. Huấn luyện One-Step-Ahead (Teacher Forcing) vs Kiểm thử Tự hồi quy (Autoregressive Rollout)
* **Khi huấn luyện:** Ở mỗi năm $t \in \{1, \dots, 40\}$, mô hình nhận đầu vào `144` chiều gồm:
  $$\mathbf{f}_{s,a,t} = \big[\,\bar{\mathbf{x}}_{s,t}^{(136)} \;\oplus\; \mathbf{y}_{s,a,t-1,\text{Dec}}^{(7)} \;\oplus\; (a / 500)\,\big] \;\longrightarrow\; \hat{\mathbf{y}}_{s,a,t,\text{Dec}}^{(7)}$$
  Trong đó trạng thái năm trước $\mathbf{y}_{t-1}$ lấy từ **giá trị thật** (*teacher forcing*).
* **Khi kiểm thử (Rollout 40 năm):** Chỉ cung cấp trạng thái khởi tạo $\mathbf{y}_0$. Từ năm $t=1$ đến $t=40$, dự báo $\hat{\mathbf{y}}_{t-1}$ của chính mô hình được đưa ngược lại làm đầu vào cho năm $t$. Cách đánh giá này phản ánh đúng bài toán thực tế và làm lộ rõ hiện tượng **tích lũy sai số (long-horizon drift)**.

### 4.2. Năm độ đo đánh giá (tính trên đơn vị vật lý)
1. **RMSE:** Sai số bình phương trung bình trên toàn bộ chuỗi 40 năm, phạt nặng sai số lớn.
2. **MAE:** Sai số tuyệt đối trung bình, ít nhạy cảm với ngoại lai hơn RMSE.
3. **$R^2$:** Tỷ lệ phương sai được mô hình giải thích cho từng biến mục tiêu.
4. **$\Delta$ Error (`delta`):** RMSE của độ biến thiên giữa hai năm liên tiếp ($\Delta y_t = y_t - y_{t-1}$), đo khả năng nắm bắt động lực học theo năm.
5. **Cumulative Error (`cumulative`):** RMSE tại bước thời gian cuối cùng ($t = 40$), đo trực tiếp mức độ trôi sai số tích lũy sau 4 thập kỷ.

---

## 5. Rủi ro kỹ thuật & Phương án giảm thiểu
* **Dung lượng RAM lớn:** Mảng `float64` toàn cầu tốn >28 GB RAM $\rightarrow$ sử dụng `float32`, tập `res_train4_test8.npz` và `mmap_mode='r'` cho mảng toàn cầu.
* **Trôi sai số khi cuộn 40 năm:** Mô hình cây quyết định đơn bước dễ bị lệch dần nếu dự báo sai ở các năm đầu $\rightarrow$ cần làm sạch ngoại lai chuẩn hóa nghiêm ngặt, kiểm tra tương quan tự hồi quy bậc 1 ở Tuần 2, và so sánh nhiều họ mô hình (Random Forest, LightGBM, XGBoost, Deep Learning) ở Tuần 3–4.
