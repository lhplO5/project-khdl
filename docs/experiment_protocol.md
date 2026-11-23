# Giao thức Thực nghiệm Tuần 3–4: Tập con Dữ liệu Thật

## 1. Mục tiêu và phạm vi

Trong Tuần 3–4, mọi quyết định khoa học (EDA, ablation, chọn đặc trưng, chốt ngưỡng làm sạch và huấn luyện baseline) được thực hiện trên **một tập con dữ liệu Kaggle thật, cố định theo seed**. Đây là **mục tiêu khoa học có ý nghĩa** của đồ án; dữ liệu mô phỏng chỉ dùng để kiểm chứng pipeline.

| Tầng dữ liệu | Quy mô | Vai trò | Có dùng để báo cáo khoa học? |
| :--- | :--- | :--- | :---: |
| **Real subset (seeded)** | **200–500 site train** | **EDA, ablation, chốt feature/ngưỡng clean, baseline modeling** | **Có** |
| Full ML-ready | 3.373 site train / 852 site test8 | Xác nhận cuối cùng và mở rộng nếu đủ ngân sách tính toán | Có (kết quả cuối) |

Tập con **chỉ lấy từ train**. Tập `test8` (852 site cố định) được giữ nguyên cho giao thức autoregressive rollout 40 năm, nhờ đó mọi cấu hình được đánh giá trên cùng một tập và kết quả có thể so sánh trực tiếp.

---

## 2. Cấu hình tập con

Cấu hình nằm trong `configs/pipeline.yaml`:

```yaml
seed: 42

subset:
  enabled: true
  max_sites: 500
  strategy: "stratified"   # head | random | stratified
  stratify_by: "agb"
  n_bins: 5
  subset_test: false       # giữ nguyên toàn bộ test8
```

### 2.1. Lấy mẫu phân tầng (mặc định)

1. Mỗi site được gán tầng theo **phân vị của `agb` tại trạng thái ban đầu** (`y_train[:, :, 0, 11, agb]`, trung bình qua 15 mức tuổi). `n_bins = 5` tạo 5 dải sinh khối từ thấp đến cao.
2. Số site lấy từ mỗi tầng **tỷ lệ với kích thước tầng**, mỗi tầng có ít nhất 1 site.
3. Trong từng tầng, site được chọn ngẫu nhiên không hoàn lại bằng `numpy.random.default_rng(seed)`.
4. Chỉ số site được sắp xếp tăng dần và ghi lại trong `data/interim/audit_real.json` → tái lập chính xác 100%.

Vì phân bổ theo tỷ lệ, tập con xấp xỉ phân phối toàn cầu của train nhưng **giảm phương sai giữa các dải sinh thái**, tránh trường hợp một mẫu ngẫu nhiên nhỏ bỏ sót rừng già hoặc vùng khô hạn.

### 2.2. Lấy mẫu ngẫu nhiên

Đặt `strategy: "random"` để lấy mẫu ngẫu nhiên đều, vẫn cố định theo `seed`. Dùng làm **ablation đối chứng** cho lựa chọn phân tầng.

### 2.3. Kích thước mẫu

- Khoảng **200–500 site** cân bằng giữa độ phủ không gian–sinh thái và chi phí CPU.
- Ở mức 500 site, tập one-step có $500 \times 15 \times 40 = 300.000$ dòng, huấn luyện LightGBM/XGBoost/Ridge trên CPU trong vài phút.
- Không giảm dưới 200 site vì mỗi nhóm sinh thái chỉ còn quá ít mẫu, làm ablation và phân tích lỗi theo vùng kém tin cậy.

---

## 3. Việc gì được quyết định trên tập con?

| Hạng mục | Được quyết định trên subset? | Ghi chú |
| :--- | :---: | :--- |
| EDA, phân bố, tương quan, autocorrelation | Có | Dùng để hiểu dữ liệu; kiểm tra lại trên full nếu kết luận quan trọng |
| Ablation đặc trưng (mùa vụ, lag, tuổi) | Có | So sánh công bằng trên cùng subset + cùng test8 |
| Ngưỡng làm sạch (`clip_z_range`) | Có | Chọn ngưỡng theo phân vị ngoại lai của subset, không dựa vào test |
| Chọn họ mô hình baseline và hyperparameter sơ bộ | Có | Chỉ chốt cấu hình, không chốt con số báo cáo |
| Đánh giá rollout 40 năm | **Không** | Luôn chạy trên **toàn bộ test8** |
| Bảng kết quả cuối kỳ | **Không** | Xác nhận lại trên full train (nếu đủ compute) và test8 |

---

## 4. Quy tắc chống rò rỉ và thiên lệch

- Không bao giờ dùng `test8` để chọn site, chọn ngưỡng hay chọn đặc trưng.
- `x_mean/x_std/y_mean/y_std` lấy từ `data_stats.npz` toàn cục, **không tính lại trên subset**, để giữ nguyên không gian chuẩn hóa giữa train/test và giữa các lần chạy.
- Cùng một `seed` + cùng cấu hình ⇒ cùng chỉ số site. Thay đổi seed phải được báo cáo như một thí nghiệm độ nhạy.
- Khi so sánh mô hình, giữ cố định tập con, tập test, seed và thứ tự đặc trưng.

---

## 5. Quy trình chạy
append infront `uv run`
```bash
# 1. Tải dữ liệu thật vào data/raw/ (Kaggle CLI)
python -m src.ingest

# 2. Sinh tập con + audit + đặc trưng one-step
python -m src.run_sprint1_2

# 3. Kiểm tra tái lập: chạy lại lệnh trên, so sánh
#    trường "site_indices" trong data/interim/audit_real.json
```

> ! If kaggle setup isn't valid for data ingestion: 
Paste in the Kaggle API token on command 
```fish
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle

read -P "Paste Kaggle token: " token
printf '%s' "$token" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
set -e token
```
and then run
```fish
uv pip install --python .venv/bin/python --upgrade kaggle
```

```fish 
uv run kaggle --version
uv run kaggle competitions list
```

Đầu ra:
- `data/interim/audit_real.json` — audit chất lượng + cấu hình subset + danh sách `site_indices`.
- `data/processed/one_step_real.npz` — `X_feat`, `Y_target` của tập con.

---

## 6. Mẫu ghi nhận quyết định (decision log)

| Ngày | Quyết định | Bằng chứng trên subset | Ảnh hưởng lên test8 (rollout) |
| :--- | :--- | :--- | :--- |
| | Ví dụ: giữ `clip_z_range = ±10` | đuôi phân phối, số điểm bị cắt | RMSE/MAE/cumulative thay đổi ra sao |
| | Ví dụ: thêm đặc trưng mùa vụ | ablation trên 500 site | delta error giảm bao nhiêu |

Mọi thay đổi chỉ được coi là "đã chốt" khi có cả bằng chứng trên subset **và** kiểm chứng rollout trên test8.
