# Kiến trúc Pipeline Dữ liệu (Tuần 1 & Tuần 2)

## 1. Luồng xử lý End-to-End
```text
Kaggle / data/raw/ (res_train4_test8.npz, data_stats.npz, mask2d.npy)
   ├─▶ [Tuần 1] Kiểm toán chất lượng (src/profiling.py) ──▶ data/interim/audit_*.json
   └─▶ [Tuần 2] Chuẩn hóa Z-score (src/dataset.py)
                    └─▶ Làm sạch NaN/Inf & Winsorizing (src/clean.py)
                            └─▶ Tạo đặc trưng 1-step 144 chiều (src/features.py)
                                    └─▶ data/processed/one_step_*.npz
```

## 2. Quy tắc xử lý & Tạo đặc trưng
1. **Chuẩn hóa Z-score:** Dùng trực tiếp `x_mean`, `x_std`, `y_mean`, `y_std` từ `data_stats.npz` với ngưỡng bảo vệ `eps = 1e-6` tránh chia cho 0.
2. **Làm sạch mảng đa chiều:** Thay thế giá trị `NaN`/`Inf` bằng trung bình chuẩn hóa (`0.0`), sau đó cắt ngưỡng ngoại lai cực đoan trong khoảng `[-10.0, 10.0]`.
3. **Chuyển đổi sang bài toán One-Step-Ahead (144 $\rightarrow$ 7):**
   - Trung bình 12 tháng của đầu vào thành vector năm (`136` chiều).
   - Ghép thêm vector đầu ra tháng 12 của năm $t-1$ (`7` chiều) và tuổi rừng chuẩn hóa `age / 500.0` (`1` chiều) $\rightarrow$ tổng **`144` đặc trưng** để dự báo 7 biến tháng 12 năm $t$.

## 3. Lệnh chạy nhanh
```powershell
# Chạy kiểm thử tự động
python -c "from tests.test_pipeline import test_end_to_end_sprint1_and_sprint2; test_end_to_end_sprint1_and_sprint2(); print('PASS')"

# Chạy pipeline Tuần 1 + Tuần 2 trên dữ liệu mô phỏng
python -m src.run_sprint1_2 --synthetic
```
