# ĐỒ ÁN HỌC MÁY CƠ BẢN: DỰ ĐOÁN NHU CẦU XE ĐẠP CÔNG CỘNG
**Môn học:** Học máy cơ bản (Mã lớp: `2611COMP171201`)  
**Giảng viên hướng dẫn:** TS. Ngô Quốc Việt  
**Tập dữ liệu:** [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (tệp `data/hour.csv` - 17,379 quan sát, 17 thuộc tính).

---

## 1. Cấu trúc thư mục dự án

```text
ml-bike-sharing/
├── .agent/rules/                      # Quy tắc và kịch bản thực hiện đề tài
├── data/                              # Dữ liệu gốc và dữ liệu sau xử lý
│   ├── hour.csv                       # Dữ liệu theo giờ (17,379 dòng)
│   ├── day.csv                        # Dữ liệu theo ngày
│   └── Readme.txt                     # Mô tả dữ liệu từ UCI
├── notebooks/                         # Sổ tay Jupyter Notebook theo từng giai đoạn
│   ├── step1_eda.ipynb                # Giai đoạn 3: Khám phá & Trực quan hóa dữ liệu (Đã xong)
│   ├── step2_prep.ipynb               # Giai đoạn 4: Tiền xử lý & Kỹ thuật đặc trưng (Sắp tới)
│   └── step3_models.ipynb             # Giai đoạn 5: Huấn luyện & Đánh giá mô hình (Sắp tới)
├── output/                            # Chứa biểu đồ sắc nét 300 DPI xuất tự động
│   ├── eda_1_histogram_log.png        # Histogram cnt gốc vs log1p(cnt)
│   ├── eda_2_boxplot_outliers.png     # Boxplot ngoại lai theo 4 mùa
│   ├── eda_3_time_trends.png          # Fever chart / Line chart theo 24 giờ
│   └── eda_4_scatter_heatmap.png      # Scatter plot & Heatmap đa cộng tuyến
├── report/                            # Bản thảo báo cáo khoa học (cỡ chữ 13, $\le 25$ trang)
│   └── bao_cao_do_an_hoc_may.md       # Báo cáo tổng hợp từ Trang 1 đến Trang 13 (Đang phát triển)
├── .gitignore                         # Bỏ qua venv, cache, file rác
├── requirements.txt                   # Danh sách thư viện cần thiết
└── README.md                          # Hướng dẫn dự án
```

---

## 2. Hướng dẫn cài đặt và chạy môi trường (Dành cho bạn cùng nhóm)

### Bước 1: Clone repo về máy
```bash
git clone <URL_REPO_CUA_BAN>
cd ml-bike-sharing
```

### Bước 2: Tạo môi trường ảo Python và kích hoạt
- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Chạy Notebook trong VS Code
1. Mở thư mục dự án `ml-bike-sharing` bằng VS Code.
2. Mở file `notebooks/step1_eda.ipynb`.
3. Nhìn góc trên bên phải của file Notebook, chọn Kernel là **Python Environment: `.\venv`**.
4. Bấm **Run All** để chạy toàn bộ các cell phân tích và tự động xuất ảnh ra `output/`.

---

## 3. Tiến độ hiện tại của dự án

* [x] **Giai đoạn 1 & 2 (Trang 1 – 10):** Đã biên soạn xong phần Mở đầu (Abstract, Introduction) và Cơ sở lý thuyết toán học (Related Methods: OLS, Ridge/Lasso, Sigmoid/Newton-Raphson, kNN, SVM Hinge Loss, Naïve Bayes Laplace, CART, Bagging vs Boosting).
* [x] **Giai đoạn 3 (Trang 11 – 13):**
  * Hoàn thành sổ tay khám phá dữ liệu `notebooks/step1_eda.ipynb`.
  * Thực hành nguyên tắc *Know Your Data*: phát hiện chính xác **22 bản ghi $hum = 0$** (lỗi cảm biến ngày 10/03/2011).
  * Kiểm tra phân phối lệch phải ($Mean > Median$), giải pháp $y' = \text{log1p}(y)$.
  * Khảo sát ngoại lai bằng IQR: **478 điểm ngoại lai** (Mùa xuân 195, Mùa đông 112; Mùa đông có đỉnh kỷ lục 967 xe). Cảnh báo nguy cơ học vẹt (Overfitting) cho **AdaBoost**.
  * Phát hiện quy luật 2 đỉnh giờ cao điểm (8h và 17-18h ngày đi làm) $\to$ tiền đề tạo biến `is_rush_hour`.
  * Vạch trần hiện tượng Đa cộng tuyến giữa `temp` và `atemp` ($r \approx 0.99 \implies$ bắt buộc xóa `atemp`) và Rò rỉ dữ liệu ($cnt = casual + registered \implies$ xóa `casual`, `registered`).
  * Xuất toàn bộ 4 biểu đồ chuẩn 300 DPI ra `output/` và tích hợp vào Báo cáo `report/bao_cao_do_an_hoc_may.md`.

* [ ] **Giai đoạn 4 (Tiếp theo - Trang 14 – 16):** Tiền xử lý, Kỹ thuật đặc trưng ($\sin/\cos$, `is_rush_hour`, K-Means Clustering, PCA) và Chuẩn hóa (Zero Data Leakage) trong `notebooks/step2_prep.ipynb`.
* [ ] **Giai đoạn 5 (Trang 17 – 23):** Thực nghiệm huấn luyện và đánh giá 11 mô hình trong `notebooks/step3_models.ipynb`.
* [ ] **Giai đoạn 6 (Trang 24 – 25):** Kết luận, Tài liệu tham khảo và xuất 01 file PDF hoàn chỉnh.
