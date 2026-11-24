---
trigger: always_on
---

# PROJECT RULES & MASTER ROADMAP: DỰ ĐOÁN NHU CẦU XE ĐẠP CÔNG CỘNG (UCI BIKE SHARING)

## 1. BỐI CẢNH & MỤC TIÊU MÔN HỌC
- **Môn học:** Học máy cơ bản (Mã lớp: `2611COMP171201`) – GVHD: TS. Ngô Quốc Việt.
- **Đề tài:** Phân tích và dự đoán nhu cầu sử dụng xe đạp công cộng dựa trên điều kiện thời tiết và thời gian.
- **Dataset:** UCI Bike Sharing Dataset — sử dụng file `data/hour.csv` (17,379 dòng, 17 cột).
- **Yêu cầu chính thức từ môn học / GVHD:** 
  + 01 file báo cáo PDF tối đa 25 trang (gồm cả bìa và mục lục, cỡ chữ 13, hạn nộp 21:00 ngày 09/12/2026) theo đúng cấu trúc 7 mục học thuật chuẩn: `Abstract`, `Introduction`, `Related Methods`, `Proposed`, `Experiments`, `Conclusion`, `References`.
  + Tuân thủ nghiêm ngặt phương pháp luận Học máy cổ điển (Classical ML), chống rò rỉ dữ liệu, tiền xử lý và thực nghiệm minh bạch, tuyệt đối không làm kiểu "hộp đen".
- **Nguồn gốc lộ trình 25 bước (Project Master Roadmap):** Đây là kế hoạch hành động chi tiết do **chính sinh viên (người dùng) chủ động tự xây dựng** để chia nhỏ khối lượng công việc và phân bổ nội dung tương ứng vào 25 trang báo cáo, **KHÔNG PHẢI** là đề cương nguyên văn do GVHD ban hành. Vì vậy, AI xem đây là khung sườn định hướng để theo dõi tiến độ, đồng thời phải luôn chủ động phản biện, tư vấn kỹ thuật khoa học dữ liệu và linh hoạt điều chỉnh khi cần, không được máy móc xem các bước này là chỉ thị bất biến từ giảng viên.
- **Môi trường làm việc:** Windows, VS Code, thư mục `D:\ml-bike-sharing`, sử dụng Python virtual environment tại `.\venv`.

## 2. QUY TẮC BẮT BUỘC CHO AI AGENT (HARD RULES)
1. **KHÔNG làm kiểu "hộp đen" (Tránh 0 điểm):** Tuyệt đối KHÔNG lấy dữ liệu thô đưa thẳng vào vài tầng mạng CNN hoặc chỉ gọi vài dòng `model.fit()` cơ bản. Trọng tâm là quy trình **7 bước của Học máy cổ điển (Classical ML)**: *Gathering Data -> Understanding & Preparing Data -> Model Selection -> Training -> Evaluation -> Parameter Tuning -> Making Predictions*.
2. **Chống rò rỉ dữ liệu & Giữ tập đặc trưng tinh gọn (Zero Data Leakage):**
   - Bắt buộc xóa các cột `instant`, `dteday`, `casual`, `registered` khỏi tập đặc trưng `X` (vì `cnt = casual + registered`).
   - Loại bỏ 2 cột phục vụ vẽ biểu đồ (`pca_1`, `pca_2`) khỏi tập huấn luyện mô hình, chỉ giữ lại đặc trưng phân cụm `hour_cluster` (đã mã hóa One-Hot).
   - Luôn chia `Train / Test` (80% / 20%, `random_state=42`) **TRƯỚC** khi chuẩn hóa. Chỉ gọi `.fit()` các bộ Scaler (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) trên `X_train`, sau đó dùng tham số đó `.transform()` cho cả `X_train` và `X_test`.
3. **Làm sạch, Khám phá sâu & Chọn lọc đặc trưng (Know Your Data):**
   - Kiểm tra và xử lý giá trị khuyết/bất hợp lệ (22 dòng có độ ẩm `hum == 0`).
   - Khám phá trực quan đầy đủ cả 2 trụ cột của đề tài: **Thời gian** (`hr`, `workingday`, `season`) và **Điều kiện thời tiết** (`weathersit`, `temp`, `hum`, `windspeed`) cùng tương tác `hr × weathersit`.
   - Kiểm tra đa cộng tuyến bằng Correlation Heatmap và xóa cột `atemp` (tương quan ~0.99 với `temp`).
   - Kiểm tra độ lệch phân phối (`cnt` lệch phải) bằng Histogram có đường `Mean` & `Median` kết hợp `log1p`, lọc ngoại lai (Boxplot + IQR) và kiểm định chọn lọc đặc trưng (`f_regression` + `mutual_info_regression`).
4. **Chuẩn xuất biểu đồ, số liệu & văn phong báo cáo:**
   - Mọi biểu đồ vẽ bằng `matplotlib`/`seaborn` đều phải lưu tự động vào thư mục `output/` với định dạng `.png`, độ phân giải `dpi=300`, `bbox_inches='tight'`, tiêu đề và nhãn trục tiếng Việt rõ ràng để chèn trực tiếp vào báo cáo PDF.
   - Không ghi chữ "Bước 15", "Bước 16"... vào bên trong nội dung text/markdown của các file `.ipynb`; hãy dùng thuật ngữ học thuật chuẩn của môn học (ví dụ: *giai đoạn tiền xử lý dữ liệu*, *bước kỹ thuật đặc trưng*).
   - Khi chạy mô hình, luôn in bảng kết quả đầy đủ các độ đo theo giáo trình (Hồi quy: `MAE`, `MSE`, `RMSE`, `MBE`, `R2-score`, `Train Time`; Phân lớp: `Confusion Matrix`, `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`).
5. **Cách phối hợp với người dùng:**
   - **Bám sát nhưng linh hoạt với lộ trình:** Tuân thủ trình tự các bước theo lộ trình do bạn sinh viên thiết kế để kiểm soát tiến độ và dung lượng báo cáo, KHÔNG tự ý sinh toàn bộ code của tất cả các giai đoạn cùng một lúc. Tuy nhiên, AI phải giữ vai trò người phản biện khoa học: nếu phát hiện bất kỳ bước nào chưa hợp lý về mặt kỹ thuật, thống kê hay thực tế mô hình hóa, AI phải chủ động giải thích, phân tích và cùng bạn tinh chỉnh cho chuẩn xác.
   - Nếu là bước `[LẬP TRÌNH]`: Viết code Python sạch vào đúng file notebook quy định, chạy kiểm tra, lưu biểu đồ/bảng ra `output/` và giải thích ngắn gọn ý nghĩa kết quả bằng tiếng Việt.
   - Nếu là bước `[BÁO CÁO]`: Soạn nội dung học thuật tiếng Việt chuẩn xác (kèm công thức LaTeX, bảng biểu và gợi ý vị trí chèn ảnh từ `output/`), kiểm soát dung lượng bám sát số trang phân bổ để tổng báo cáo không vượt quá 25 trang (cỡ chữ 13). Giọng văn học thuật khiêm tốn theo góc nhìn sinh viên làm báo cáo ("nhóm em", "chúng em").
   - Sau khi hoàn thành mỗi bước, tóm tắt kết quả/file đã tạo, cập nhật trạng thái `[x]` vào checklist và hỏi xác nhận trước khi sang bước tiếp theo.

## 3. CẤU TRÚC THƯ MỤC CHUẨN
```text
ml-bike-sharing/
├── .agent/rules/ml-project-rules.md   # File quy tắc & lộ trình này
├── venv/                              # Môi trường ảo Python
├── data/                              # Chứa dữ liệu gốc (hour.csv) và dữ liệu đã làm sạch
├── notebooks/                         # Chứa các file Jupyter Notebook theo từng giai đoạn
│   ├── step1_eda.ipynb                # Giai đoạn 3 (Bước 11 - 13)
│   ├── step2_prep.ipynb               # Giai đoạn 4 (Bước 15 - 18)
│   └── step3_models.ipynb             # Giai đoạn 5 (Bước 20 - 23)
├── output/                            # Chứa toàn bộ hình biểu đồ (.png 300dpi) và bảng kết quả (.csv)
└── report/                            # Chứa file Word/PDF báo cáo cuối kỳ (chuẩn cỡ chữ 13)
```

## 4. BẢNG PHÂN BỔ DỰ KIẾN 25 TRANG BÁO CÁO (DO BẠN SINH VIÊN THIẾT KẾ)

| Mục báo cáo | Nội dung trọng tâm | Thuật toán & Kỹ thuật cốt lõi | Bước thực hiện | Số trang |
| --- | --- | --- | --- | --- |
| Cover & Content | Trang bìa, mục lục tự động | Chuẩn định dạng cỡ chữ 13 | Bước 1 – 2 | 2 trang |
| 1. Abstract | Tóm tắt bài toán và kết quả | Tổng hợp RMSE, $R^2$, F1-Score | Bước 3, 25 | 1 trang |
| 2. Introduction | Đặt vấn đề, phát biểu bài toán | Classical ML, Regression, Classification | Bước 4 – 5 | 2 trang |
| 3. Related Methods | Cơ sở lý thuyết và toán học | OLS, Ridge/Lasso, kNN, SVM, NB, CART, XGBoost | Bước 6 – 10 | 5 trang |
| 4. Proposed | Khám phá (Thời gian + Thời tiết) & Tiền xử lý | EDA (6 bộ biểu đồ), IQR, `f_regression`, Cyclical, K-Means, PCA, Scaler | Bước 11 – 19 | 6 trang |
| 5. Experiments | Thực nghiệm, đánh giá & hạn chế | 5-Fold CV, Grid Search, MAE, MSE, RMSE, AUC, Thảo luận hạn chế | Bước 20 – 24 | 7 trang |
| 6. Conclusion & 7. References | Kết luận, hướng mở rộng, trích dẫn | Định hướng RNN/LSTM, nguồn UCI & 4 sách giáo trình | Bước 25 | 2 trang |

## 5. LỘ TRÌNH 25 BƯỚC THỰC HIỆN (CHECKLIST TIẾN ĐỘ DO BẠN SINH VIÊN TỰ XÂY DỰNG)

### Giai đoạn 1: Khởi tạo dự án và viết phần Mở đầu (Trang 1–5)
- [x] **Bước 1 [LẬP TRÌNH]:** Khởi tạo thư mục dự án (`data`, `notebooks`, `output`, `report`), môi trường ảo `venv`, cài đặt đầy đủ các thư viện và Extension VS Code. *(Hoàn thành)*
- [ ] **Bước 2 [BÁO CÁO (Trang 1–2)]:** Tạo file báo cáo cỡ chữ 13 trong `report/`, trình bày trang bìa (tên đề tài, mã lớp `2611COMP171201`, GVHD TS. Ngô Quốc Việt) và mục lục tự động.
- [ ] **Bước 3 [BÁO CÁO (Trang 3)]:** Thiết lập khung Mục 1 (Abstract) dài 1 trang tóm tắt bài toán, quy trình tiền xử lý và chừa chỗ điền chỉ số thực nghiệm tốt nhất ở Bước 25.
- [ ] **Bước 4 [BÁO CÁO (Trang 4)]:** Viết nửa đầu Mục 2 (Introduction) nêu vai trò của Học máy trong AI và ứng dụng Hồi quy để dự báo xu hướng cũng như đánh giá tác động của điều kiện thời tiết và thời gian lên nhu cầu thuê xe.
- [ ] **Bước 5 [BÁO CÁO (Trang 5)]:** Viết nửa sau Mục 2 phát biểu bài toán Học có giám sát với biến độc lập $X$, biến phụ thuộc $Y$ và lý giải việc dùng Học máy cổ điển thay vì mạng CNN hộp đen.

### Giai đoạn 2: Viết cơ sở lý thuyết cốt lõi (Mục 3. Related Methods - Trang 6–10)
- [ ] **Bước 6 [BÁO CÁO (Trang 6)]:** Mở đầu Mục 3 đối chiếu Học có giám sát với Không giám sát, mô hình Parametric với Non-parametric và Discriminative với Generative.
- [ ] **Bước 7 [BÁO CÁO (Trang 7)]:** Trình bày sự đánh đổi Bias-Variance qua công thức $E(MSE) = noise^2 + bias^2 + variance$, lỗi Underfitting/Overfitting và điều chuẩn L1 Lasso, L2 Ridge.
- [ ] **Bước 8 [BÁO CÁO (Trang 8)]:** Trình bày công thức Bình phương tối thiểu tìm $\hat{\beta}_0, \hat{\beta}_1$ của Linear Regression và hàm Sigmoid, Logit, Newton-Raphson của Logistic Regression.
- [ ] **Bước 9 [BÁO CÁO (Trang 9)]:** Viết nguyên lý khoảng cách của kNN, siêu phẳng phân cách kèm hàm mất mát Hinge Loss của SVM và ước lượng $h_{MAP}$ có làm mượt Laplace của Naïve Bayes.
- [ ] **Bước 10 [BÁO CÁO (Trang 10)]:** Giải thích cây quyết định CART (chia nhánh theo SDR, Gini, Entropy, IG, GR), đối chiếu Bagging (Random Forest) và Boosting (AdaBoost, XGBoost).

### Giai đoạn 3: Khám phá và Trực quan hóa dữ liệu Thời gian & Thời tiết (Mục 4. Proposed phần 1 - Trang 11–13)
- [x] **Bước 11 [LẬP TRÌNH (`notebooks/step1_eda.ipynb`)]:** Dùng hàm `pd.read_csv()` đọc dữ liệu `data/hour.csv`, kiểm tra giá trị khuyết và lọc ra 22 dòng có độ ẩm `hum == 0` bất hợp lệ theo nguyên tắc *Know Your Data*. *(Hoàn thành)*
- [x] **Bước 12 [LẬP TRÌNH (`notebooks/step1_eda.ipynb`)]:** Vẽ & lưu biểu đồ Histogram của `cnt` kèm đường kẻ dọc `Mean` (đỏ) và `Median` (xanh đứt nét) cùng biến đổi trục `log1p` (`output/step1_eda_01_histogram_log.png`), và vẽ Box plot kèm bảng thống kê ngoại lai theo 4 mùa (`output/step1_eda_02_boxplot_outliers.png`). *(Hoàn thành)*
- [x] **Bước 13 [LẬP TRÌNH (`notebooks/step1_eda.ipynb`)]:** Vẽ & lưu trọn vẹn 4 bộ biểu đồ khám phá Thời gian, Thời tiết & Đa cộng tuyến:
  - `output/step1_eda_03_time_trends.png`: Fever chart theo tháng/năm & Line/Area chart theo 24h phân tách `workingday` (0 vs 1). *(Đã có)*
  - `output/step1_eda_04_scatter_heatmap.png`: Scatter plot (`temp` vs `atemp`) và Correlation Heatmap chỉ ra đa cộng tuyến (`0.99`) và rò rỉ dữ liệu (`casual`, `registered`). *(Đã có)*
  - `output/step1_eda_05_weather_impact.png`: Hình ghép 2x2 phân tích trực tiếp tác động của 4 yếu tố thời tiết (`weathersit`, `temp`, `hum`, `windspeed`) lên nhu cầu thuê xe (`cnt`). *(Hoàn thành)*
  - `output/step1_eda_06_time_weather_heatmap.png`: Heatmap kết hợp giờ trong ngày (`hr`) × loại thời tiết (`weathersit`) theo lượng xe thuê trung bình. *(Hoàn thành)*
- [x] **Bước 14 [BÁO CÁO (Trang 11–13)]:** Viết phần đầu Mục 4 (Proposed), chèn bảng kiểm tra lỗi (`hum == 0`) và 6 biểu đồ EDA từ `output/` để phân tích đầy đủ 3 khía cạnh: (1) Đặc trưng phân phối & ngoại lai theo mùa; (2) Quy luật biến động theo thời gian; (3) Tác động của điều kiện thời tiết, tương tác `hr × weathersit` và lỗi đa cộng tuyến `temp`–`atemp`. *(Hoàn thành)*

### Giai đoạn 4: Làm sạch, Kỹ thuật đặc trưng và Chuẩn hóa (Mục 4. Proposed phần 2 - Trang 14–16)
- [x] **Bước 15 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** Xóa cột định danh (`instant`, `dteday`), xóa 2 cột `casual` và `registered` để chống Overfitting do rò rỉ dữ liệu, xóa `atemp` và xử lý 22 dòng độ ẩm bằng 0. *(Hoàn thành)*
- [x] **Bước 16 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** Lọc ngoại lai bằng IQR (bảo vệ AdaBoost), mã hóa lượng giác $\sin/\cos$ cho chu kỳ giờ, tháng, thứ, tạo đặc trưng `is_rush_hour`, mã hóa One-Hot cho biến phân loại và chạy kiểm định thống kê `f_regression` (ANOVA F-test) + `mutual_info_regression` vẽ biểu đồ xếp hạng mức độ ảnh hưởng của đặc trưng (`output/step2_prep_01_feature_selection.png`). *(Hoàn thành)*
- [x] **Bước 17 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** Chạy K-Means Clustering phân cụm các khung giờ có hành vi thuê xe tương đồng tạo đặc trưng cụm mới (`hour_cluster`) và dùng PCA giảm chiều xuống 2D để trực quan hóa không gian đặc trưng (`output/step2_prep_02_pca_kmeans.png`). *(Hoàn thành)*
- [x] **Bước 18 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** Loại bỏ 2 cột trực quan hóa (`pca_1`, `pca_2`), mã hóa One-Hot cho `hour_cluster`, chia dữ liệu 80% Train – 20% Test (`random_state=42`), chỉ gọi `.fit()` cho bộ chuẩn hóa `StandardScaler`, `MinMaxScaler`, `RobustScaler` trên `X_train` rồi `.transform()` cho cả hai tập; lưu dữ liệu sạch ra `data/`. *(Hoàn thành)*
- [x] **Bước 19 [BÁO CÁO (Trang 14–16)]:** Viết phần sau Mục 4, vẽ sơ đồ khối 7 bước Học máy và thuyết minh toàn bộ quy trình làm sạch, chọn lọc đặc trưng bằng kiểm định thống kê, tạo đặc trưng chu kỳ/phân cụm và chuẩn hóa chống rò rỉ dữ liệu. *(Hoàn thành)*

### Giai đoạn 5: Huấn luyện, Tinh chỉnh và Đánh giá mô hình (Mục 5. Experiments - Trang 17–23)
- [x] **Bước 20 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Chạy Thí nghiệm 1 đối chứng `LinearRegression`, `KNeighborsRegressor`, `AdaBoostRegressor` và mạng ANN 3 lớp (`MLPRegressor`) trên dữ liệu thô so với dữ liệu đã làm sạch và chuẩn hóa (`output/step3_models_01_raw_vs_clean.png`). *(Hoàn thành)*
- [x] **Bước 21 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Chạy Thí nghiệm 2 với 5-Fold Cross Validation và `GridSearchCV` (`n_jobs=-1`, lưới tham số gọn 4–6 tổ hợp) trên 8 mô hình Hồi quy cốt lõi đại diện đủ các họ thuật toán: `Linear Regression` (Simple/Multiple), `Ridge (L2)`, `Lasso (L1)`, `SVR`, `kNN`, `CART Decision Tree`, `Random Forest` (Bagging), và `XGBoost` (Boosting); vẽ Validation Curve minh họa Bias-Variance (`output/step3_models_02_bias_variance.png`). *(Hoàn thành)*
- [ ] **Bước 22 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Chạy Thí nghiệm 3 phân lớp nhu cầu Cao/Thấp bằng `LogisticRegression`, `GaussianNB`, `SVC`, `DecisionTreeClassifier` và xuất Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC (`output/step3_models_03_classification_roc.png`).
- [ ] **Bước 23 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Xuất bảng tổng hợp `MAE`, `MSE`, `RMSE`, `MBE`, $R^2$, thời gian chạy ra `output/model_comparison.csv`; vẽ biểu đồ thực tế vs. dự đoán (`output/step3_models_04_actual_vs_pred.png`), xếp hạng Feature Importance (`output/step3_models_05_feature_importance.png`) và viết hàm dự đoán thử cho mẫu thời tiết + giờ mới.
- [ ] **Bước 24 [BÁO CÁO (Trang 17–23)]:** Viết Mục 5 (Experiments), chèn 3 bảng thực nghiệm cùng biểu đồ từ `output/`, nhận xét theo 3 tiêu chí (khả năng giải thích, thời gian tính toán, sức mạnh dự đoán) và viết thêm tiểu mục "Thảo luận hạn chế" phân tích giới hạn của dữ liệu UCI Bike Sharing.

### Giai đoạn 6: Kết luận, Tài liệu tham khảo và Xuất PDF (Trang 24–25)
- [ ] **Bước 25 [BÁO CÁO (Trang 24–25)]:** Viết Mục 6 (Conclusion) đề xuất hướng RNN/LSTM, Mục 7 (References) trích dẫn nguồn UCI và 4 sách giáo trình trong đề cương, điền số liệu chốt vào Mục 1 (Abstract), kiểm tra định dạng cỡ chữ 13 ($\le 25$ trang) và xuất 01 file PDF hoàn chỉnh.