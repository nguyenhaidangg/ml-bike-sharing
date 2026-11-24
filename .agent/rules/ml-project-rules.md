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
- **Nhánh Git:** bản chống rò rỉ dữ liệu nằm trên nhánh `fix/zero-leakage-step2-step3` (đã gộp toàn bộ `main` đến bước 25). Không commit/push trực tiếp vào `main`; gộp vào `main` qua Pull Request khi cả nhóm đồng ý.

## 2. QUY TẮC BẮT BUỘC CHO AI AGENT (HARD RULES)
1. **KHÔNG làm kiểu "hộp đen" (Tránh 0 điểm):** Tuyệt đối KHÔNG lấy dữ liệu thô đưa thẳng vào vài tầng mạng CNN hoặc chỉ gọi vài dòng `model.fit()` cơ bản. Trọng tâm là quy trình **7 bước của Học máy cổ điển (Classical ML)**: *Gathering Data -> Understanding & Preparing Data -> Model Selection -> Training -> Evaluation -> Parameter Tuning -> Making Predictions*.
2. **Chống rò rỉ dữ liệu (Zero Data Leakage) – quy tắc quan trọng nhất:**
   - Bắt buộc xóa các cột `instant`, `dteday`, `casual`, `registered` khỏi tập đặc trưng `X` (vì `cnt = casual + registered`).
   - **Chia `Train / Test` (80% / 20%, `random_state=42`) TRƯỚC mọi phép xử lý có học tham số từ dữ liệu.** Phân biệt hai loại phép xử lý:
     + *Không học tham số* (áp dụng độc lập từng dòng, được làm trước khi chia): xóa cột, mã hóa `sin/cos`, tạo `is_rush_hour` theo quy tắc giờ cố định, One-Hot với danh mục cố định.
     + *Có học tham số* (làm sau khi chia, **chỉ `.fit()` / tính trên Train** rồi áp dụng cho Test): trung vị điền khuyết `hum == 0`, ngưỡng IQR, `f_regression` / `mutual_info_regression`, K-Means `hour_cluster`, PCA, các bộ Scaler (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
   - **Không bao giờ dùng `cnt` của tập Test để tạo đặc trưng** (ví dụ hồ sơ lượng thuê theo giờ cho K-Means chỉ tính từ `y_train`).
   - **Không chỉnh sửa biến mục tiêu của tập Test:** `y_test` luôn là `cnt` gốc. Mọi so sánh giữa các mô hình/phiên bản dữ liệu phải dùng **cùng một tập Test và cùng `cnt` gốc**.
   - Tọa độ PCA chỉ để trực quan hóa, không đưa vào tập huấn luyện. `hour_cluster` phải mã hóa One-Hot (`hour_cluster_1`, `hour_cluster_2`).
3. **Làm sạch, Khám phá sâu & Kiểm chứng giả thuyết (Know Your Data):**
   - Kiểm tra và xử lý giá trị bất hợp lệ (22 dòng có độ ẩm `hum == 0`, ngày 10/03/2011) bằng trung vị theo (tháng, giờ) học từ Train.
   - Khám phá trực quan đầy đủ cả 2 trụ cột của đề tài: **Thời gian** (`hr`, `workingday`, `season`) và **Điều kiện thời tiết** (`weathersit`, `temp`, `hum`, `windspeed`) cùng tương tác `hr × weathersit`. EDA được phép thực hiện trên toàn bộ dữ liệu vì chỉ dùng để quan sát, không tạo tham số cho mô hình.
   - Kiểm tra đa cộng tuyến bằng Correlation Heatmap và xóa cột `atemp` (tương quan ~0.99 với `temp` – "rất cao", không gọi là "hoàn hảo").
   - **Ngoại lai theo IQR không đồng nghĩa với dữ liệu sai.** Không mặc định cắt trần/xóa ngoại lai; mọi giả thuyết xử lý (Capping, `log1p`…) phải được **kiểm chứng bằng thực nghiệm** và đánh giá trên `y_test` gốc trước khi đưa vào kết luận.
   - Kiểm định thống kê (`f_regression` + `mutual_info_regression`) dùng để phân tích, diễn giải; không tự ý loại đặc trưng dựa trên bảng xếp hạng.
4. **Chuẩn xuất biểu đồ, số liệu & văn phong báo cáo:**
   - Mọi biểu đồ vẽ bằng `matplotlib`/`seaborn` đều phải lưu tự động vào thư mục `output/` với định dạng `.png`, độ phân giải `dpi=300`, `bbox_inches='tight'`, tiêu đề và nhãn trục tiếng Việt rõ ràng để chèn trực tiếp vào báo cáo PDF.
   - **Không gõ cứng số liệu vào code** (tiêu đề biểu đồ, đường "tối ưu" trên Validation Curve, ngưỡng phân lớp…): luôn lấy từ kết quả tính toán.
   - **Mọi con số trong phần nhận xét notebook và báo cáo phải khớp với output đã chạy và đã lưu trong notebook.** Sau khi sửa code phải chạy lại toàn bộ notebook (Run All) và lưu output trước khi commit.
   - Không ghi chữ "Bước 15", "Bước 16"... vào bên trong nội dung text/markdown của các file `.ipynb`; hãy dùng thuật ngữ học thuật chuẩn của môn học (ví dụ: *giai đoạn tiền xử lý dữ liệu*, *bước kỹ thuật đặc trưng*).
   - Khi chạy mô hình, luôn in bảng kết quả đầy đủ các độ đo theo giáo trình (Hồi quy: `MAE`, `MSE`, `RMSE`, `MBE`, `R2-score`, `Train Time`; Phân lớp: `Confusion Matrix`, `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`).
   - Văn phong khách quan, khiêm tốn: tránh các từ phóng đại ("đanh thép", "tuyệt đối", "bác bỏ hoàn toàn", "phân tách rõ nét" khi số liệu không ủng hộ); nêu rõ hạn chế khi có.
5. **Cách phối hợp với người dùng:**
   - **Bám sát nhưng linh hoạt với lộ trình:** Tuân thủ trình tự các bước theo lộ trình do bạn sinh viên thiết kế để kiểm soát tiến độ và dung lượng báo cáo, KHÔNG tự ý sinh toàn bộ code của tất cả các giai đoạn cùng một lúc. Tuy nhiên, AI phải giữ vai trò người phản biện khoa học: nếu phát hiện bất kỳ bước nào chưa hợp lý về mặt kỹ thuật, thống kê hay thực tế mô hình hóa, AI phải chủ động giải thích, phân tích và cùng bạn tinh chỉnh cho chuẩn xác.
   - Nếu là bước `[LẬP TRÌNH]`: Viết code Python sạch vào đúng file notebook quy định, chạy kiểm tra, lưu biểu đồ/bảng ra `output/` và giải thích ngắn gọn ý nghĩa kết quả bằng tiếng Việt.
   - Nếu là bước `[BÁO CÁO]`: Soạn nội dung học thuật tiếng Việt chuẩn xác (kèm công thức LaTeX, bảng biểu và gợi ý vị trí chèn ảnh từ `output/`), kiểm soát dung lượng bám sát số trang phân bổ để tổng báo cáo không vượt quá 25 trang (cỡ chữ 13). Giọng văn học thuật khiêm tốn theo góc nhìn sinh viên làm báo cáo ("nhóm em", "chúng em").
   - Sau khi hoàn thành mỗi bước, tóm tắt kết quả/file đã tạo, cập nhật trạng thái `[x]` vào checklist và hỏi xác nhận trước khi sang bước tiếp theo. Chỉ đánh dấu `[x]` khi đã có bằng chứng (output đã lưu, file đã tồn tại).

## 3. CẤU TRÚC THƯ MỤC CHUẨN
```text
ml-bike-sharing/
├── .agent/rules/ml-project-rules.md   # File quy tắc & lộ trình này
├── venv/                              # Môi trường ảo Python
├── data/
│   ├── hour.csv, day.csv, Readme.txt  # Dữ liệu gốc UCI
│   ├── hour_cleaned.csv               # Dữ liệu sau tiền xử lý (24 đặc trưng + cnt gốc)
│   ├── X_train.csv, X_test.csv        # Chưa chuẩn hóa (cho mô hình cây)
│   ├── X_train_scaled.csv, X_test_scaled.csv  # StandardScaler fit trên Train
│   ├── y_train.csv, y_test.csv        # cnt GỐC (mục tiêu chính)
│   └── y_train_capped.csv             # cnt Train đã Capping IQR – chỉ dùng cho thí nghiệm kiểm chứng
├── notebooks/
│   ├── step1_eda.ipynb                # Giai đoạn 3 (Bước 11 - 13)
│   ├── step2_prep.ipynb               # Giai đoạn 4 (Bước 15 - 18)
│   └── step3_models.ipynb             # Giai đoạn 5 (Bước 20 - 23)
├── output/                            # Biểu đồ (.png 300dpi) và bảng kết quả (model_comparison.csv)
└── report/                            # bao_cao_do_an_hoc_may.md → xuất 01 file PDF (cỡ chữ 13)
```

## 4. BẢNG PHÂN BỔ DỰ KIẾN 25 TRANG BÁO CÁO (DO BẠN SINH VIÊN THIẾT KẾ)

| Mục báo cáo | Nội dung trọng tâm | Thuật toán & Kỹ thuật cốt lõi | Bước thực hiện | Số trang |
| --- | --- | --- | --- | --- |
| Cover & Content | Trang bìa, mục lục tự động | Chuẩn định dạng cỡ chữ 13 | Bước 1 – 2 | 2 trang |
| 1. Abstract | Tóm tắt bài toán và kết quả | Tổng hợp RMSE, $R^2$, F1-Score | Bước 3, 25 | 1 trang |
| 2. Introduction | Đặt vấn đề, phát biểu bài toán | Classical ML, Regression, Classification | Bước 4 – 5 | 2 trang |
| 3. Related Methods | Cơ sở lý thuyết và toán học | OLS, Ridge/Lasso, kNN, SVM, NB, CART, XGBoost | Bước 6 – 10 | 5 trang |
| 4. Proposed | Khám phá (Thời gian + Thời tiết) & Tiền xử lý chống rò rỉ | EDA (6 bộ biểu đồ), IQR, `f_regression`, Cyclical, K-Means, PCA, Scaler – học trên Train | Bước 11 – 19 | 6 trang |
| 5. Experiments | Thực nghiệm, kiểm chứng giả thuyết, đánh giá & hạn chế | Đối chứng Thô/Sạch, kiểm chứng Capping, 5-Fold CV, Grid Search, MAE, MSE, RMSE, AUC, Thảo luận hạn chế | Bước 20 – 24 | 7 trang |
| 6. Conclusion & 7. References | Kết luận, hướng mở rộng, trích dẫn | Định hướng chia theo thời gian, RNN/LSTM, nguồn UCI & 4 sách giáo trình | Bước 25 | 2 trang |

## 5. LỘ TRÌNH 25 BƯỚC THỰC HIỆN (CHECKLIST TIẾN ĐỘ DO BẠN SINH VIÊN TỰ XÂY DỰNG)

### Giai đoạn 1: Khởi tạo dự án và viết phần Mở đầu (Trang 1–5)
- [x] **Bước 1 [LẬP TRÌNH]:** Khởi tạo thư mục dự án (`data`, `notebooks`, `output`, `report`), môi trường ảo `venv`, cài đặt đầy đủ các thư viện và Extension VS Code. *(Hoàn thành)*
- [ ] **Bước 2 [BÁO CÁO (Trang 1–2)]:** Trình bày trang bìa (tên đề tài, mã lớp `2611COMP171201`, GVHD TS. Ngô Quốc Việt) và mục lục tự động trong file báo cáo cỡ chữ 13. *(Bìa đã có trong `report/bao_cao_do_an_hoc_may.md`; mục lục hiện gõ tay số trang – cần tạo mục lục tự động khi xuất PDF.)*
- [x] **Bước 3 [BÁO CÁO (Trang 3)]:** Viết Mục 1 (Abstract) dài 1 trang tóm tắt bài toán, quy trình tiền xử lý chống rò rỉ và kết quả tốt nhất. *(Đã có; số liệu đã cập nhật theo kết quả chạy lại.)*
- [x] **Bước 4 [BÁO CÁO (Trang 4)]:** Viết nửa đầu Mục 2 (Introduction) nêu vai trò của Học máy trong AI và ứng dụng Hồi quy để dự báo xu hướng cũng như đánh giá tác động của điều kiện thời tiết và thời gian lên nhu cầu thuê xe. *(Đã có trong báo cáo.)*
- [x] **Bước 5 [BÁO CÁO (Trang 5)]:** Viết nửa sau Mục 2 phát biểu bài toán Học có giám sát với biến độc lập $X$, biến phụ thuộc $Y$ và lý giải việc dùng Học máy cổ điển thay vì mạng CNN hộp đen. *(Đã có trong báo cáo.)*

### Giai đoạn 2: Viết cơ sở lý thuyết cốt lõi (Mục 3. Related Methods - Trang 6–10)
- [x] **Bước 6 [BÁO CÁO (Trang 6)]:** Mở đầu Mục 3 đối chiếu Học có giám sát với Không giám sát, mô hình Parametric với Non-parametric và Discriminative với Generative. *(Đã có trong báo cáo.)*
- [x] **Bước 7 [BÁO CÁO (Trang 7)]:** Trình bày sự đánh đổi Bias-Variance qua công thức $E(MSE) = noise^2 + bias^2 + variance$, lỗi Underfitting/Overfitting và điều chuẩn L1 Lasso, L2 Ridge. *(Đã có trong báo cáo.)*
- [x] **Bước 8 [BÁO CÁO (Trang 8)]:** Trình bày công thức Bình phương tối thiểu tìm $\hat{\beta}_0, \hat{\beta}_1$ của Linear Regression và hàm Sigmoid, Logit, Newton-Raphson của Logistic Regression. *(Đã có trong báo cáo.)*
- [x] **Bước 9 [BÁO CÁO (Trang 9)]:** Viết nguyên lý khoảng cách của kNN, siêu phẳng phân cách kèm hàm mất mát Hinge Loss của SVM và ước lượng $h_{MAP}$ có làm mượt Laplace của Naïve Bayes. *(Đã có trong báo cáo.)*
- [x] **Bước 10 [BÁO CÁO (Trang 10)]:** Giải thích cây quyết định CART (chia nhánh theo SDR, Gini, Entropy, IG, GR), đối chiếu Bagging (Random Forest) và Boosting (AdaBoost, XGBoost). *(Đã có trong báo cáo.)*

### Giai đoạn 3: Khám phá và Trực quan hóa dữ liệu Thời gian & Thời tiết (Mục 4. Proposed phần 1 - Trang 11–13)
- [x] **Bước 11 [LẬP TRÌNH (`notebooks/step1_eda.ipynb`)]:** Dùng hàm `pd.read_csv()` đọc dữ liệu `data/hour.csv`, kiểm tra giá trị khuyết và lọc ra 22 dòng có độ ẩm `hum == 0` bất hợp lệ theo nguyên tắc *Know Your Data*. *(Hoàn thành)*
- [x] **Bước 12 [LẬP TRÌNH (`notebooks/step1_eda.ipynb`)]:** Vẽ & lưu biểu đồ Histogram của `cnt` kèm đường kẻ dọc `Mean` (đỏ) và `Median` (xanh đứt nét) cùng khảo sát biến đổi `log1p` (`output/step1_eda_01_histogram_log.png`), và vẽ Box plot kèm bảng thống kê ngoại lai theo 4 mùa (`output/step1_eda_02_boxplot_outliers.png`). *(Hoàn thành. Ghi chú: `log1p` chỉ là khảo sát; các mô hình ở Giai đoạn 5 huấn luyện và đánh giá trên `cnt` gốc.)*
- [x] **Bước 13 [LẬP TRÌNH (`notebooks/step1_eda.ipynb`)]:** Vẽ & lưu trọn vẹn 4 bộ biểu đồ khám phá Thời gian, Thời tiết & Đa cộng tuyến:
  - `output/step1_eda_03_time_trends.png`: Fever chart theo tháng/năm & Line/Area chart theo 24h phân tách `workingday` (0 vs 1).
  - `output/step1_eda_04_scatter_heatmap.png`: Scatter plot (`temp` vs `atemp`) và Correlation Heatmap chỉ ra đa cộng tuyến (`0.99`) và rò rỉ dữ liệu (`casual`, `registered`).
  - `output/step1_eda_05_weather_impact.png`: Hình ghép 2x2 phân tích tác động của 4 yếu tố thời tiết (`weathersit`, `temp`, `hum`, `windspeed`) lên `cnt`.
  - `output/step1_eda_06_time_weather_heatmap.png`: Heatmap giờ trong ngày (`hr`) × loại thời tiết (`weathersit`) theo lượng xe thuê trung bình. *(Hoàn thành; ảnh đã có trong `output/`. Một số ô code vẽ biểu đồ trong notebook chưa lưu output – nên Run All và lưu lại để làm bằng chứng.)*
- [x] **Bước 14 [BÁO CÁO (Trang 11–13)]:** Viết phần đầu Mục 4 (Proposed), chèn bảng kiểm tra lỗi (`hum == 0`) và 6 biểu đồ EDA để phân tích 3 khía cạnh: (1) Phân phối & ngoại lai theo mùa – nêu rõ ngoại lai IQR là nhu cầu thực tế, không phải lỗi; (2) Quy luật biến động theo thời gian; (3) Tác động của thời tiết, tương tác `hr × weathersit` và đa cộng tuyến `temp`–`atemp`. *(Hoàn thành, đã chỉnh câu chữ.)*

### Giai đoạn 4: Làm sạch, Kỹ thuật đặc trưng và Chuẩn hóa chống rò rỉ (Mục 4. Proposed phần 2 - Trang 14–16)
- [x] **Bước 15 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** Xóa cột định danh (`instant`, `dteday`), xóa `casual` và `registered` (rò rỉ dữ liệu), xóa `atemp` (đa cộng tuyến). Mã hóa lượng giác `sin/cos` cho giờ, tháng, thứ; tạo `is_rush_hour` (ngày làm việc, giờ 7–9h và 17–19h; 2,983 giờ = 17.2%); One-Hot `season`, `weathersit` với danh mục cố định `{1,2,3,4}` và `drop_first=True`. *(Hoàn thành – các phép biến đổi không học tham số, làm trước khi chia.)*
- [x] **Bước 16 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** Chia dữ liệu 80% Train – 20% Test (`random_state=42`). Sau đó, **chỉ học trên Train**: điền 22 dòng `hum == 0` (18 Train, 4 Test) bằng trung vị (tháng, giờ); tính ngưỡng IQR theo mùa (370 điểm vượt ngưỡng = 2.66% Train), **giữ `cnt` gốc** và lưu riêng `y_train_capped.csv` để kiểm chứng; chạy `f_regression` + `mutual_info_regression` vẽ biểu đồ xếp hạng (`output/step2_prep_01_feature_selection.png`). *(Hoàn thành)*
- [x] **Bước 17 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** K-Means ($k=3$) phân cụm 24 khung giờ theo hồ sơ lượng thuê **tính từ tập Train** → `hour_cluster` (Cụm 0: 0–6h, 22–23h; Cụm 1: 7h, 9–16h, 19–21h; Cụm 2: 8h, 17–18h), áp bảng giờ → cụm cho cả Train/Test; PCA fit trên Train chỉ để trực quan hóa (PC1+PC2 ≈ 25.3% phương sai) (`output/step2_prep_02_pca_kmeans.png`). *(Hoàn thành)*
- [x] **Bước 18 [LẬP TRÌNH (`notebooks/step2_prep.ipynb`)]:** Mã hóa One-Hot `hour_cluster` (`hour_cluster_1`, `hour_cluster_2`) → **24 đặc trưng**; chỉ `.fit()` `StandardScaler`, `MinMaxScaler`, `RobustScaler` trên `X_train` rồi `.transform()` cho cả hai tập; lưu 8 file CSV ra `data/` (xem Mục 3). *(Hoàn thành)*
- [x] **Bước 19 [BÁO CÁO (Trang 14–16)]:** Viết phần sau Mục 4: sơ đồ khối 7 bước (chia Train/Test trước các bước có học tham số), thuyết minh làm sạch, thứ tự xử lý chống rò rỉ, ngưỡng IQR học trên Train, kiểm định thống kê, đặc trưng chu kỳ/giờ cao điểm/phân cụm và chuẩn hóa; nêu hạn chế chia ngẫu nhiên trên chuỗi thời gian. *(Hoàn thành, đã cập nhật.)*

### Giai đoạn 5: Huấn luyện, Tinh chỉnh và Đánh giá mô hình (Mục 5. Experiments - Trang 17–23)
- [x] **Bước 20 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Thí nghiệm 1 đối chứng `LinearRegression`, `KNeighborsRegressor`, `AdaBoostRegressor`, `MLPRegressor` (3 lớp) trên dữ liệu thô vs. dữ liệu đã xử lý, **cùng tập Test và cùng `cnt` gốc** (`output/step3_models_01_raw_vs_clean.png`). Bổ sung **thí nghiệm kiểm chứng Capping**: huấn luyện trên `y_train` gốc vs. `y_train_capped`, đánh giá trên `y_test` gốc. *(Hoàn thành. Kết quả: Linear Regression $R^2$ 0.388 → 0.744; kNN giảm nhẹ; Capping không cải thiện, làm MLP kém đi → giữ `cnt` gốc.)*
- [x] **Bước 21 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Thí nghiệm 2 với 5-Fold CV và `GridSearchCV` (`n_jobs=-1`, lưới gọn 4–8 tổ hợp) trên **9 cấu hình thuộc 8 họ thuật toán**: Linear Regression (Simple/Multiple), Ridge (L2), Lasso (L1), SVR, kNN, CART, Random Forest (Bagging), XGBoost (Boosting); Validation Curve Bias-Variance với đường tối ưu tính tự động (`output/step3_models_02_bias_variance.png`). *(Hoàn thành. Tốt nhất: XGBoost $R^2 = 0.9538$, RMSE = 38.27.)*
- [x] **Bước 22 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Thí nghiệm 3 phân lớp nhu cầu Cao/Thấp (ngưỡng trung vị `y_train` = 143 xe/giờ) bằng `LogisticRegression`, `GaussianNB`, `SVC`, `DecisionTreeClassifier`; xuất Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC (`output/step3_models_03_classification_roc.png`). *(Hoàn thành. Tốt nhất: SVC AUC = 0.9874, F1 = 0.9421.)*
- [x] **Bước 23 [LẬP TRÌNH (`notebooks/step3_models.ipynb`)]:** Xuất `output/model_comparison.csv` trực tiếp từ kết quả Thí nghiệm 2; vẽ Thực tế vs. Dự đoán dùng đúng XGBoost tốt nhất từ GridSearch (`output/step3_models_04_actual_vs_pred.png` – ghi rõ 120 mẫu Test không liên tục theo thời gian); Feature Importance (`output/step3_models_05_feature_importance.png`); hàm `predict_bike_demand` khớp 24 đặc trưng (bảng giờ → cụm lấy từ Train, ngưỡng CAO/THẤP = trung vị). *(Hoàn thành)*
- [x] **Bước 24 [BÁO CÁO (Trang 17–23)]:** Viết Mục 5 (Experiments): bảng Thí nghiệm 1 + bảng kiểm chứng Capping, bảng Thí nghiệm 2, bảng Thí nghiệm 3 cùng biểu đồ; nhận xét theo 3 tiêu chí (khả năng giải thích, thời gian tính toán, sức mạnh dự đoán); tiểu mục "Thảo luận hạn chế" gồm cả hạn chế chia ngẫu nhiên trên chuỗi thời gian và việc dùng thời tiết đo thực tế thay vì thời tiết dự báo. *(Hoàn thành, số liệu đã khớp kết quả chạy lại.)*

### Giai đoạn 6: Kết luận, Tài liệu tham khảo và Xuất PDF (Trang 24–25)
- [ ] **Bước 25 [BÁO CÁO (Trang 24–25)]:** Viết Mục 6 (Conclusion) đề xuất hướng mở rộng (thí nghiệm chia theo thời gian / `TimeSeriesSplit`, RNN/LSTM, dữ liệu dự báo thời tiết), Mục 7 (References) trích dẫn nguồn UCI và 4 sách giáo trình, điền số liệu chốt vào Mục 1 (Abstract), kiểm tra định dạng cỡ chữ 13 ($\le 25$ trang) và xuất 01 file PDF hoàn chỉnh. *(Mục 6, 7 và Abstract đã viết và cập nhật số liệu; **chưa xuất PDF, chưa kiểm tra giới hạn 25 trang**.)*

## 6. NHẬT KÝ ĐIỀU CHỈNH KỊCH BẢN (SO VỚI BẢN ĐẦU)
| Hạng mục | Bản đầu | Bản hiện tại | Lý do |
| --- | --- | --- | --- |
| Thời điểm chia Train/Test | Sau K-Means/PCA (Bước 18) | Ngay sau các phép biến đổi không học tham số (Bước 16) | Chống rò rỉ dữ liệu |
| Ngoại lai `cnt` | Capping trên toàn bộ dữ liệu (kể cả Test) | Giữ `cnt` gốc; ngưỡng IQR học từ Train; Capping chỉ để kiểm chứng | Không sửa đáp án Test; thực nghiệm cho thấy Capping không giúp |
| `hour_cluster` | Tính từ `cnt` toàn bộ dữ liệu, để dạng số 0/1/2 | Tính từ `y_train`, One-Hot 2 cột | Rò rỉ biến mục tiêu; biến danh nghĩa |
| Điền `hum == 0`, F-test/MI, PCA | Tính trên toàn bộ dữ liệu | Chỉ tính trên Train | Chống rò rỉ dữ liệu |
| Số đặc trưng | 23 | 24 | One-Hot `hour_cluster` |
| Thí nghiệm 1 | Thô (`cnt` gốc) vs. Sạch (`cnt` đã cắt trần) | Cùng tập Test, cùng `cnt` gốc + thí nghiệm kiểm chứng Capping | So sánh công bằng |
| Số mô hình hồi quy | "8 mô hình" | 9 cấu hình thuộc 8 họ thuật toán | Ghi đúng số lượng |
| Kết quả tốt nhất | XGBoost $R^2$ 0.9559 (trên dữ liệu rò rỉ) | XGBoost $R^2$ 0.9538, RMSE 38.27 | Số liệu sau khi sửa rò rỉ |
