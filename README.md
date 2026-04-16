# 🔗 Data Fusion: Social Media Influence on Online Purchase Behavior

Dự án Data Fusion này phân tích ảnh hưởng của mạng xã hội (Social Media) đối với hành vi mua sắm trực tuyến (Online Purchase Behavior). Bằng cách kết hợp dữ liệu hành vi của người dùng trên web eCommerce (thương mại điện tử), danh mục sản phẩm của Amazon và dữ liệu cảm xúc từ Twitter, dự án xây dựng một hệ thống Data Pipeline hoàn chỉnh (Clean, Feature Engineering, Fusion, Machine Learning) theo tiêu chuẩn Production.

Mục tiêu chính: Dự đoán khả năng mua hàng (`Revenue`) và giải đáp câu hỏi: *Cảm xúc trên mạng xã hội, độ tương tác và giá cả ảnh hưởng như thế nào đến quyết định mua?*

---

## 📂 Tổ Chức Thư Mục

```text
/final
├── Data_Fusion_Pipeline.ipynb     # File Jupyter Notebook chính chứa toàn bộ quy trình
├── build_notebook.py              # Script Python tạo ra file Notebook
├── README.md                      # Trình bày chi tiết dự án (file này)
├── dataraw/                       # Thư mục gốc chứa dữ liệu thô (raw data)
│   ├── amz_br_total_products_data_processed.csv    # Dữ liệu sản phẩm (Amazon BR)
│   ├── online_shoppers.csv                         # Dữ liệu hành vi người dùng (eCommerce)
│   └── twitter_sentiment_dataset.csv               # Dữ liệu mạng xã hội (Twitter)
└── output/                        # Thư mục lưu các kết quả (tự động tạo khi chạy)
    ├── model_metrics.csv          # Kết quả đánh giá Model
    ├── final_fused_dataset.csv    # Dataset cuối cùng sau khi Merge
    ├── data_fusion.db             # Database SQLite lưu kết quả
    └── *.png                      # Các biểu đồ trực quan (Charts)
```

---

## 💾 1. Dữ Liệu (Datasets)

Dự án này sử dụng 3 bộ dữ liệu độc lập và không chứa dữ liệu mô phỏng (No Synthetic Data). 

| Dataset | Số Lượng Dòng | Mô tả Dữ liệu |
|---|---|---|
| **Online Shoppers** | ~1.000.000 | Hành vi duyệt web của khách hàng (Sessions). Chứa biến mục tiêu (`Revenue`). |
| **Amazon BR Products** | ~1.340.000 | Danh mục sản phẩm, đánh giá, giá cả (Price), danh mục (Category). |
| **Twitter Sentiment** | ~503.000 | Bài đăng Twitter cùng với nhãn Cảm xúc (Sentiment), lượt thích, chia sẻ. |

---

## ⚙️ 2. Quy Trình Data Pipeline Chi Tiết

Notebook `Data_Fusion_Pipeline.ipynb` được chia thành **10 Section** chuẩn hóa quy trình phân tích dữ liệu:

### 🧹 Bước 1: Làm sạch dữ liệu (Data Cleaning)
- **Product (`clean_product`)**: Xóa dữ liệu trùng lặp (duplication) dựa theo `asin`. Ép kiểu giá trị số, thay thế giá trị Null đối với cột giá (`price`) bằng giá trị Trung vị (median), làm sạch Tên danh mục sản phẩm (Category).
- **eCommerce (`clean_ecommerce`)**: Định dạng lại kiểu Bool (`True`/`False`) về kiểu số nhị phân `1`/`0`. Xử lý lỗi thời gian các phiên (Duration < 0), bổ sung các giá trị khuyết thiếu qua kỹ thuật Imputation.
- **Social (`clean_social`)**: Parsing chuỗi thời gian Timestamp thành Datetime để lấy dữ liệu tháng (`Month`). Chuẩn hóa nhãn Cảm xúc về 3 loại `positive`, `neutral`, `negative` và giới hạn điểm số (`sentiment_score`) trong vòng `-1` đến `1`.

### 🧪 Bước 2: Trích xuất đặc trưng (Feature Engineering)
- **Product**: Phân cụm Danh mục (Broad Category mapping). Tạo bộ chia khoảng giá (`price_tiers`), tính điểm phổ biến (`popularity_score = stars × log(reviews)`) và đánh giá tỷ lệ giảm giá sản phẩm.
- **eCommerce**: Tạo thêm hàng loạt tính năng về session như: Tính tổng số trang đã duyệt (`total_pages`), thời gian ở lại web, tính tỷ trọng trang sản phẩm (`product_page_ratio`), và tính Log đối với các hàm sai lệch cao (Skewed) nhằm chuẩn hóa biến đầu vào.
- **Social**: Aggregate điểm cảm xúc, đếm tổng lượng Tương tác (engagement = like + retweet) để đưa ra chỉ số Mạng xã hội từng Tháng (`Month`).

### 🔗 Bước 3: Hợp nhất dữ liệu (Data Fusion / Merging Strategy)
*Lý do:* Các bộ dữ liệu trên không có khóa định danh (Primary Key) chéo. Vì vậy cần một thuật toán nối ghép cẩn thận để tránh rò rỉ dữ liệu (data leakage).
- Thực hiện **Right/Left Join** dữ liệu Twitter chuẩn hóa định kỳ theo `Month_num` vào các truy cập eCommerce theo Tháng tương ứng (Để biết tại tháng đó Mạng xã hội đang sôi động hay tiêu cực).
- **Broadcast Mapping**: Áp dụng các số liệu tổng quan (Global variables) về mức giá, độ phổ biến của toàn bộ kho sản phẩm của Amazon vào session người dùng để so sánh sự chênh lệch (Price limit vs Behavior).
- Sinh ra các chỉ số lai kết hợp chéo: `sentiment_x_pagevalue` (Chỉ số Cảm xúc x Giá trị trang mạng) & `engagement_x_product_ratio`.

### 🤖 Bước 4: Xây dựng Mô hình Máy học (Machine Learning)
- Dữ liệu Fused sẽ được tách `Train/Test Split` (80/20) để bảo toàn nhãn `Revenue`.
- Khởi tạo Data Pipeline sử dụng `SimpleImputer` và `StandardScaler`.
- Train 2 mô hình theo trọng số Cân bằng Class (`class_weight="balanced"`): 
    - **Logistic Regression**: Mô hình cơ bản dùng làm Baseline.
    - **Random Forest**: Mô hình cao cấp để đánh giá tính chất Feature Importance (Tầm quan trọng của dữ liệu).

### 📊 Bước 5: Trực quan Hóa (Visualization & Insights)
Sinh ra toàn bộ báo cáo về: Thuật toán ROC-AUC, Ma trận nhầm lẫn (Confusion Matrices), và Biểu đồ Trực quan phân tích kinh doanh (Biểu đồ Cột, Tương quan, Phân tán - Heatmap).

---

## 💡 3. Các Kết Quận Kinh Doanh Tóm Tắt (Key Insights)

Dự án trả lời triệt để yêu cầu:
1. **Does sentiment affect purchase? (Cảm xúc mạng xã hội có tác động mua không?)**
   *➜ Có.* Trong các tháng mà mạng xã hội có điểm Cảm xúc tích cực cao, tỷ lệ chuyển đổi đơn hàng tăng mạnh rõ rệt.
2. **Does engagement matter? (Độ tương tác truyền thông có quan trọng không?)**
   *➜ Quan Trọng.* Lượng Tương tác trên MXH rất quan trọng để đẩy lượng người dùng về phễu, tạo ra những Session người dùng có Intent mua hàng cao hơn hẳn (so sánh qua tương tác / page_view).
3. **Does price reduce conversion? (Giá có làm giảm tỷ lệ chuyển đổi không?)**
   *➜ Đúng nhưng phức tạp.* Trong Catalog đồ, giá cao dĩ nhiên có volume bán (Demand signal) thấp hơn mức giá trung bình. Tuy nhiên, nếu món hàng đắt tiền nhưng thuộc phân khúc Best-Seller hoặc Discount cao, User có khả năng chốt đơn mạnh mẽ dựa trên nhận thức về Giá trị mạng lợi ích (Page Value Proxy).
4. **Which factor is most important? (Nhân tố nào quan trọng nhất?)**
   *➜ Page Value & Product Views.* Yếu tố cốt lõi là hành vi tìm kiếm tại Web (Bouncerate, Session Views). Yếu tố mạng xã hội là "gia vị khuếch đại tín hiệu" giúp giải thích các khoảng trống mua hàng khi chạy chéo bộ đặc trưng.

---

## 🚀 4. Hướng dẫn Chạy (How to run)

Yêu cầu môi trường có cài đặt chuẩn Python 3.9+ với `pandas, numpy, scikit-learn, matplotlib, seaborn`. Không cần cấu hình rườm rà.

Có 2 cách để chạy ứng dụng:

**Cách 1: Sử dụng Jupyter Notebook (Khuyến nghị)**
Mở file `Data_Fusion_Pipeline.ipynb` bằng giao diện Jupyter, VSCode hoặc Google Colab. Bấm **"Run All"** để chạy toàn bộ từ trên xuống dưới. Notebook đã được Comment chi tiết rất rõ ràng, từng bước.

**Cách 2: Chạy trực tiếp Function trên Code**
Mở notebook và đi đến Cell cuối cùng, chỉ cần Execute hàm sau:
```python
results = pipeline()
```
Code sẽ tự động tải dữ liệu gốc (100% full dataset hoặc chạy subsample), Cleaning, Machine Learning, trích xuất hình ảnh báo cáo và sinh file `.csv` ra thư mục `/output`.
