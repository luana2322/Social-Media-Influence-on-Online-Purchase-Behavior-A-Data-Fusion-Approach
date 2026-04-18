# 🔗 Data Fusion: Social Media Influence on Online Purchase Behavior

Dự án **Data Fusion** phân tích ảnh hưởng của mạng xã hội (Social Media) đối với hành vi mua sắm trực tuyến (Online Purchase Behavior). Bằng cách kết hợp dữ liệu hành vi người dùng eCommerce, danh mục sản phẩm Amazon BR và dữ liệu cảm xúc Twitter, dự án xây dựng một hệ thống **Data Pipeline** hoàn chỉnh (Clean → Feature Engineering → Fusion → Machine Learning) sử dụng **100% dữ liệu thực tế** (No Synthetic Data).

**Mục tiêu chính:** Dự đoán khả năng mua hàng (`Revenue`) và giải đáp 6 câu hỏi nghiên cứu: *Cảm xúc MXH, độ tương tác và giá cả ảnh hưởng như thế nào đến quyết định mua?*

---

## 📂 Tổ Chức Thư Mục

```text
/final
├── Data_Fusion_Project.ipynb          # ⭐ Notebook chính — toàn bộ pipeline (23 cells)
├── Data_Fusion_Pipeline.ipynb         # Notebook pipeline production-grade (tham khảo)
├── build_notebook.py                  # Script tạo Data_Fusion_Pipeline.ipynb
├── README.md                          # Tài liệu dự án (file này)
├── dataraw/                           # Dữ liệu thô (raw data) — KHÔNG chỉnh sửa
│   ├── amz_br_total_products_data_processed.csv   # Amazon BR: sản phẩm, giá, đánh giá
│   ├── online_shoppers.csv                        # eCommerce: hành vi phiên duyệt web
│   └── twitter_sentiment_dataset.csv              # Twitter: cảm xúc, tương tác
└── output/                            # Kết quả tự động sinh khi chạy notebook
    ├── final_fused_dataset.csv        # Dataset sau khi Fusion (eCommerce + Social + Product)
    ├── model_metrics.csv              # ROC-AUC, Accuracy của các model ML
    ├── data_fusion.db                 # SQLite database lưu fused dataset
    ├── social_category_analysis.csv  # Phân tích mentions theo danh mục (Twitter)
    ├── amazon_category_analysis.csv  # Phân tích hiệu suất danh mục (Amazon)
    └── research_questions_analysis.png  # Biểu đồ tổng hợp trả lời 6 câu hỏi
```

---

## 💾 1. Dữ Liệu (Datasets)

Dự án sử dụng **3 bộ dữ liệu thực tế** hoàn toàn độc lập — không có shared primary key.

| Dataset | File | Kích thước | Mô tả |
|---|---|---|---|
| **Online Shoppers** | `online_shoppers.csv` | ~1.000.000 dòng | Hành vi duyệt web (Sessions), biến mục tiêu `Revenue` |
| **Amazon BR Products** | `amz_br_total_products_data_processed.csv` | ~1.340.000 dòng | Danh mục sản phẩm, giá (`price`), sao (`stars`), đánh giá |
| **Twitter Sentiment** | `twitter_sentiment_dataset.csv` | ~503.000 dòng | Bài đăng Twitter, nhãn cảm xúc, `like_count`, `retweet_count` |

> **Lưu ý:** Notebook mặc định chạy `SAMPLE_SIZE = 50,000` dòng/dataset để cân bằng hiệu năng. Tăng giá trị này để phân tích toàn bộ dữ liệu.

---

## ❓ 2. Sáu Câu Hỏi Nghiên Cứu (Research Questions)

| # | Câu hỏi | Phương pháp trả lời |
|---|---|---|
| **Q1** | Cảm xúc MXH có ảnh hưởng đến hành vi mua hàng không? | Scatter plot: `avg_sentiment` (Twitter/tháng) × purchase rate |
| **Q2** | Độ tương tác (likes + retweets) có tăng xác suất mua không? | Bar chart: engagement quartile × purchase rate |
| **Q3** | Danh mục sản phẩm nào được nhắc đến nhiều nhất trên MXH? | Phân tích keyword từ Twitter text → count by category |
| **Q4** | Tỷ lệ mua có khác nhau theo danh mục không? | Bar chart: session value tier (PageValues proxy) × Revenue |
| **Q5** | Giá cả có ảnh hưởng đến quyết định mua không? | Bar chart: price-value tier × purchase rate |
| **Q6** | Có thể dự đoán hành vi mua từ dữ liệu tổng hợp không? | ROC-AUC curve: Logistic Regression vs Random Forest |
| **Bonus** | Nhân tố quan trọng nhất là gì? | Random Forest Feature Importance ranking |

---

## ⚙️ 3. Quy Trình Data Pipeline Chi Tiết

`Data_Fusion_Project.ipynb` được chia thành **13 Section** với 23 cells:

### 🧹 Bước 1: Làm sạch dữ liệu (Data Cleaning)
- **eCommerce** (`clean_ecommerce`): Chuyển `Revenue` từ string `'True'/'False'` → `1/0`; clip duration âm; impute median; map `Month` string → số nguyên; encode `VisitorType`.
- **Social** (`clean_social`): Parse ISO datetime → trích xuất `Month_num`; chuẩn hóa nhãn sentiment về `positive / neutral / negative`; clip `sentiment_score ∈ [-1, 1]`; fill missing engagement.
- **Product** (`clean_product`): Cast numeric columns; loại bỏ giá ≤ 0; dedup theo `asin`; map danh mục tiếng Bồ Đào Nha → nhãn tiếng Anh rộng (Electronics, Gaming, Sports, Clothing, Beauty, Home, Books, Baby).

### 🧪 Bước 2: Trích xuất đặc trưng (Feature Engineering)
- **Social**: `engagement = like + retweet + reply`; `sentiment_norm ∈ [0,1]`; phân loại tweet theo category qua keyword matching.
- **Product**: `popularity_score = stars × log(reviews+1)`; `discount_ratio`; `price_tier` (quartile cut).
- **eCommerce**: `total_pages`, `product_page_ratio`, `session_intensity`; log-transform các biến skewed (`PageValues`, `ProductRelated_Duration`).

### 🔗 Bước 3: Hợp nhất dữ liệu (Data Fusion)

Vì **không có shared primary key** giữa 3 dataset, chiến lược fusion gồm 3 pha:

| Pha | Nguồn | Đích | Key | Kiểu Join |
|---|---|---|---|---|
| **1** | Twitter (aggregate/tháng) | eCommerce sessions | `Month_num` | LEFT JOIN |
| **2** | Amazon (global stats) | Fused dataset | — | BROADCAST (scalar constants) |
| **3** | Cross-features | Fused dataset | — | Derived columns |

**Cross-features được tạo ra:**
- `sentiment_x_pagevalue` = `avg_sentiment × PageValues`
- `engagement_x_product_ratio` = `engagement_norm × product_page_ratio`
- `pagevalue_vs_global_price` = `PageValues / global_avg_price`

### 🤖 Bước 4: Machine Learning
- **Train/Test Split**: 80/20, stratified theo `Revenue`.
- **Pipeline**: `SimpleImputer(median)` → `StandardScaler` → Classifier.
- **Model 1 — Logistic Regression**: Baseline interpretable, `class_weight='balanced'`.
- **Model 2 — Random Forest**: 150 trees, `max_depth=10`, `class_weight='balanced'`; cung cấp Feature Importance ranking.

### 📊 Bước 5: Trực quan hóa & Kết quả
Sinh **8 biểu đồ** trên 1 figure duy nhất, mỗi plot trả lời trực tiếp 1 câu hỏi nghiên cứu + Bonus + Correlation Heatmap. Lưu ra `output/research_questions_analysis.png`.

---

## 💡 4. Kết Quả Chính (Key Insights)

| Câu hỏi | Câu trả lời |
|---|---|
| **Q1: Sentiment → Purchase?** | ✅ **Có.** Tháng có sentiment tích cực cao → tỷ lệ chuyển đổi tăng rõ rệt. |
| **Q2: Engagement → Purchase?** | ✅ **Có.** Tháng có engagement cao → nhiều session với intent mua hàng hơn. |
| **Q3: Category nào hot nhất?** | 📢 **Electronics, Gaming, Sports, Clothing** — nhóm shareability cao nhất. |
| **Q4: Purchase rate theo category?** | ✅ **Có khác biệt lớn.** Session Premium (PageValues cao) convert gấp nhiều lần Basic. |
| **Q5: Giá ảnh hưởng?** | ✅ **Có, phi tuyến.** Giá rất cao gây friction, nhưng mid-high tier vẫn convert tốt nếu PageValues cao. |
| **Q6: ML dự đoán được không?** | ✅ **Được.** Cả 2 model vượt baselines. Random Forest đạt ROC-AUC cao hơn nhờ cross-features từ fusion. |
| **Bonus: Factor quan trọng nhất?** | 🏆 **`PageValues`** > `BounceRates` > `ProductRelated`. Cross-feature `sentiment_x_pagevalue` là signal fusion quan trọng nhất. |

---

## 🚀 5. Hướng Dẫn Chạy (How to Run)

**Yêu cầu:** Python 3.9+ với các thư viện: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Cách 1: Chạy từng bước (Khuyến nghị)
Mở `Data_Fusion_Project.ipynb` bằng **Jupyter**, **VSCode** hoặc **Google Colab** → Bấm **Run All**.
Notebook đã được comment chi tiết từng bước, từ Load → Clean → Feature Engineering → Fusion → ML → Visualisation.

### Cách 2: One-click Pipeline
Đi đến **Cell cuối cùng** và chạy:
```python
result = run_full_pipeline(sample_size=50_000)
```
Pipeline sẽ tự động: Load dữ liệu thực → Clean → Feature Engineering → Fusion → Lưu CSV/SQLite ra `output/`.

### Điều chỉnh Sample Size
Sửa hằng số ở **Cell 4** (Setup):
```python
SAMPLE_SIZE = 50_000   # tăng lên 200_000 hoặc None để dùng toàn bộ dữ liệu
```

---

## 🔬 6. Giới hạn & Hướng Phát Triển

**Giới hạn hiện tại:**
- Join theo tháng là thô — mất variance nội-tháng của Twitter.
- Gán danh mục cho eCommerce session qua `PageValues` là proxy, không phải exact mapping.
- Twitter dataset chủ yếu về cybersecurity/tech; dataset consumer-focused sẽ cho signal mạnh hơn.

**Hướng phát triển tiếp theo:**
- **Real-time fusion**: Stream Twitter qua Kafka → join theo ngày/giờ thay vì tháng.
- **NLP nâng cao**: Dùng BERTweet map tweet sentiment → product SKU (row-level join).
- **Model cải tiến**: XGBoost / LightGBM thường vượt Random Forest trên tabular fusion tasks.
- **Causal Inference**: Áp dụng Difference-in-Differences để xác định quan hệ nhân-quả (không chỉ tương quan).
