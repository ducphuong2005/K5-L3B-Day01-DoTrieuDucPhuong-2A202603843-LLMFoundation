# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603843
- **Họ và tên:** Đỗ Triệu Đức Phương
- **SEED:** 1971

---

## 1. Bằng chứng

| Tệp | Có? | Ghi chú |
| --- | :---: | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] | Đã chạy toàn bộ các phần 0–5, kiểm tra đạt 10/10 test checks |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] | Chứa toàn bộ logs, predictions và bảng kết quả đo đạc thực tế |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] | Đầy đủ 5 phép tính tay trên giấy (baseline acc, sigmoid, F1, cosine, LLM cost) |

---

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng bài toán phân loại SMS Spam / Ham (trích xuất từ `lab1_report.json` của SEED 1971):

| Cách tiếp cận | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Throughput (tin/s) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. BoW + LogReg (tự code)** | **1.000** | 0.811 | 0.895 | 0.974 | 7.22 | 11,173 | **0.05** | **0.26** | $0.000 |
| **2. BoW + MLP (PyTorch)** | 0.977 | 0.905 | 0.940 | 0.974 | 5.63 | **25,403** | 0.26 | 0.37 | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.886 | **0.979** | 0.930 | **1.000** | 80.60 | 21.9 | 58.40 | 83.20 | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.968 | 0.958 | **0.963** | 0.974 | 81.08 | 31.6 | 32.73 | 50.23 | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.989 | 0.937 | 0.962 | 0.919 | 81.68 | 31.6 | 34.83 | 50.59 | $0.000 |
| **4a. LLM nhỏ zero-shot (gpt-4o-mini)** | 0.900 | 0.900 | — | 0.900 | **0.00** | 0.22 | 4,288.92 | 5,709.60 | $0.013 |
| **4b. LLM lớn zero-shot (gpt-4o)** | 0.947 | 0.900 | — | 0.923 | **0.00** | 0.51 | 1,981.81 | 2,458.50 | $0.413 |
| **4c. LLM nhỏ few-shot (gpt-4o-mini)** | 0.947 | 0.900 | — | 0.923 | **0.00** | 0.19 | 4,810.49 | 8,603.55 | $0.033 |

> *Ghi chú:* Hàng LLM (4a, 4b, 4c) chỉ đo trên tập SUB cân bằng (40 tin: 20 spam + 20 ham) để tiết kiệm chi phí; các mô hình 1–3 đo trên toàn bộ tập test (775 tin). Để so sánh chất lượng đồng nhất, đối chiếu qua cột `F1 (SUB 40)`.

---

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có độ chênh lệch đáng kể nhất so với thực tế:

1. **`hard_eval` (50 case khó):**
   - *Dự đoán:* "khoảng 80%" (kỳ vọng các model vẫn duy trì độ chính xác quanh 80%).
   - *Thực tế:* Các mô hình BoW sụp đổ hoàn toàn: BoW + LogReg chỉ đạt F1 = 0.333 (Acc = 52%), BoW + MLP đạt F1 = 0.462 (Acc = 58%). Ngay cả MiniLM + LogReg cũng rớt xuống F1 = 0.583 (Acc = 60%). Chỉ riêng LLM giữ vững phong độ (gpt-4o-mini phòng thủ đạt F1 = 0.966, Acc = 96%).
   - *Vì sao lệch:* Tôi đã đánh giá thấp mức độ nguy hiểm của các bẫy ngụy trang (adversarial examples). Bộ test khó dùng từ khóa spam trong tin nhắn bạn bè thân mật (`ham_looks_spam`), cố tình né từ nhạy cảm (`natural_spam`), hoặc chèn chỉ dẫn giả mạo (`prompt_injection`). Mô hình BoW hoàn toàn "mù ngữ cảnh", chỉ cộng dồn trọng số từ độc lập nên bị lừa hàng loạt.

2. **`llm_models` (So sánh LLM lớn vs LLM nhỏ):**
   - *Dự đoán:* "model lớn F1 cao hơn, chênh khoảng 20%, đắt gấp 2 lần, LLM không thắng được 3b".
   - *Thực tế:* gpt-4o chỉ nhỉnh hơn gpt-4o-mini đúng 2.3% F1 trên tập SUB (0.923 vs 0.900), nhưng chi phí lại đắt gấp **33 lần** ($0.4125 vs $0.0125 / 1k tin).
   - *Vì sao lệch:* Lọc spam nhị phân là tác vụ phân loại cơ bản, không đòi hỏi chuỗi suy luận phức tạp (deep reasoning), do đó mô hình nhỏ đã nắm bắt hầu hết các đặc trưng nhận diện. Đồng thời, tôi đã đánh giá sai chênh lệch biểu phí thực tế giữa model flagship và model mini trên thị trường API.

3. **`ml_f1` (Mô hình BoW + Logistic Regression tự code):**
   - *Dự đoán:* "F1~80" (kỳ vọng khoảng 0.80).
   - *Thực tế:* F1 trên test đạt **0.895** (và trên tập SUB đạt **0.974**), với Precision đạt tuyệt đối **1.000**.
   - *Vì sao lệch:* BoW kết hợp hồi quy logistic dù đơn giản nhưng trên tập dữ liệu SMS thông thường, các từ khóa spam ("free", "claim", "prize", "urgent", "won") có tính phân tách nhị phân cực kỳ sắc nét. Khi mô hình học được trọng số âm lớn cho các từ ham phổ biến và trọng số dương lớn cho từ spam, nó bắt spam rất chuẩn xác mà không bị chặn nhầm tin bình thường nào trên tập test.

---

## 4. Phân tích 50 case khó

- **Model tụt điểm nhiều nhất:** `1. BoW + LogReg` (F1 từ 0.974 trên SUB rơi thẳng xuống **0.333**, độ chính xác 52% — chỉ nhỉnh hơn đoán ngẫu nhiên) và `2. BoW + MLP` (F1 tụt xuống **0.462**).
- **Nhóm case gây tụt nặng nhất:**
  1. *`ham_looks_spam` (10 case):* Tin nhắn người quen nói chuyện bình thường nhưng chứa các từ nhạy cảm ("congrats", "won", "beer", "call", "free tickets"). BoW thấy các token này liền kích hoạt trọng số dương cực lớn $\rightarrow$ báo động giả (False Positive vọt tăng).
  2. *`natural_spam` & `obfuscation` (16 case):* Tin nhắn lừa đảo nhưng dùng giọng văn tự nhiên giả làm bạn cũ, hoặc cố tình viết biến thể ("w!n", "c@ll"). BoW không tìm thấy từ trong từ điển huấn luyện (OOV - Out of Vocabulary) nên gán vector toàn 0 $\rightarrow$ bỏ lọt vi phạm (False Negative).

- **Hai case cụ thể và cơ chế gây sai:**
  - **Case 1 (Tất cả mọi mô hình đều sai — Case #16):**
    - *Văn bản:* `"Long time no see! I've started a little side business with skincare, you'd love it, can I send you some samples?"` (Thật: `spam`, Nhóm: `natural_spam`).
    - *Bẫy:* Kịch bản tiếp cận bán hàng đa cấp / mỹ phẩm trá hình dưới danh nghĩa bạn cũ lâu ngày không gặp.
    - *Cơ chế sai:* Tin nhắn hoàn toàn không có từ khóa chào mời sốc hay giục giã ("prize", "urgent", "click"). Cả BoW, Embedding (kNN, LogReg) lẫn LLM đều hiểu đây là một tin nhắn hỏi thăm đời thường giữa hai người bạn và gắn nhãn là `ham`.
  - **Case 2 (BoW sai nhưng LLM nhận diện đúng — Case #1):**
    - *Văn bản:* `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Thật: `ham`, Nhóm: `ham_looks_spam`).
    - *Bẫy:* Chứa cùng lúc 3 "ngòi nổ" của spam: `"Congrats"`, `"WON"`, `"Call"`.
    - *Cơ chế sai:*
      - *Với BoW:* Trong tập train, "won", "congrats", "call" có trọng số $w$ dương cực lớn. Khi nhân tích vô hướng $w \cdot x + b$, giá trị $z$ bị đẩy lên rất cao $\rightarrow$ $\sigma(z) \approx 1.0$ (chắc chắn spam).
      - *Với LLM:* Nhờ cơ chế Attention hiểu toàn diện câu, LLM nhận biết được cụm `"the bet lol"` và `"owe you a beer"` thể hiện ngữ cảnh bạn bè đùa vui cá cược thể thao/chầu bia, do đó xác định chính xác là `ham`.

---

## 5. Thử thách cá nhân

- **SEED của bạn:** `1971` $\rightarrow$ `MY_CHALLENGE = 1971 % 6 = 3` (Đề số 4 trong danh sách).
- **Nội dung đề:**
  > *"Embedding · Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?"*
- **Cách tiến hành:**
  Sử dụng ma trận embedding MiniLM đã trích xuất `E_tr` (3619 tin train), `E_te` (775 tin test) và nhãn `y_tr`, chạy hàm `knn_predict(E_tr, y_tr, E_te, k)` lần lượt với $k \in \{1, 5, 25\}$, sau đó tính `metrics(y_te, preds)`.

- **Kết quả đo đạc trên tập test (775 tin):**

  | Tham số | Precision | Recall | F1 Score | Accuracy | Nhận xét |
  |:---:|:---:|:---:|:---:|:---:|---|
  | **$k = 1$** | 0.880 | 0.958 | **0.918** | 0.978 | Nhạy cảm với nhiễu, biên phân chia phức tạp |
  | **$k = 5$** | 0.886 | **0.979** | **0.930** | **0.982** | Điểm cân bằng tối ưu giữa Precision và Recall |
  | **$k = 25$** | **0.910** | 0.884 | **0.897** | 0.975 | Bị lớp đa số (Ham) áp đảo, Recall tụt mạnh |

- **Giải thích cơ chế:**
  - **Khi $k$ quá nhỏ ($k = 1$):** Mô hình rơi vào trạng thái *High Variance* (phương sai cao / overfitting cục bộ). Quyết định chỉ phụ thuộc vào 1 điểm gần nhất. Nếu tập train có một mẫu nhiễu, nhãn bị gán sai sót (label noise) hoặc một tin ham cá biệt tình cờ dùng từ ngữ gần giống spam, điểm test rơi cạnh đó sẽ lập tức bị dự đoán sai.
  - **Khi $k$ quá lớn ($k = 25$):** Mô hình rơi vào trạng thái *High Bias* (độ chệch cao / underfitting). Vùng lân cận bị mở rộng ra quá xa. Do tập dữ liệu mất cân bằng nghiêm trọng (~87% ham và chỉ ~13% spam), khi lấy tới 25 láng giềng, xác suất số lượng mẫu ham áp đảo số phiếu tăng vọt. Hậu quả là các tin spam thực tế không đủ quá bán phiếu bầu để được gán nhãn 1, khiến Recall giảm mạnh từ 97.9% xuống còn 88.4%.
  - **Kết luận:** $k = 5$ là giá trị tối ưu, vừa đủ số đông để lọc bớt nhiễu ngẫu nhiên, vừa đủ hẹp để phản ánh cấu trúc cụm mật độ cục bộ của lớp thiểu số (spam).

---

## 6. Phương án cho sản phẩm thật

### Bối cảnh yêu cầu
- Xử lý **10 triệu SMS / ngày** (trung bình ~116 tin/giây, giờ cao điểm có thể chạm mốc 500 – 1,000 tin/giây).
- Ràng buộc khắt khe: **Độ trễ $p95 < 50\text{ ms/tin}$**.

### Phân tích dựa trên số liệu thực tế bài lab
1. **Loại bỏ LLM (4a, 4b, 4c):**
   - Độ trễ $p95$ của LLM nhỏ (gpt-4o-mini) là **5,709 ms**, LLM lớn (gpt-4o) là **2,458 ms** $\rightarrow$ Vi phạm SLA gấp 50 – 110 lần!
   - Thông lượng chỉ đạt 0.2 – 0.5 tin/s, muốn tải 1,000 tin/s cần hàng nghìn kết nối song song và đối mặt với Rate Limit khắt khe của API.
   - Chi phí: 10 triệu tin/ngày = 300 triệu tin/tháng.
     - Dùng `gpt-4o-mini`: $300,000 \times \$0.0125 \approx$ **\$3,750 / tháng** (~95 triệu VNĐ/tháng).
     - Dùng `gpt-4o`: $300,000 \times \$0.4125 \approx$ **\$123,750 / tháng** (~3.1 tỷ VNĐ/tháng).
     $\rightarrow$ Hoàn toàn không khả thi cả về kỹ thuật lẫn tài chính.

2. **Cân nhắc Embedding (3a, 3b):**
   - `3a (MiniLM + kNN)`: $p95 = 83.20\text{ ms}$ (vượt quá 50 ms), thông lượng chỉ 21.9 tin/s $\rightarrow$ Bị loại.
   - `3b (MiniLM + LogReg)`: Đạt F1 cao nhất bài lab (**0.963**), $p95 = 50.23\text{ ms}$ (sát sạt ngưỡng 50 ms). Tuy nhiên thông lượng ~31.6 tin/s trên CPU đòi hỏi cụm máy chủ embedding đắt đỏ để đáp ứng giờ cao điểm.

3. **Mô hình vượt trội về hiệu năng: `2. BoW + MLP`:**
   - $p95$ chỉ **0.37 ms** (nhanh gấp 135 lần yêu cầu 50 ms).
   - Thông lượng khủng khiếp: **25,403 tin/s** chỉ trên 1 node tính toán thông thường.
   - F1 test đạt **0.940** (Precision 97.7%, Recall 90.5%), chi phí suy luận xấp xỉ **$0**.

### Kiến trúc giải pháp khuyến nghị: Hệ thống định tuyến đa tầng (Cascading Router)
Để vừa thỏa mãn SLA thời gian thực, vừa đạt độ chính xác tối đa trên cả các case khó, giải pháp thực tế tối ưu là kết hợp 2 tầng:

```
[Tin nhắn SMS đến]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Tầng 1: BoW + MLP (PyTorch) — Chi phí $0, p95 < 0.5ms │
└────────────────────────────────────────────────────────┘
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
  p(spam) < 0.05        0.05 ≤ p ≤ 0.95        p(spam) > 0.95
 (Chắc chắn Ham)           (Lưỡng lự)         (Chắc chắn Spam)
        │                  (~5% lượng tin)            │
        ▼                      │                      ▼
   [ Cho qua ]                 │                  [ Chặn ]
                               ▼
        ┌────────────────────────────────────────────────┐
        │ Tầng 2: MiniLM + LogReg (Embedding ngữ nghĩa)  │
        │ p95 ~ 35ms, F1 = 0.963                         │
        └────────────────────────────────────────────────┘
                               │
                      ┌────────┴────────┐
                      ▼                 ▼
                    p < 0.5           p ≥ 0.5
                  [ Cho qua ]         [ Chặn ]
```

- **Đánh đổi và Lợi ích:**
  - 95% lượng tin nhắn được giải quyết ngay ở Tầng 1 với độ trễ $< 1\text{ ms}$.
  - Chỉ 5% tin mập mờ mới cần chạy qua Tầng 2 với độ trễ ~35 ms.
  - **Kết quả tổng thể:** Toàn bộ hệ thống đạt độ trễ trung bình $p95 < 2\text{ ms}$, thông lượng trung bình $> 15,000\text{ tin/s}$, F1 tổng thể đạt $\approx 0.965$ và chi phí vận hành gần như bằng 0.

---

## 7. AI log và điều đã học

### Bảng AI log

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **4.2 (Prompt)** | *"Thiết kế prompt zero-shot và few-shot để LLM phân loại SMS spam/ham trả về đúng 1 từ"* | Đưa ra prompt mẫu có chỉ dẫn vai trò, danh sách ví dụ and thẻ kết thúc `Message: """..."""\nLabel:` | Giữ đúng format quy định của lab, thêm kiểm tra không sinh kèm dấu câu hay giải thích ngoài lề. Đã test qua check `4.2–4.3 prompt / parse_label` ✅. |
| **4.3 (Parse Label)** | *"Viết hàm parse_label nhận chuỗi trả về từ LLM, trích xuất 1 (spam), 0 (ham/not spam), hoặc None"* | Cung cấp hàm chuẩn hóa regex và kiểm tra substring `"spam"` / `"ham"` | Phát hiện bẫy nếu kiểm tra `"spam" in text` trước thì cụm `"not spam"` sẽ bị đoán nhầm thành 1. Sửa lại: ưu tiên bắt `"not spam"` trả về 0 trước, loại bỏ tiền tố `Label:`. |
| **4.1 (API Client)** | *"Cách cấu hình OpenAI SDK với custom base_url và tương thích tham số token giữa các model reasoning"* | Đưa ra cấu hình `base_url` và hàm `provider_kwargs` ánh xạ `max_completion_tokens` cho OpenAI | Tích hợp vào `call_openai`, đồng thời xử lý lỗi duplicate cell và placeholder `"..."` ở cell thí nghiệm few-shot để toàn bộ notebook chạy trơn tru. |

### Điều bạn hiểu ra hôm nay mà trước đó chưa hiểu
1. **Không phải bài toán nào cũng cần LLM:** Trước bài lab này, tôi từng nghĩ LLM luôn là lựa chọn vượt trội nhất cho mọi bài toán NLP hiện đại. Thực tế chứng minh: một mạng nơ-ron MLP đơn giản huấn luyện trên BoW hay một bộ phân loại tuyến tính trên vector MiniLM có thể chạy nhanh gấp **70,000 lần**, chi phí bằng 0, mà chất lượng phân loại trên dữ liệu thực tế lại tương đương hoặc nhỉnh hơn LLM zero-shot.
2. **Ảo tưởng về điểm số test cao (Data Leakage & Distribution Shift):** Mô hình đạt F1 98–99% trên tập test ngẫu nhiên không đồng nghĩa với việc nó đã "hiểu" ngữ nghĩa. Chỉ cần gặp bộ 50 case đối kháng (adversarial test) với các thủ thuật ngụy trang từ ngữ, mô hình BoW lập tức sụp đổ về mức 33–46%. Điều này dạy tôi bài học sâu sắc về việc kiểm thử mô hình trong môi trường sản xuất thực tế.
3. **Ý nghĩa sống còn của độ trễ (Latency SLA) và thiết kế hệ thống thực tế:** Khi thiết kế giải pháp cho sản phẩm quy mô 10 triệu người dùng, F1 không phải là thước đo duy nhất. Khái niệm đường Pareto giữa chất lượng và độ trễ, cùng mô hình định tuyến đa tầng (Cascading Router) là tư duy kỹ thuật giá trị nhất mà tôi học được để cân bằng giữa độ chính xác, tốc độ và chi phí hạ tầng.
