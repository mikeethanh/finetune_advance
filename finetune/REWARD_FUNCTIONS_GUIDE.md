# Hướng dẫn chi tiết về Reward Functions cho Vietnamese Legal GRPO Training

## Tổng quan

Trong quá trình huấn luyện GRPO (Generalized Reward Preference Optimization) cho model Vietnamese Legal Llama3.2-3B, chúng ta sử dụng **5 reward functions chính** được tối ưu hóa để đánh giá chất lượng các response của model:

### **Functions cho Format và Response Structure:**
1. **match_format_exactly** - Kiểm tra format reasoning hoàn hảo (regex matching)
2. **match_format_approximately** - Đánh giá từng thành phần format riêng biệt  
3. **check_response_length_only** - Kiểm tra độ dài response phù hợp

### **Functions cho Content Quality:**
4. **check_vietnamese_language_consistency** - Kiểm tra tính nhất quán ngôn ngữ tiếng Việt  
5. **check_vietnamese_legal_reasoning** - Đánh giá suy luận pháp lý tiếng Việt

> **🎯 Cải tiến chính:** Tách rời việc đánh giá format structure thành 2 functions riêng biệt để có độ chính xác cao hơn và thêm function chuyên biệt cho độ dài response.



## 1. `check_vietnamese_language_consistency`

### Mục tiêu
Kiểm tra tính nhất quán của ngôn ngữ tiếng Việt trong response, đảm bảo model không trộn lẫn quá nhiều tiếng Anh hoặc các ngôn ngữ khác.

**Các bước thực hiện:**

2. **Kiểm tra ngôn ngữ trang trọng pháp lý**:
   - Pattern: "được", "sẽ", "phải", "theo", "quy định", "luật", v.v.
   - +0.2 điểm cho mỗi từ unique (tối đa 1.0 điểm)

3. **Phạt nếu quá nhiều tiếng Anh** (chỉ với response >50 ký tự):
   - >40% từ tiếng Anh: -1.0 điểm (quá nhiều cho văn bản pháp lý VN)
   - >20% từ tiếng Anh: -0.3 điểm

4. **Thưởng cấu trúc câu tiếng Việt**: +0.5 nếu có ≥2 câu hoàn chỉnh

**Điểm số:** -1.0 đến 1.5

---

## 2. `match_format_exactly`

### Mục tiêu
Kiểm tra response có tuân thủ **hoàn hảo** cấu trúc reasoning format theo regex pattern. Function này có độ chính xác cao nhất và chỉ thưởng nếu format đúng 100%.

### Giải thích code chi tiết

```python
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores
```

**Logic đánh giá:**
1. **Regex matching hoàn hảo**: Sử dụng `match_format` regex pattern để kiểm tra cấu trúc:
   - `<start_working_out>` + nội dung + `<end_working_out>` + `<SOLUTION>` + nội dung + `</SOLUTION>`
   - **+3.0 điểm** nếu format hoàn hảo theo đúng thứ tự
   - **0 điểm** nếu không match hoàn hảo

**Điểm số:** 0 hoặc 3.0 (binary scoring)

---

## 3. `match_format_approximately`

### Mục tiêu
Đánh giá từng thành phần format riêng biệt và cho điểm partial, giúp model học dần từng bước một thay vì chỉ học "all-or-nothing".

### Giải thích code chi tiết

```python
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores
```

**Logic đánh giá:**

1. **Đếm chính xác từng tag**:
   - `<start_working_out>`: +0.5 nếu có đúng 1 lần, -1.0 nếu 0 hoặc >1 lần
   - `<end_working_out>`: +0.5 nếu có đúng 1 lần, -1.0 nếu 0 hoặc >1 lần  
   - `<SOLUTION>`: +0.5 nếu có đúng 1 lần, -1.0 nếu 0 hoặc >1 lần
   - `</SOLUTION>`: +0.5 nếu có đúng 1 lần, -1.0 nếu 0 hoặc >1 lần

2. **Ưu điểm**: 
   - Cho phép model học từng bước (gradual learning)
   - Phạt nặng việc lặp lại tags
   - Không quan tâm đến thứ tự (khác với `match_format_exactly`)

**Điểm số:** -4.0 đến +2.0

---

## 4. `check_response_length_only`

### Mục tiêu
Chuyên biệt đánh giá độ dài response cho văn bản pháp lý tiếng Việt, đảm bảo câu trả lời không quá ngắn hoặc quá dài dòng.

### Giải thích code chi tiết

```python
def check_response_length_only(prompts, completions, answer=None,
                               min_words=20,
                               ideal_max_words=150,
                               hard_max_words=300,
                               explain=False):
    def word_count(text):
        text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
        tokens = re.findall(r"\w+['-]?\w*|\w+", text, flags=re.UNICODE)
        return len(tokens)

    scores = []
    for completion in completions:
        response = completion[0]["content"]
        wc = word_count(response)
        
        score = 0
        if wc < min_words:
            score -= 2
        elif min_words <= wc <= ideal_max_words:
            score += 2
        elif ideal_max_words < wc <= hard_max_words:
            score += 1
        else:
            score -= 2
        scores.append(score)
    return scores
```

**Logic đánh giá:**

1. **Word counting thông minh**:
   - Loại bỏ code blocks (```...```) trước khi đếm
   - Sử dụng regex UNICODE để đếm từ tiếng Việt chính xác
   - Hỗ trợ từ có dấu nối (ex: "văn-bản")

2. **Phân loại độ dài tối ưu cho pháp lý**:
   - **< 20 từ**: -2 điểm (quá ngắn, không đủ thông tin)
   - **20-150 từ**: +2 điểm (độ dài lý tưởng)  
   - **150-300 từ**: +1 điểm (hơi dài nhưng chấp nhận được)
   - **> 300 từ**: -2 điểm (quá dài dòng)

3. **Tham số có thể tuỳ chỉnh**: min_words, ideal_max_words, hard_max_words

**Điểm số:** -2 đến +2

---

## 5. `check_vietnamese_legal_reasoning`

### Mục tiêu
Đánh giá chất lượng reasoning và solution trong bối cảnh pháp lý tiếng Việt, kiểm tra độ tương đồng với câu trả lời chuẩn và việc sử dụng thuật ngữ pháp lý.


**Các bước thực hiện:**
3. **Đánh giá thuật ngữ pháp lý mở rộng**:
   - **Pattern nhận diện đã mở rộng**: bao gồm 40+ thuật ngữ từ cơ bản đến chuyên sâu
     - Cơ bản: luật, nghị định, điều, khoản, quy định, vi phạm
     - Phương tiện: ô tô, xe máy, xe đạp, phương tiện giao thông
     - Hạ tầng: biển báo, đèn tín hiệu, vạch kẻ đường, làn đường
     - Vi phạm: nồng độ cồn, ma túy, chất kích thích
     - Xử phạt: tước quyền, tạm giữ, tịch thu, phạt tiền, đình chỉ
     - Thủ tục: đăng ký, đăng kiểm, bảo hiểm
   - **Scoring nâng cao**:
     - +0.3 điểm cho mỗi thuật ngữ unique (tăng từ 0.2)

4. **Đánh giá chất lượng reasoning nâng cao**:
   - **Khuyến khích reasoning dài**:
     - ≥50 từ: +1.0 điểm (chi tiết)
     - ≥20 từ: +0.7 điểm (vừa phải)
     - ≥10 từ: +0.5 điểm (cơ bản)
     - <10 từ: +0.2 điểm (quá ngắn)

---

## 6. `check_vietnamese_language_consistency` 

### Mục tiêu  
Đánh giá chất lượng câu trả lời từ góc độ cấu trúc văn bản, patterns giải thích, và tính lặp lại - **không bao gồm độ dài** (đã có function riêng).

**Các bước thực hiện:**

1. **Kiểm tra cấu trúc câu**: +0.5 nếu có ≥2 câu (split bằng dấu '.')

2. **Tìm patterns giải thích rõ ràng**:
   - +0.5 nếu có: "theo quy định", "căn cứ", "cụ thể", "do đó", etc.

3. **Phạt nội dung lặp lại**:
   - Tính tỷ lệ: tổng từ / từ unique
   - Nếu tỷ lệ > 3: -0.5 (quá lặp lại)

4. **Bonus cho trình bày có cấu trúc**: +0.3 nếu có "1.", "2.", "-", "•"

**Điểm số:** -0.5 đến 1.3 (giảm range do không còn đánh giá độ dài)

