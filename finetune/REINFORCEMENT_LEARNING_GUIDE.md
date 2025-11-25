# Reinforcement Learning cho Fine-tuning LLM: Tổng Quan & Đề Xuất

## 📚 Mục lục
1. [Giới thiệu chung](#1-giới-thiệu-chung)
2. [RLHF - Reinforcement Learning from Human Feedback](#2-rlhf---reinforcement-learning-from-human-feedback)
3. [DPO - Direct Preference Optimization](#3-dpo---direct-preference-optimization)
4. [GRPO - Group Relative Policy Optimization](#4-grpo---group-relative-policy-optimization)
5. [Các phương pháp khác](#5-các-phương-pháp-khác)
6. [So sánh các phương pháp](#6-so-sánh-các-phương-pháp)
7. [Đề xuất cho bài toán luật pháp](#7-đề-xuất-cho-bài-toán-luật-pháp)

---

## 1. Giới thiệu chung

### Tại sao cần Reinforcement Learning sau SFT?

**Supervised Fine-Tuning (SFT)** giúp model học được:
- ✅ Định dạng câu trả lời
- ✅ Kiến thức từ dữ liệu training
- ✅ Cách trả lời theo instruction

**Nhưng SFT có hạn chế:**
- ❌ Không phân biệt được câu trả lời "tốt" vs "xuất sắc"
- ❌ Dễ bị overfitting vào format cụ thể
- ❌ Không tối ưu hóa cho mục tiêu con người (helpfulness, harmlessness, honesty)

**Reinforcement Learning (RL)** giải quyết bằng cách:
- 🎯 Học từ feedback/preference của con người
- 🎯 Tối ưu hóa cho các mục tiêu cụ thể (accuracy, safety, coherence)
- 🎯 Cải thiện chất lượng output theo hướng mong muốn

---

## 2. RLHF - Reinforcement Learning from Human Feedback

### 2.1. Nguyên lý

RLHF là phương pháp RL cổ điển cho LLM, được sử dụng bởi OpenAI (ChatGPT), Anthropic (Claude).

**Pipeline 3 bước:**

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  1. SFT Model   │  →   │  2. Reward Model │  →   │  3. RL Training │
│  (Base model)   │      │  (Preference)    │      │  (PPO/RLHF)     │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

### 2.2. Chi tiết từng bước

#### Bước 1: Supervised Fine-Tuning (SFT)
- Train model trên instruction dataset (đã làm ở phần trước)
- Tạo base model có khả năng follow instructions

#### Bước 2: Train Reward Model (RM)
**Mục tiêu:** Học được preference của con người

**Dữ liệu cần:**
```json
{
  "prompt": "Tuổi nghỉ hưu của công nhân viên chức là bao nhiêu?",
  "chosen": "Theo Luật Lao động 2019, tuổi nghỉ hưu là...",  // Better response
  "rejected": "Không rõ, bạn nên hỏi luật sư."  // Worse response
}
```

**Cách hoạt động:**
- Train một model phân loại để score responses
- Model học predict: `score(chosen) > score(rejected)`
- Loss function: `L = -log(σ(r_chosen - r_rejected))`

**Số lượng data cần:**
- Tối thiểu: 1,000-5,000 pairs
- Lý tưởng: 10,000-50,000 pairs
- Càng nhiều càng tốt!

#### Bước 3: RL Training với PPO

**Proximal Policy Optimization (PPO):**

Công thức objective:
```
L_PPO = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)] - β·KL(π_θ || π_ref)
```

Trong đó:
- `π_θ`: Policy hiện tại (model đang train)
- `π_ref`: Reference policy (SFT model)
- `KL`: KL divergence (đảm bảo không đi quá xa reference)
- `β`: KL penalty coefficient
- `A`: Advantage (từ reward model)

**Training loop:**
```
for batch in dataset:
    1. Generate responses từ current policy
    2. Score responses bằng reward model
    3. Compute advantages
    4. Update policy với PPO objective
    5. Ensure KL divergence không quá lớn
```

### 2.3. Ưu điểm

✅ **Hiệu quả cao**: Đã được chứng minh với ChatGPT, Claude
✅ **Linh hoạt**: Có thể tối ưu cho nhiều objectives khác nhau
✅ **Controllable**: KL penalty giữ model không đi quá xa base

### 2.4. Nhược điểm

❌ **Phức tạp**: Cần train 2 models (reward model + policy)
❌ **Tốn tài nguyên**: 
  - Cần ~2x VRAM (lưu cả reward model và policy)
  - Training chậm hơn SFT nhiều
❌ **Dữ liệu đắt**: Cần human preference data
❌ **Khó train**: PPO unstable, cần tune nhiều hyperparameters
❌ **Thời gian**: ~3-5x thời gian SFT

### 2.5. Ước tính thời gian cho Kaggle

**Với 2xT4 GPU (30h/week):**
- Train reward model: ~5-8 giờ (trên 10k pairs)
- RL training: ~20-25 giờ
- **Tổng: ~30 giờ** (vừa khít giới hạn!)

---

## 3. DPO - Direct Preference Optimization

### 3.1. Nguyên lý

DPO là **phương pháp RL không cần reward model**, được đề xuất bởi Stanford 2023.

**Key insight:** Thay vì train reward model rồi dùng RL, ta có thể tối ưu trực tiếp từ preference data!

### 3.2. Cách hoạt động

**DPO objective:**

```
L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

Trong đó:
- `y_w`: Chosen response (better)
- `y_l`: Rejected response (worse)
- `π_θ`: Model đang train
- `π_ref`: Reference model (SFT)
- `β`: Temperature parameter

**Hiểu đơn giản:**
- Tăng probability của `y_w` (chosen)
- Giảm probability của `y_l` (rejected)
- Giữ model gần với reference model

### 3.3. Pipeline đơn giản

```
┌─────────────────┐      ┌──────────────────┐
│  1. SFT Model   │  →   │  2. DPO Training │
│  (Base model)   │      │  (Preference)    │
└─────────────────┘      └──────────────────┘
```

Chỉ 2 bước thay vì 3!

### 3.4. Dữ liệu cần

Giống RLHF, cần preference pairs:
```json
{
  "prompt": "Điều kiện cấp bằng lái xe là gì?",
  "chosen": "Theo Luật Giao thông 2008, điều kiện cấp bằng lái xe...",
  "rejected": "Cần đủ tuổi và thi đậu."
}
```

**Số lượng:**
- Tối thiểu: 1,000 pairs
- Khuyến nghị: 5,000-10,000 pairs
- Ít hơn RLHF vì không cần train reward model riêng

### 3.5. Ưu điểm

✅ **Đơn giản hơn RLHF**: Không cần reward model
✅ **Ít tài nguyên hơn**: Chỉ cần 1 model thay vì 2
✅ **Training ổn định hơn**: Không có PPO instability
✅ **Nhanh hơn**: ~2x nhanh hơn RLHF
✅ **Hiệu quả tương đương**: Kết quả tốt như RLHF trong nhiều tasks

### 3.6. Nhược điểm

❌ **Vẫn cần preference data**: Tốn công tạo/label
❌ **Ít linh hoạt hơn RLHF**: Khó tối ưu cho multiple objectives
❌ **Mới hơn**: Ít được test trong production

### 3.7. Ước tính thời gian cho Kaggle

**Với 2xT4 GPU:**
- DPO training: ~8-12 giờ (trên 5k pairs)
- **Tổng: ~10 giờ** (rất phù hợp với 30h/week!)

---

## 4. GRPO - Group Relative Policy Optimization

### 4.1. Nguyên lý

GRPO là biến thể của RLHF, được DeepSeek phát triển, tối ưu cho **nhóm responses**.

**Key idea:** Thay vì so sánh 2 responses, ta generate nhiều responses và rank chúng.

### 4.2. Cách hoạt động

**Pipeline:**

```
1. Generate K responses cho mỗi prompt (K=4-8)
2. Score tất cả responses bằng reward model (hoặc auto metric)
3. Rank responses theo score
4. Update policy để:
   - Tăng prob của top responses
   - Giảm prob của bottom responses
```

**Objective:**

```
L_GRPO = E[∑(r_i - r_mean) * log π_θ(y_i|x)]
```

Trong đó:
- `r_i`: Reward của response i
- `r_mean`: Average reward trong group
- Responses tốt hơn mean được tăng prob, ngược lại giảm

### 4.3. Ưu điểm

✅ **Ổn định hơn PPO**: Group-based normalization
✅ **Sample efficient**: Học từ nhiều responses cùng lúc
✅ **Tốt cho ranking tasks**: Phù hợp khi có nhiều levels of quality

### 4.4. Nhược điểm

❌ **Tốn compute**: Phải generate K responses mỗi lần
❌ **Vẫn cần reward model**: Giống RLHF
❌ **Implementation phức tạp**: Ít library support

### 4.5. Ước tính thời gian

**Với 2xT4 GPU:**
- Train reward model: ~5-8 giờ
- GRPO training: ~15-20 giờ
- **Tổng: ~25 giờ**

---

## 5. Các phương pháp khác

### 5.1. RLAIF - RL from AI Feedback

**Ý tưởng:** Dùng AI (GPT-4, Claude) để generate preference data thay vì human.

**Ưu điểm:**
✅ Rẻ hơn human labeling
✅ Scale dễ dàng
✅ Consistent

**Nhược điểm:**
❌ Phụ thuộc vào AI teacher (có thể biased)
❌ Cần API access (GPT-4)

**Phù hợp:** Khi không có budget cho human labeling

### 5.2. KTO - Kahneman-Tversky Optimization

**Ý tưởng:** Dựa trên prospect theory, optimize cho binary feedback (good/bad).

**Dữ liệu:**
```json
{
  "prompt": "...",
  "response": "...",
  "label": "good"  // hoặc "bad"
}
```

**Ưu điểm:**
✅ Đơn giản hơn preference pairs
✅ Dữ liệu dễ collect (chỉ cần thumbs up/down)

**Nhược điểm:**
❌ Kém thông tin hơn pairs
❌ Mới, chưa được test rộng rãi

### 5.3. IPO - Identity Preference Optimization

**Ý tưởng:** Variant of DPO với regularization tốt hơn.

**Objective:**
```
L_IPO = E[(log(π_θ(y_w|x)/π_θ(y_l|x)) - 1/β)^2]
```

**Ưu điểm:**
✅ Ổn định hơn DPO
✅ Less sensitive to β

### 5.4. ORPO - Odds Ratio Preference Optimization

**Ý tưởng:** Combine SFT và preference learning trong 1 loss.

**Loss:**
```
L_ORPO = L_SFT + λ·L_OR
```

**Ưu điểm:**
✅ Train trong 1 bước (không cần SFT riêng)
✅ Hiệu quả tốt

**Nhược điểm:**
❌ Mới, ít documentation

---

## 6. So sánh các phương pháp

### 6.1. Bảng so sánh tổng quan

| Phương pháp | Độ phức tạp | Thời gian train | VRAM cần | Dữ liệu cần | Hiệu quả | Độ ổn định |
|-------------|-------------|-----------------|----------|-------------|----------|------------|
| **RLHF** | ⭐⭐⭐⭐⭐ | 25-30h | 2x model | 10k+ pairs | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **DPO** | ⭐⭐⭐ | 10-12h | 1.5x model | 5k+ pairs | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GRPO** | ⭐⭐⭐⭐ | 20-25h | 2x model | 10k+ pairs | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **KTO** | ⭐⭐ | 8-10h | 1.5x model | 5k+ samples | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **IPO** | ⭐⭐⭐ | 10-12h | 1.5x model | 5k+ pairs | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ORPO** | ⭐⭐ | 8-10h | 1x model | 5k+ pairs | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 6.2. Flowchart lựa chọn

```
Bạn có preference data?
│
├─ Có (pairs: chosen/rejected)
│  │
│  ├─ Có nhiều GPU resources & thời gian?
│  │  └─ YES → RLHF (best quality)
│  │  └─ NO → DPO (recommended)
│  │
│  └─ Muốn đơn giản nhất?
│     └─ ORPO hoặc IPO
│
└─ Không
   │
   ├─ Có budget dùng GPT-4?
   │  └─ YES → RLAIF (generate preference data)
   │
   ├─ Chỉ có binary feedback (good/bad)?
   │  └─ KTO
   │
   └─ Không có gì
      └─ Tạo preference data bằng:
         - Self-critique
         - Rule-based scoring
         - Hoặc skip RL, chỉ làm SFT
```

---

## 7. Đề xuất cho bài toán luật pháp

### 7.1. Phân tích yêu cầu

**Đặc điểm bài toán:**
- ✅ Có 97k instruction data (SFT done)
- ❌ KHÔNG có preference data
- ✅ Giới hạn thời gian: 30h/week Kaggle
- ✅ GPU: 2xT4 (limited VRAM)
- 🎯 Mục tiêu: Câu trả lời chính xác, có trích dẫn luật

### 7.2. Đề xuất: **DPO với Synthetic Preference Data**

**Tại sao DPO?**

1. ✅ **Đơn giản & ổn định**: Dễ implement, ít bug
2. ✅ **Tiết kiệm thời gian**: ~10-12h training
3. ✅ **Hiệu quả cao**: Gần bằng RLHF
4. ✅ **Phù hợp VRAM**: Chỉ cần ~20-24GB (OK với 2xT4)

**Giải quyết vấn đề không có preference data:**

### 7.3. Cách tạo Synthetic Preference Data

#### Phương án 1: Self-Critique (Tự động 100%)

**Ý tưởng:** Dùng chính SFT model để generate multiple responses, sau đó tự rank.

```python
# Pseudo-code
for sample in dataset:
    prompt = sample['instruction'] + sample['input']
    
    # Generate 3-5 responses với different temps
    responses = []
    for temp in [0.7, 0.9, 1.1]:
        response = model.generate(prompt, temperature=temp)
        responses.append(response)
    
    # Score responses bằng các metrics
    scores = []
    for resp in responses:
        score = (
            check_citation(resp) * 0.4 +      # Có trích dẫn luật?
            check_accuracy(resp, reference) * 0.4 +  # Chính xác?
            check_coherence(resp) * 0.2       # Mạch lạc?
        )
        scores.append(score)
    
    # Tạo preference pair
    best_idx = argmax(scores)
    worst_idx = argmin(scores)
    preference_data.append({
        'prompt': prompt,
        'chosen': responses[best_idx],
        'rejected': responses[worst_idx]
    })
```

**Metrics tự động:**
- **Citation check**: Regex tìm "Điều", "Luật", "Nghị định"
- **Accuracy**: Rouge score với reference answer
- **Length**: Prefer longer, detailed answers
- **Coherence**: Perplexity score

**Ưu điểm:**
- ✅ Hoàn toàn tự động
- ✅ Free, scalable
- ✅ Có thể tạo 10k+ pairs

**Nhược điểm:**
- ❌ Quality không cao bằng human
- ❌ Có thể reinforcement bias của SFT model

#### Phương án 2: RLAIF với GPT-4 (Nửa tự động)

**Setup:**
```python
# Generate 2 responses
response_a = sft_model.generate(prompt, temperature=0.7)
response_b = sft_model.generate(prompt, temperature=1.0)

# Dùng GPT-4 để judge
judge_prompt = f"""
Bạn là chuyên gia đánh giá câu trả lời pháp luật.

Câu hỏi: {prompt}

Trả lời A: {response_a}
Trả lời B: {response_b}

Đánh giá trả lời nào tốt hơn theo tiêu chí:
1. Chính xác về mặt pháp luật
2. Có trích dẫn cụ thể
3. Đầy đủ, chi tiết
4. Dễ hiểu

Trả lời: (A/B/Tie)
Lý do: ...
"""

result = gpt4_api.chat(judge_prompt)
# Parse result và tạo preference pair
```

**Chi phí ước tính:**
- 10k pairs × $0.01/request = **$100**
- Hoặc dùng GPT-3.5: **$20-30**

**Ưu điểm:**
- ✅ Quality cao hơn self-critique
- ✅ Có thể control criteria

**Nhược điểm:**
- ❌ Tốn tiền (nhưng không nhiều)
- ❌ Cần API access

#### Phương án 3: Rule-Based + Manual Sampling (Hybrid)

**Bước 1:** Tạo 80% data bằng rule-based
**Bước 2:** Manual review 20% quan trọng nhất

```python
def score_response(response, reference):
    score = 0
    
    # Rule 1: Phải có trích dẫn
    if re.search(r'(Điều|Luật|Nghị định) \d+', response):
        score += 30
    
    # Rule 2: Độ dài phù hợp (100-500 words)
    words = len(response.split())
    if 100 <= words <= 500:
        score += 20
    
    # Rule 3: Giống reference
    rouge = compute_rouge(response, reference)
    score += rouge * 30
    
    # Rule 4: Không có câu lặp
    if not has_repetition(response):
        score += 10
    
    # Rule 5: Có kết luận rõ ràng
    if has_conclusion(response):
        score += 10
    
    return score
```

### 7.4. Implementation Plan

**Tuần 1: Tạo Preference Data**
- Ngày 1-2: Implement self-critique pipeline
- Ngày 3-4: Generate 10k preference pairs
- Ngày 5: Validate quality (sample 100 pairs)
- Ngày 6-7: (Optional) Refine với GPT-4 cho 1k critical cases

**Tuần 2: DPO Training**
- Setup DPO với TRL library
- Train 10-12 giờ
- Evaluate & iterate

**Tuần 3: Testing & Refinement**
- A/B test với SFT model
- Human evaluation trên 50-100 cases
- Fine-tune hyperparameters nếu cần

### 7.5. Code template cho DPO

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load SFT model
model = AutoModelForCausalLM.from_pretrained("path/to/sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("path/to/sft-model")
tokenizer = AutoTokenizer.from_pretrained("path/to/sft-model")

# Load preference dataset
train_dataset = load_dataset("json", data_files="preference_train.jsonl")
eval_dataset = load_dataset("json", data_files="preference_eval.jsonl")

# DPO Config
dpo_config = DPOConfig(
    output_dir="./legal-model-dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,  # Lower than SFT
    num_train_epochs=1,  # Usually 1 epoch enough
    beta=0.1,  # Temperature for DPO
    max_length=2048,
    max_prompt_length=1024,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    bf16=True,
    remove_unused_columns=False,
)

# Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train!
dpo_trainer.train()
```

### 7.6. Metrics để evaluate

**Automatic metrics:**
1. **Citation Rate**: % responses có trích dẫn luật
2. **Rouge Score**: So với reference answers
3. **Win Rate**: A/B test với SFT model (dùng GPT-4 judge)

**Human evaluation (sample 50-100):**
1. Accuracy (1-5 scale)
2. Completeness (1-5 scale)
3. Citation quality (1-5 scale)
4. Helpfulness (1-5 scale)

### 7.7. Timeline & Resource estimate

| Phase | Thời gian | GPU time | Cost |
|-------|-----------|----------|------|
| Generate preference data | 2-3 ngày | 5-8h | $0-100 |
| DPO training | 1 ngày | 10-12h | $0 |
| Evaluation | 1-2 ngày | 2-3h | $0 |
| **Tổng** | **5-6 ngày** | **~20h** | **$0-100** |

**Vừa khít với giới hạn 30h/week Kaggle! ✅**

---

## 8. Kết luận & Recommendations

### 8.1. Top recommendations cho project của bạn

**🥇 Tier 1 (Highly Recommended):**
1. **DPO với Self-Critique** - Best balance of effort/quality
   - Thời gian: ~20h
   - Chi phí: $0
   - Complexity: Medium
   - Expected improvement: +15-25% over SFT

2. **DPO với RLAIF (GPT-3.5)** - If have small budget
   - Thời gian: ~20h
   - Chi phí: ~$30
   - Complexity: Medium
   - Expected improvement: +20-30% over SFT

**🥈 Tier 2 (Alternative):**
3. **ORPO** - Nếu muốn đơn giản tối đa
   - Thời gian: ~15h
   - Chi phí: $0
   - Complexity: Low
   - Expected improvement: +10-20% over SFT

**🥉 Tier 3 (Advanced):**
4. **RLHF** - Nếu có thêm compute & data
   - Thời gian: ~30h
   - Chi phí: $0-200 (for RM training data)
   - Complexity: High
   - Expected improvement: +25-35% over SFT

### 8.2. Recommended learning path

**Nếu bạn mới bắt đầu:**
```
Week 1: SFT (done!) ✓
Week 2: Generate preference data (self-critique)
Week 3: DPO training
Week 4: Evaluation & iteration
```

**Nếu bạn có kinh nghiệm:**
```
Week 1: SFT + Start preference data generation ✓
Week 2: DPO training + RLAIF for critical cases
Week 3: RLHF or advanced methods
```

### 8.3. Khi nào nên SKIP RL?

**Skip RL nếu:**
- ❌ SFT model đã đủ tốt cho use case
- ❌ Không có resources để tạo preference data
- ❌ Deadline gấp
- ❌ Data quá domain-specific (RL có thể làm worse)

**Dấu hiệu SFT đã đủ:**
- ✅ Model trả lời đúng >80% test cases
- ✅ Citations đầy đủ
- ✅ Format nhất quán
- ✅ Users hài lòng

---

## 9. Resources & References

### 9.1. Papers

**RLHF:**
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) (OpenAI, 2020)

**DPO:**
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Stanford, 2023)

**GRPO:**
- [DeepSeekMath: Pushing the Limits](https://arxiv.org/abs/2402.03300) (DeepSeek, 2024)

**Other methods:**
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [IPO: A General Framework for Preference Optimization](https://arxiv.org/abs/2310.12036)
- [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691)

### 9.2. Code & Libraries

**TRL (Transformer Reinforcement Learning):**
```bash
pip install trl
```
- Supports: DPO, PPO, RLHF, KTO, ORPO
- Docs: https://huggingface.co/docs/trl

**Unsloth với DPO:**
```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig
```

**Alignment Handbook (HF):**
- https://github.com/huggingface/alignment-handbook
- Best practices & recipes

### 9.3. Datasets (for reference)

**Preference datasets:**
- Anthropic HH-RLHF: https://huggingface.co/datasets/Anthropic/hh-rlhf
- OpenAssistant Conversations: https://huggingface.co/datasets/OpenAssistant/oasst1
- UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback

### 9.4. Tools for creating preference data

**LabelStudio:**
- UI để human annotation
- https://labelstud.io/

**Argilla:**
- Annotation platform with AI assistance
- https://argilla.io/

**LLM-as-a-Judge:**
- Prometheus: https://huggingface.co/prometheus-eval
- Auto-evaluation framework

---

## 10. Next Steps

**Immediate (Ngay bây giờ):**
1. ✅ Đọc xong document này
2. ⬜ Quyết định method: DPO recommended
3. ⬜ Bắt đầu tạo preference data

**This Week:**
1. Implement self-critique pipeline
2. Generate 5-10k preference pairs
3. Validate quality

**Next Week:**
1. Setup DPO training
2. Train model
3. Evaluate & compare với SFT

**Future (Optional):**
1. Try RLHF nếu DPO không đủ
2. Experiment với ORPO, IPO
3. A/B test trong production

---

**Chúc bạn thành công với project! 🚀**

Nếu có câu hỏi gì, hãy hỏi tôi nhé!
