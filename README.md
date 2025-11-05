# Part A – Decoder Base Model with Classification Head


---

##  Dataset Information

**Source:** Kaggle Competition Dataset  
**Files Used:**
- `train.csv` → split into train/validation/test  
- `test.csv` → for final Kaggle inference  
- `sample_submission.csv` → used as a submission template  

**Split ratio:**
| Subset | Percentage | Purpose |
|:-------|:------------|:--------|
| Train | 80 % | model training |
| Validation | 10 % | monitor metrics during training |
| Test | 10 % | final local evaluation |
| Kaggle Test | competition `test.csv` | inference + submission only |

---

## Model Setup

**Architecture:** Decoder-based LLM (Google Gemma-2 2B)  
**Head:** Single-node binary classification layer  

**Frameworks:**
- Hugging Face Transformers  
- PEFT (LoRA r = 64, α = 128, dropout = 0.05)  
- Bits and Bytes (4-bit quantization)  
- WandB for experiment tracking  
- PyTorch + Evaluate for metrics  

**Hyperparameters**
| Parameter | Value |
|:-----------|:------|
| Learning Rate | 2 × 10⁻⁵ |
| Batch Size (train/eval) | 4 / 8 |
| Accumulation Steps | 2 |
| Epochs | 4 |
| Scheduler | Cosine |
| Warmup Ratio | 0.05 |
| Precision | FP16 |
| Gradient Checkpointing | Enabled |

---

## Training and Evaluation

Training was performed on **Google Colab A100** using the binary cross-entropy loss with logits.

**Best validation metrics (epoch 3):**

| Metric | Score |
|:--------|:------|
| Accuracy | 0.8227 |
| Precision | 0.8102 |
| Recall | 0.8495 |
| F1 Score | **0.8294** |
| ROC-AUC | 0.8903 |

---

## Threshold Analysis

| Threshold | Accuracy | Macro F1 | ROC-AUC |
|:-----------|:----------|:----------|:----------|
| 0.50 (default) | 0.8227 | 0.8224 | 0.8903 |
| **0.33 (optimal)** | 0.8202 | 0.8189 | 0.8903 |

**Best F1 = 0.8345 at threshold = 0.33**

**Confusion Matrix @ 0.33**
```
[[149  51]
 [ 22 184]]
```

---

## 🧾 Inference and Submission

The trained model was used for inference on **Kaggle’s test.csv**.  
Predictions were converted to binary labels using the best threshold (0.33) and saved as:

```
submission_partA_gemma2_lora.csv
```

**File columns:**  
`row_id, rule_violation`

**Sample Output:**
| row_id | rule_violation |
|:--------|:----------------|
| 0 | 1 |
| 1 | 0 |
| 2 | 1 |

---

## Tools and Experiment Tracking
- All experiments logged to Weights & Biases  
  *Run Name:* `partA_gemma_lora_bce_cosine`  
  *Project URL:* [wandb.ai/ronchaudhuri29-the-university-of-texas-at-dallas/huggingface/runs/n7iu1ygz](https://wandb.ai/ronchaudhuri29-the-university-of-texas-at-dallas/huggingface/runs/n7iu1ygz)

---

## Results Summary
- **Best F1 Score (Validation): 0.8345**
- **Optimal Threshold: 0.33**
- **ROC-AUC: 0.8903**
- **CSV generated for Kaggle submission**

---

## File Structure

```
PartA_Classification_Head.ipynb
│
├── datasets/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── gemma_lora_binary_out/
├── submission_partA_gemma2_lora.csv
└── README.md
```

---

# Part B – Decoder Base Model with Language Head


---

## Dataset Information

**Source:** Kaggle Competition Dataset  

**Files Used:**
- `train.csv` → split into train / validation / test
- `test.csv` → for local and Kaggle inference

**Split Ratio:**

| Subset | Percentage | Purpose |
|:-------|:------------|:--------|
| Train | 80 % | model training |
| Validation | 10 % | monitor metrics during training |
| Test | 10 % | final evaluation |

---

## Model Setup

**Architecture:** Google Gemma-2 2B (Decoder LLM)  
**Objective:** Binary classification (“complies” vs “violates”) using a Language Model Head  
**Loss Type:** Completion-only loss (masking non-target tokens)  

**Frameworks:**
- Hugging Face Transformers + TRL (SFTTrainer)
- PEFT (LoRA r = 32, α = 32, dropout = 0.05)
- Bits and Bytes (4-bit NF4 quantization)
- Weights & Biases for tracking
- PyTorch + Evaluate for metrics

**Prompt Formatting:**  
Each sample was converted into a completion-style prompt as follows:

```
Decide if the Reddit comment complies with the subreddit rules.
Reply with exactly one word: complies or violates.

Rule: <rule_text>
Comment: <comment_text>
Answer:
```

---

## Hyperparameters

| Parameter | Value |
|:-----------|:------|
| Learning Rate | 1 × 10⁻⁵ |
| Batch Size (train / eval) | 4 / 4 |
| Accumulation Steps | 4 |
| Epochs | 2 |
| Precision | BF16 |
| Optimizer | AdamW (Torch) |
| Gradient Checkpointing | Enabled |
| Max Sequence Length | 1024 |
| Quantization | 4-bit (NF4) |
| Weight Decay | 0.0 |

---

## Training and Evaluation

Training was performed on **Google Colab A1000** with LoRA adapters and 4-bit quantization.  
Loss was computed only on completion tokens following the “Answer:” marker to stabilize learning.

**Final Evaluation Metrics (Test Set):**

| Metric | Score |
|:--------|:------|
| Accuracy | 0.5961 |
| F1 Macro | 0.3879 |

**Confusion Matrix**
```
[[81 15]
 [65 40]]
```

---

## Observations

- The model learns basic rule-violation patterns but still shows bias toward the majority class (“complies”).
- The Language Model Head setup demonstrates emerging capability in text-reasoning tasks but would benefit from class-balanced sampling and longer training.
- Validation loss steadily decreased across epochs indicating effective gradient optimization.

---

## 🧾 WandB Experiment Tracking

**Run Name:** `jigsaw_exp_lmh_gemma_base`  
**Project:** `jigsaw_language_head_2025`  
**Entity:** `ronchaudhuri29-the-university-of-texas-at-dallas`  

🔗 **Project URL:**  
[wandb.ai/ronchaudhuri29-the-university-of-texas-at-dallas/jigsaw_language_head_2025](https://wandb.ai/ronchaudhuri29-the-university-of-texas-at-dallas/jigsaw_language_head_2025)

---

## Results Summary

- **Accuracy:** 0.5961  
- **Macro F1:** 0.3879  
- **Observation:** Model shows partial understanding of rule context but overpredicts “complies.”  
- Metrics and training loss were successfully logged to WandB for monitoring.  

---

## File Structure

```
PartB_Language_Head_Finetuning.ipynb
│
├── datasets/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
│
├── models/
│   └── jigsaw_exp_lmh_gemma_base/
│
├── logs/
│   └── wandb_run_summary.json
│
└── README.md
```

---

## Notes

- Part B extends Part A by replacing the classification head with a language-model head and fine-tuning via completion-only loss.  
- Quantization and LoRA reduced VRAM usage while retaining trainability on Colab A1000.  
- Results establish a functional baseline for the instruction-tuning phase in Part C.

# Part C – Instruction Fine-tuning using Chat Template


## Dataset Information

**Source:** Kaggle Competition Dataset (Reddit moderation comments)  

**Files Used:**
- `train.csv` → split into train / validation / test
- `test.csv` → used for instruction-style inference  

**Split Ratio:**

| Subset | Percentage | Purpose |
|:-------|:------------|:--------|
| Train | 80 % | model training |
| Validation | 10 % | model evaluation |
| Test | 10 % | held-out testing |

---

## Model Setup

**Architecture:** Google Gemma-2 2B (Decoder LLM)  
**Objective:** Binary classification (“complies” vs “violates”) through **instruction fine-tuning**  
**Format:** Chat-based instruction dataset (system + user + assistant roles)  

**Frameworks:**
- Hugging Face Transformers + TRL (SFTTrainer)
- PEFT (LoRA r = 32, α = 32, dropout = 0.05)
- Bits and Bytes (4-bit NF4 quantization)
- Weights & Biases for experiment tracking
- PyTorch + Evaluate for metrics  

**Instruction Template:**  
Each input sample was reformatted into a structured instruction-based chat format:

```
User: You are a content policy checker. Analyze the given COMMENT, RULE, 
and SUBREDDIT context below. Output exactly one word:
- 'complies' if the comment follows the subreddit rules, or
- 'violates' if it breaks them.

COMMENT: <comment_text>
RULE: <rule_text>
SUBREDDIT: <subreddit_name>

Assistant: <label_text>
```

---

## Hyperparameters

| Parameter | Value |
|:-----------|:------|
| Learning Rate | 1 × 10⁻⁵ |
| Batch Size (train / eval) | 2 / 2 |
| Accumulation Steps | 8 |
| Epochs | 2 |
| Precision | BF16 |
| Optimizer | AdamW (Torch) |
| Gradient Checkpointing | Enabled |
| Max Sequence Length | 1024 |
| Quantization | 4-bit (NF4) |
| Weight Decay | 0.0 |

---

## Training and Evaluation

Training was conducted on **Google Colab A1000 GPU** using instruction-formatted chat prompts.  
LoRA adapters were applied to key projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, etc.) with completion-only loss active.  

**Validation Set Results:**

| Metric | Score |
|:--------|:------|
| Accuracy | 0.7586 |
| F1 Macro | 0.7586 |

**Confusion Matrix**
```
[[76 26]
 [23 78]]
```

---

## Observations

- Instruction fine-tuning improved generalization and balance across classes.  
- The model demonstrated stronger contextual understanding due to explicit COMMENT–RULE–SUBREDDIT conditioning.  
- Validation F1 increased significantly compared to Part B (Language Model Head).  
- Generated completions were more consistent and semantically aligned with human moderation reasoning.  

---

## 🧾 WandB Experiment Tracking

**Run Name:** `jigsaw_inst_gemma_instruction_ft`  
**Project:** `jigsaw_inst_language_head2025`  
**Entity:** `ronchaudhuri29-the-university-of-texas-at-dallas`  

🔗 **Project URL:**  
[wandb.ai/ronchaudhuri29-the-university-of-texas-at-dallas/jigsaw_inst_language_head2025](https://wandb.ai/ronchaudhuri29-the-university-of-texas-at-dallas/jigsaw_inst_language_head2025)

---

## Results Summary

- **Accuracy:** 0.7586  
- **Macro F1:** 0.7586  
- **Observation:** Instruction fine-tuning yielded significant performance gains by leveraging structured, rule-based prompts with subreddit context.  
- All evaluation metrics logged to WandB for detailed performance tracking.  

---

## File Structure

```
PartC_Instruction_Finetuning.ipynb
│
├── datasets/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
│
├── models/
│   └── jigsaw_chat_gemma_instruction_ft/
│
├── logs/
│   └── wandb_run_summary.json
│
└── README.md
```

---

## Notes

- This stage extends Part B by introducing **instruction-style fine-tuning** with multi-turn chat templates.  
- The approach allows Gemma-2 to perform context-aware moderation and decision-making with improved interpretability.  
- LoRA + 4-bit quantization enabled efficient fine-tuning on Colab’s A1000 runtime.  
- The resulting model achieves robust contextual reasoning for binary rule classification.
