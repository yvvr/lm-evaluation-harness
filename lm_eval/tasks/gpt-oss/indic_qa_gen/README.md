# IndicQA Generation Task

## Overview

The `indic_qa_gen` task evaluates language models on extractive question-answering using the ai4bharat/IndicQA dataset in a generation format similar to SQuAD completion. This task requires models to generate answers based on context and questions in various Indian languages.

## Task Format

**Input Format:**
```
Context: [passage text]
Question: [question text]
Answer:
```

**Expected Output:**
The model should generate the answer text that corresponds to the correct answer found in the passage.

## Supported Languages

This task supports 11 Indian languages:

- `indic_qa_gen_hi`: Hindi
- `indic_qa_gen_as`: Assamese
- `indic_qa_gen_bn`: Bengali
- `indic_qa_gen_gu`: Gujarati
- `indic_qa_gen_kn`: Kannada
- `indic_qa_gen_ml`: Malayalam
- `indic_qa_gen_mr`: Marathi
- `indic_qa_gen_or`: Odia
- `indic_qa_gen_pa`: Punjabi
- `indic_qa_gen_ta`: Tamil
- `indic_qa_gen_te`: Telugu

## Dataset

- **Source:** `ai4bharat/IndicQA`
- **Format:** JSON files with nested structure (data -> paragraphs -> qas)
- **Splits:** Only test split is available
- **Loading:** Uses Hugging Face Hub API to download raw JSON files

## Dataset Structure

The IndicQA dataset has a nested structure:
```json
{
  "version": 1.0,
  "data": [
    {
      "title": "",
      "paragraphs": [
        {
          "context": "passage text...",
          "qas": [
            {
              "id": 0,
              "category": "SHORT",
              "question": "question text?",
              "answers": [
                {
                  "text": "answer text",
                  "answer_start": 123
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

The task automatically flattens this structure into individual QA pairs for evaluation.

## Evaluation Metrics

The task uses the following evaluation metrics (standard for SQuAD-style QA):

1. **F1 Score**: Token-level F1 score between the generated answer and ground truth
   - Measures overlap between prediction and answer tokens
   - Primary metric for extractive QA tasks
   - Calculated as: `2 * (precision * recall) / (precision + recall)`

2. **Exact Match (EM)**: Binary metric checking if normalized answer exactly matches
   - Returns 1.0 if the answer matches exactly after normalization, 0.0 otherwise
   - Normalization includes: lowercasing, removing punctuation (including Indian language punctuation like ।), Unicode normalization (NFC), and whitespace cleanup

3. **Contains**: Checks if any of the correct answers appears anywhere in the generated text
   - Case-insensitive substring match
   - More lenient metric, useful for partial credit

### Answer Normalization

For F1 and Exact Match calculations, answers are normalized by:
- Unicode normalization (NFC)
- Converting to lowercase
- Removing punctuation (including Indian language punctuation like `।`)
- Normalizing whitespace

This normalization is standard for SQuAD-style QA evaluation and helps handle formatting variations.

## Generation Parameters

- **until**: `["\n", "Context:", "Question:"]` - Stop generation at newlines or when context/question patterns appear
- **max_gen_toks**: `64` - Maximum number of tokens to generate

## Answer Categories

IndicQA includes question categories:
- **SHORT**: Questions with short factual answers
- **NO**: Unanswerable questions (empty answer string)

## Comparison with Related Tasks

### vs. `igb_xquad_in_gen`
- **IndicQA**: Uses ai4bharat/IndicQA dataset, 11 languages (no English)
- **IGB XQuAD**: Uses google/IndicGenBench_xquad_in, 12 languages + English
- **Similarity**: Both use generation-based QA evaluation with similar metrics

### vs. `squad_completion`
- **SQuAD**: English-only
- **IndicQA**: 11 Indian languages
- **Similarity**: Both use extractive QA with generation format

## Technical Details

### Dataset Loading

Due to the deprecated loading script in the IndicQA dataset, this task uses the Hugging Face Hub API to directly download and load the raw JSON files:

```python
from huggingface_hub import hf_hub_download

filename = f"data/indicqa.{lang_code}.json"
local_path = hf_hub_download(
    repo_id="ai4bharat/IndicQA",
    filename=filename,
    repo_type="dataset"
)
```

### Data Flattening

The task automatically flattens the nested structure (data -> paragraphs -> qas) into individual QA examples for easier evaluation.

## Usage

```bash
# Run all languages
lm_eval --model hf --model_args pretrained=your_model --tasks indic_qa_gen

# Run specific language
lm_eval --model hf --model_args pretrained=your_model --tasks indic_qa_gen_hi

# Run multiple languages
lm_eval --model hf --model_args pretrained=your_model --tasks indic_qa_gen_hi,indic_qa_gen_bn
```

## Example

```bash
# Evaluate on Hindi IndicQA
lm_eval --model hf \
    --model_args pretrained=ai4bharat/indic-bert \
    --tasks indic_qa_gen_hi \
    --batch_size 4 \
    --output_path ./results
```

## Citation

If you use this task, please cite:

```bibtex
@inproceedings{doddapaneni-etal-2023-leaving,
    title = "Leaving No Context Behind: Efficient Prediction of Context-dependent Messages",
    author = "Doddapaneni, Sumanth  and
      Ramesh, Gowtham  and
      Kunchukuttan, Anoop  and
      Kumar, Pratyush  and
      Khapra, Mitesh M.",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```

## Notes

- The IndicQA dataset contains some unanswerable questions (category: "NO") with empty answer strings
- The task uses the `contains` metric which checks for substring matches
- All text matching is case-insensitive to handle various writing styles
- The dataset is loaded using direct file download due to deprecated loading scripts in the original dataset repository