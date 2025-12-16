# Dakshina Transliteration Task

## Overview

This task evaluates **transliteration** from romanized (Latin) text to native Indic scripts using the **Dakshina** dataset.

**Dataset**: [google-research-datasets/dakshina](https://github.com/google-research-datasets/dakshina)

## Task Description

- **Input**: Romanized text (Latin script)
- **Output**: Native script (Devanagari, Bengali, Tamil, etc.)
- **Task Type**: `generate_until` (0-shot generation)
- **Evaluation**: Character Error Rate (CER), chrF++

## Supported Languages

Dakshina supports 12 South Asian languages:

| Code | Language   | Script       |
|------|------------|--------------|
| hi   | Hindi      | Devanagari   |
| bn   | Bengali    | Bengali      |
| gu   | Gujarati   | Gujarati     |
| kn   | Kannada    | Kannada      |
| ml   | Malayalam  | Malayalam    |
| mr   | Marathi    | Devanagari   |
| pa   | Punjabi    | Gurmukhi     |
| ta   | Tamil      | Tamil        |
| te   | Telugu     | Telugu       |
| ur   | Urdu       | Perso-Arabic |
| si   | Sinhala    | Sinhala      |
| ne   | Nepali     | Devanagari   |

## Setup Instructions

### 1. Clone the Dakshina Dataset

```bash
git clone https://github.com/google-research-datasets/dakshina.git
```

This will create a directory structure like:

```
dakshina/
└── dakshina_dataset_v1.0/
    ├── hi/
    │   └── lexicons/
    │       ├── hi.translit.sampled.train.tsv
    │       ├── hi.translit.sampled.dev.tsv
    │       └── hi.translit.sampled.test.tsv
    ├── bn/
    ├── gu/
    └── ... (other languages)
```

### 2. Set the Dataset Path

You have two options:

**Option A: Environment Variable (Recommended)**

```bash
export DAKSHINA_DATASET_PATH="/path/to/dakshina/dakshina_dataset_v1.0"
```

**Option B: CLI Argument**

Pass the path when running evaluation:

```bash
lm_eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks dakshina_gen_hi \
    --task_args dataset_path=/path/to/dakshina/dakshina_dataset_v1.0
```

## Dataset Format

Each TSV file contains two columns:

```
native_script<TAB>romanized
```

Example (Hindi):

```
भारत	bhaarat
नमस्ते	namaste
```

The task uses the **test split** (`.test.tsv` files).

## Usage

### Run Single Language

```bash
# Hindi
lm_eval --model hf \
    --model_args pretrained=ai4bharat/indic-bert \
    --tasks dakshina_gen_hi \
    --batch_size 8
```

### Run All Languages

```bash
lm_eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks dakshina_gen \
    --batch_size 8
```

### Specify Dataset Path via CLI

```bash
lm_eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks dakshina_gen_hi \
    --task_args dataset_path=/path/to/dakshina/dakshina_dataset_v1.0
```

## Metrics

### 1. Character Error Rate (CER)
- **Description**: Levenshtein distance normalized by reference length
- **Formula**: `CER = (insertions + deletions + substitutions) / len(reference)`
- **Range**: 0.0 to ∞ (typically 0.0 to 1.0)
- **Higher is better**: No (0.0 = perfect)
- **Use case**: Measures how many character edits are needed to match gold

### 2. chrF++
- **Description**: Character n-gram F-score with word n-grams
- **Details**: Combines character n-grams (1-6) and word n-grams (1-2)
- **Range**: 0.0 to 1.0
- **Higher is better**: Yes
- **Use case**: More granular evaluation of transliteration quality than CER

## Example Output

```json
{
  "results": {
    "dakshina_hi_in": {
      "cer": 0.12,
      "chrf++": 0.88
    },
    "dakshina_hi_latn": {
      "cer": 0.15,
      "chrf++": 0.85
    }
  }
}
```

## Prompt Format

```
Transliterate to {lang}:
{romanized_text}
Output:
```

Example:

```
Transliterate to hi:
bhaarat
Output:
```

Expected output: `भारत`

## Implementation Details

### File Structure
```
dakshina_gen/
├── dakshina_gen.py          # Main task implementation
├── dakshina_gen.yaml        # Group config (all languages)
├── dakshina_gen_hi.yaml     # Hindi config
├── dakshina_gen_bn.yaml     # Bengali config
└── ... (configs for other languages)
```

### Key Methods
- `_load_dataset()`: Loads TSV files from local path
- `doc_to_text()`: Formats the romanized input as a prompt
- `doc_to_target()`: Returns the native script target
- `char_error_rate()`: Calculates CER using Levenshtein distance

### Design Choices
1. **0-shot evaluation**: Transliteration should be deterministic
2. **Short prompts**: Minimal context needed for transliteration
3. **No HuggingFace upload**: Uses local TSV files directly
4. **Bidirectional tasks**: Both Indic→Latin (_in) and Latin→Indic (_latn)
5. **CER and chrF++**: Complementary metrics for character-level evaluation

## Troubleshooting

### Error: "dataset_path must be provided"

**Solution**: Set the environment variable or pass via CLI:

```bash
export DAKSHINA_DATASET_PATH="/path/to/dakshina/dakshina_dataset_v1.0"
```

### Error: "Test file not found"

**Solution**: Ensure the directory structure is correct:

```
dakshina_dataset_v1.0/
└── hi/
    └── lexicons/
        └── hi.translit.sampled.test.tsv
```

### Warning: "Skipping malformed line"

This is normal if the TSV file has empty lines or comments. The task will skip them.

## References

- **Paper**: [Dakshina: A Foundation Dataset for South Asian Languages](https://arxiv.org/abs/2007.01176)
- **Dataset**: https://github.com/google-research-datasets/dakshina
- **Citation**:

```bibtex
@inproceedings{roark2020dakshina,
  title={The Dakshina Dataset: A New Resource for Transliteration},
  author={Roark, Brian and Wolf-Sonkin, Lawrence and Kirov, Christo and Mielke, Sabrina J. and Johny, Cibu and Demirsahin, Isin and Hall, Keith},
  booktitle={Proceedings of LREC},
  year={2020}
}
```

## Notes

- The task uses **only the test split** (no training or validation)
- Each language has ~1000 test examples
- The dataset is designed for lexicon-level transliteration (single words)
- Romanization follows standard conventions for each language
