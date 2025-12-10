# IGB XQuAD-IN QA Task

## Overview

The `igb_xquad_in_gen` task evaluates language models on extractive question-answering using the IndicGenBench XQuAD-IN dataset in a generation format similar to SQuAD completion. Unlike the original `igb_xquad_lm` task which uses language modeling perplexity, this task requires models to generate answers based on context and questions.

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

This task supports 12 Indian languages plus English:

- `igb_xquad_in_gen_hi`: Hindi
- `igb_xquad_in_gen_en`: English  
- `igb_xquad_in_gen_as`: Assamese
- `igb_xquad_in_gen_bn`: Bengali
- `igb_xquad_in_gen_gu`: Gujarati
- `igb_xquad_in_gen_kn`: Kannada
- `igb_xquad_in_gen_ml`: Malayalam
- `igb_xquad_in_gen_mr`: Marathi
- `igb_xquad_in_gen_or`: Odia
- `igb_xquad_in_gen_pa`: Punjabi
- `igb_xquad_in_gen_ta`: Tamil
- `igb_xquad_in_gen_te`: Telugu

## Dataset

- **Source:** `google/IndicGenBench_xquad_in`
- **Splits:** train, validation, test
- **Language filtering:** Each task variant filters examples to only include the specified language

## Evaluation Metrics

The task uses two evaluation metrics:

1. **contains**: Checks if any of the correct answers appears anywhere in the generated text (case-insensitive substring match)
2. **exact_match**: Checks if the generated text exactly matches any of the correct answers (after normalization - stripping whitespace and lowercasing)

## Generation Parameters

- **until**: `["\n", "Context:", "Question:"]` - Stop generation at newlines or when context/question patterns appear
- **max_gen_toks**: `64` - Maximum number of tokens to generate

## Comparison with Related Tasks

### vs. `igb_xquad_lm` (Original)
- **Original**: Language modeling perplexity evaluation (loglikelihood_rolling)
- **This task**: Extractive QA generation (generate_until)
- **Advantage**: This task directly evaluates QA capability rather than just language modeling

### vs. `squad_completion`
- **SQuAD**: English-only, uses hazyresearch/based-squad dataset
- **This task**: Multilingual Indian languages, uses IndicGenBench XQuAD-IN
- **Similarity**: Both use generation-based evaluation with similar metrics

## Generating Configuration Files

The task includes utilities to automatically generate configuration files for all supported languages:

### Method 1: Using the Python generator script

```bash
cd lm_eval/tasks/gpt-oss/igb_xquad_in_gen
python generate_configs.py
```

This will:
- Validate that all language classes exist in `igb_xquad_in_gen.py`
- Generate individual YAML config files for each language
- Generate the main group YAML file
- Provide usage examples and summary

### Method 2: Using the quick regeneration script

```bash
cd lm_eval/tasks/gpt-oss/igb_xquad_in_gen
python regenerate_configs.py
```

This is a wrapper script that runs `generate_configs.py` with error handling.

### Adding a New Language

To add a new language:

1. Edit `generate_configs.py` and add the language to the `LANGUAGES` list:
   ```python
   LANGUAGES = [
       # ... existing languages ...
       ("xx", "New Language Name"),
   ]
   ```

2. Add the corresponding class to `igb_xquad_in_gen.py`:
   ```python
   class IGB_XQuad_IN_Gen_Xx(IGB_XQuad_IN_Gen_Lang):
       LANG = "xx"
   ```

3. Run the generator script:
   ```bash
   python generate_configs.py
   ```

## Usage

```bash
# Run all languages
lm_eval --model hf --model_args pretrained=your_model --tasks igb_xquad_in_gen

# Run specific language
lm_eval --model hf --model_args pretrained=your_model --tasks igb_xquad_in_gen_hi

# Run multiple languages
lm_eval --model hf --model_args pretrained=your_model --tasks igb_xquad_in_gen_hi,igb_xquad_in_gen_en
```

## Citation

If you use this task, please cite:

```
@inproceedings{singh-etal-2024-indicgenbench,
    title = "{I}ndic{G}en{B}ench: A Multilingual Benchmark to Evaluate Generation Capabilities of {LLM}s on {I}ndic Languages",
    author = "Singh, Harman and
    Gupta, Nitish and
    Bharadwaj, Shikhar and
    Tewari, Dinesh and
    Talukdar, Partha",
    editor = "Ku, Lun-Wei and
    Martins, Andre and
    Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.595/",
    doi = "10.18653/v1/2024.acl-long.595",
    pages = "11047--11073",
    abstract = "As large language models (LLMs) see increasing adoption across the globe, it is imperative for LLMs to be representative of the linguistic diversity of the world. India is a linguistically diverse country of 1.4 Billion people. To facilitate research on multilingual LLM evaluation, we release IndicGenBench {---} the largest benchmark for evaluating LLMs on user-facing generation tasks across a diverse set 29 of Indic languages covering 13 scripts and 4 language families. IndicGenBench is composed of diverse generation tasks like cross-lingual summarization, machine translation, and cross-lingual question answering. IndicGenBench extends existing benchmarks to many Indic languages through human curation providing multi-way parallel evaluation data for many under-represented Indic languages for the first time. We evaluate stateof-the-art LLMs like GPT-3.5, GPT-4, PaLM2, and LLaMA on IndicGenBench in a variety of settings. The largest PaLM-2 models performs the best on most tasks, however, there is a significant performance gap in all languages compared to English showing that further research is needed for the development of more inclusive multilingual language models. IndicGenBench isavailable at www.github.com/google-researchdatasets/indic-gen-bench"
}
```