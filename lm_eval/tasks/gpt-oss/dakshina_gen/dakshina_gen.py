import csv
import os
import unicodedata
from copy import deepcopy
from typing import List

import numpy as np
from datasets import Dataset
from sacrebleu.metrics import CHRF

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


class Dakshina_Gen_Base(ConfigurableTask):
    """
    Base class for Dakshina transliteration tasks.
    
    Dataset: https://github.com/google-research-datasets/dakshina
    
    The dataset path provided in the config.
    Each language has TSV files with format: native_script<TAB>romanized
    
    Subclasses should define:
    - DIRECTION: "romanization" (Indic→Latin) or "nativization" (Latin→Indic)
    """
    VERSION = 1
    DATASET_PATH = None  # Path to local dakshina dataset
    DATASET_NAME = None  # Will be set by language-specific subclasses
    DIRECTION = None  # "romanization" (Indic→Latin) or "nativization" (Latin→Indic)
    
    def __init__(self, config=None):
        super().__init__(config=config)
        self._dataset = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """Load dataset from local TSV files instead of HuggingFace"""
        # Don't call super().download() - we load manually from TSV
        self._dataset = self._load_dataset()

    def _load_dataset(self):
        """Load the TSV file for the specific language"""
        if not hasattr(self.config, 'dataset_path') or not self.config.dataset_path:
            raise ValueError(
                "dataset_path must be provided in config. "
                "Please download https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar "
                "and provide the path to dakshina_dataset_v1.0/"
            )
        
        lang = self.config.metadata["lang"]
        base_path = self.config.dataset_path
        
        # Path format: dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.tsv
        test_path = os.path.join(
            base_path, 
            lang, 
            "romanized", 
            f"{lang}.romanized.rejoined.tsv"
        )
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test file not found: {test_path}\n"
                f"Please ensure dakshina dataset is downloaded at: {base_path}"
            )
        
        rows = []
        with open(test_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, 1):
                if len(row) != 2:
                    print(f"Warning: Skipping malformed line {line_num} in {test_path}")
                    continue
                native, roman = row
                
                # Set source/target based on direction
                if self.DIRECTION == "romanization":
                    # Native → Roman
                    source = native.strip()
                    target = roman.strip()
                elif self.DIRECTION == "nativization":
                    # Roman → Native
                    source = roman.strip()
                    target = native.strip()
                else:
                    raise ValueError(f"Invalid DIRECTION: {self.DIRECTION}")
                
                rows.append({
                    "source": source,
                    "target": target,
                    "lang": lang,
                })
        
        return {"test": Dataset.from_list(rows)}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return []

    def validation_docs(self):
        return []

    def test_docs(self):
        """Return test documents from the dataset"""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset["test"]

    def doc_to_text(self, doc):
        """Format the prompt based on direction"""
        lang = doc.get("lang", self.config.metadata["lang"])
        if self.DIRECTION == "romanization":
            return f"Transliterate the following text into romanized {lang}.\nInput: {doc['source']}\nOutput:"
        else:  # nativization
            return f"Transliterate the following text into {lang}.\nInput: {doc['source']}\nOutput:"

    def doc_to_target(self, doc):
        """Extract the target"""
        return doc["target"]

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        """Construct generation request"""
        arguments = deepcopy(self.config.generation_kwargs) if hasattr(self.config, 'generation_kwargs') and self.config.generation_kwargs else {}
        arguments["until"] = arguments.get("until", ["\n"])
        arguments["max_gen_toks"] = arguments.get("max_gen_toks", 32)
        
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, arguments),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        """Calculate metrics for transliteration"""
        # Apply Unicode NFC normalization for fair comparison
        # This handles canonical equivalences, combining marks, nukta, vowel signs, etc.
        prediction = unicodedata.normalize("NFC", results[0].strip())
        target = unicodedata.normalize("NFC", self.doc_to_target(doc).strip())
        
        # Character Error Rate (CER)
        cer = char_error_rate(prediction, target)
        
        # chrF++ (character n-gram F-score with word n-grams)
        chrf = chrf_plus_plus(prediction, target)
        
        return {
            "cer": cer,
            "chrf++": chrf,
        }

    def aggregation(self):
        """Aggregation functions for metrics"""
        return {
            "cer": np.mean,
            "chrf++": np.mean,
        }

    def higher_is_better(self):
        """Which direction is better for each metric"""
        return {
            "cer": False,  # Lower CER is better
            "chrf++": True,  # Higher chrF++ is better
        }


# ============================================================================
# INDIC → LATIN (Native Script → Roman, e.g., भारत → bhaarat)
# ============================================================================

class Dakshina_Gen_In(Dakshina_Gen_Base):
    """
    Dakshina Indic to Latin task: Native script → Romanized text
    """
    VERSION = 1
    DIRECTION = "romanization"
    
    COMMON_CONFIG = {
        "metadata": {"version": VERSION},
        "task": "dakshina_gen_in",
        "tag": "dakshina_gen_in",
        "output_type": "generate_until",
    }


class Dakshina_Gen_In_Lang(Dakshina_Gen_In):
    """Base class for language-specific Indic to Latin tasks"""
    
    LANG = None  # To be overridden by subclasses
    
    def __init__(self, config=None):
        import copy

        lang_config = copy.deepcopy(self.COMMON_CONFIG)
        lang_config["task"] = f"dakshina_{self.LANG}_in"
        lang_config["metadata"] = {
            **lang_config.get("metadata", {}),
            "lang": self.LANG,
        }
        
        dataset_path = os.environ.get("DAKSHINA_DATASET_PATH")
        if dataset_path:
            lang_config["dataset_path"] = dataset_path

        super().__init__(config=lang_config)

    def task_lang(self):
        return self.LANG


# Indic to Latin language-specific classes
class Dakshina_Hi_In(Dakshina_Gen_In_Lang):
    LANG = "hi"

class Dakshina_Bn_In(Dakshina_Gen_In_Lang):
    LANG = "bn"

class Dakshina_Gu_In(Dakshina_Gen_In_Lang):
    LANG = "gu"

class Dakshina_Kn_In(Dakshina_Gen_In_Lang):
    LANG = "kn"

class Dakshina_Ml_In(Dakshina_Gen_In_Lang):
    LANG = "ml"

class Dakshina_Mr_In(Dakshina_Gen_In_Lang):
    LANG = "mr"

class Dakshina_Pa_In(Dakshina_Gen_In_Lang):
    LANG = "pa"

class Dakshina_Ta_In(Dakshina_Gen_In_Lang):
    LANG = "ta"

class Dakshina_Te_In(Dakshina_Gen_In_Lang):
    LANG = "te"

class Dakshina_Ur_In(Dakshina_Gen_In_Lang):
    LANG = "ur"

class Dakshina_Si_In(Dakshina_Gen_In_Lang):
    LANG = "si"

class Dakshina_Sd_In(Dakshina_Gen_In_Lang):
    LANG = "sd"


# ============================================================================
# LATIN → INDIC (Roman → Native Script, e.g., bhaarat → भारत)
# ============================================================================

class Dakshina_Gen_Latn(Dakshina_Gen_Base):
    """
    Dakshina Latin to Indic task: Romanized text → Native script
    """
    VERSION = 1
    DIRECTION = "nativization"
    
    COMMON_CONFIG = {
        "metadata": {"version": VERSION},
        "task": "dakshina_gen_latn",
        "tag": "dakshina_gen_latn",
        "output_type": "generate_until",
    }


class Dakshina_Gen_Latn_Lang(Dakshina_Gen_Latn):
    """Base class for language-specific Latin to Indic tasks"""
    
    LANG = None  # To be overridden by subclasses
    
    def __init__(self, config=None):
        import copy

        lang_config = copy.deepcopy(self.COMMON_CONFIG)
        lang_config["task"] = f"dakshina_{self.LANG}_latn"
        lang_config["metadata"] = {
            **lang_config.get("metadata", {}),
            "lang": self.LANG,
        }
        
        dataset_path = os.environ.get("DAKSHINA_DATASET_PATH")
        if dataset_path:
            lang_config["dataset_path"] = dataset_path

        super().__init__(config=lang_config)

    def task_lang(self):
        return self.LANG


# Latin to Indic language-specific classes
class Dakshina_Hi_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "hi"

class Dakshina_Bn_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "bn"

class Dakshina_Gu_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "gu"

class Dakshina_Kn_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "kn"

class Dakshina_Ml_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "ml"

class Dakshina_Mr_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "mr"

class Dakshina_Pa_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "pa"

class Dakshina_Ta_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "ta"

class Dakshina_Te_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "te"

class Dakshina_Ur_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "ur"

class Dakshina_Si_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "si"

class Dakshina_Sd_Latn(Dakshina_Gen_Latn_Lang):
    LANG = "sd"


# ============================================================================
# Utility Functions
# ============================================================================

# Initialize chrF++ metric using SacreBLEU (standard implementation)
_chrf_metric = CHRF(word_order=2, char_order=6, beta=2)


def chrf_plus_plus(prediction: str, reference: str) -> float:
    """
    Calculate chrF++ score using SacreBLEU's standard implementation.
    
    chrF++ is a character n-gram F-score metric that:
    - Computes F-score per n-gram order (char 1-6, word 1-2)
    - Averages across n-gram orders
    - Uses β=2 (recall-weighted)
    
    Args:
        prediction: Predicted text
        reference: Reference/target text
    
    Returns:
        chrF++ score (0.0 to 1.0, higher is better)
    
    Note:
        Uses SacreBLEU for standard, comparable scores.
    """
    if not reference:
        return 1.0 if not prediction else 0.0
    
    if not prediction:
        return 0.0
    
    # SacreBLEU expects reference as a list
    score = _chrf_metric.sentence_score(prediction, [reference]).score
    
    # Convert from 0-100 scale to 0-1 scale
    return score / 100.0


def char_error_rate(prediction: str, reference: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance.
    
    CER = (insertions + deletions + substitutions) / len(reference)
    
    Lower is better (0.0 = perfect match)
    """
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    
    # Calculate Levenshtein distance
    d = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(prediction) + 1):
        d[0][j] = j
    
    for i in range(1, len(reference) + 1):
        for j in range(1, len(prediction) + 1):
            if reference[i - 1] == prediction[j - 1]:
                cost = 0
            else:
                cost = 1
            
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost  # substitution
            )
    
    return d[len(reference)][len(prediction)] / len(reference)
