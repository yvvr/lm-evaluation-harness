#!/usr/bin/env python3
"""
Process VTT subtitle files and create a JSON dataset with word count statistics only.
This version does NOT include the actual transcript text - only metadata and word counts.
"""

import os
import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Any


def aml_heartbeat(interval=300):
    """Send heartbeat every 5 minutes to prevent AML timeout"""
    while True:
        print("[AML] job alive", flush=True)
        time.sleep(interval)


# Start heartbeat thread
threading.Thread(target=aml_heartbeat, daemon=True).start()
print("[AML] heartbeat thread started", flush=True)


def detect_script(text: str, lang_code: str = "hi") -> str:
    """
    Detect script type from text (Devanagari vs Latin).
    Samples first 10 words for efficiency.
    
    Args:
        text: Text to analyze
        lang_code: Language code (e.g., 'hi' or 'hi-IN')
        
    Returns:
        Script tag in format 'lang-Script' (e.g., 'hi-Deva' or 'hi-Latn')
    """
    # Extract base language code (e.g., 'hi' from 'hi-IN')
    base_lang = lang_code.split('-')[0]
    
    # Sample first 10 words for efficiency
    words = text.split()[:10]
    sample_text = ' '.join(words)
    
    # Check for Devanagari characters (U+0900 – U+097F)
    if re.search(r'[\u0900-\u097F]', sample_text):
        return f"{base_lang}-Deva"  # Devanagari script
    elif re.search(r'[A-Za-z]', sample_text):
        return f"{base_lang}-Latn"  # Latin (Roman) script
    else:
        # If mixed or unknown, default to Devanagari
        return f"{base_lang}-Deva"


def count_words_in_vtt(vtt_path: str) -> tuple:
    """
    Count words in a VTT file and extract text for script detection.
    
    Args:
        vtt_path: Path to the VTT file
        
    Returns:
        Tuple of (word_count, combined_text)
    """
    try:
        # Try UTF-8 first, fall back to other encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(vtt_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break  # Success, stop trying
            except (UnicodeDecodeError, LookupError):
                continue  # Try next encoding
        
        if content is None:
            # Last resort: read as binary and decode with errors='ignore'
            with open(vtt_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
        
        # Split into lines
        lines = content.split('\n')
        
        text_parts = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, WEBVTT header, Kind, Language, and timestamp lines
            if not line or line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                continue
            
            # Check if line is a timestamp (format: 00:00:00.000 --> 00:00:00.000)
            if '-->' in line:
                continue
            
            # Check if line is a number (subtitle index)
            if line.isdigit():
                continue
            
            # Remove timestamp tags like <00:00:00.480> and all HTML-like tags
            clean_line = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', line)  # Remove timestamps
            clean_line = re.sub(r'<[^>]+>', '', clean_line)  # Remove all HTML-like tags (<c>, </c>, <b>, <it>, <bold>, etc.)
            
            # Remove music/sound annotations in brackets or parentheses
            # Examples: [संगीत], [Music], [music], (संगीत), (Music), etc.
            clean_line = re.sub(r'[\[\(][^\]\)]*(?:संगीत|Music|music|MUSIC|Applause|applause|Laughter|laughter)[^\]\)]*[\]\)]', '', clean_line, flags=re.IGNORECASE)
            
            clean_line = clean_line.strip()
            
            # Skip if line becomes empty after cleaning
            if not clean_line:
                continue
            
            # This is subtitle text
            text_parts.append(clean_line)
        
        # Join all text parts and count words
        combined_text = ' '.join(text_parts)
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        word_count = len(combined_text.split()) if combined_text else 0
        return (word_count, combined_text)
    
    except Exception as e:
        print(f"Error processing {vtt_path}: {e}")
        return (0, "")


def extract_metadata_from_path(vtt_path: Path, base_dir: Path, single_lang: str = None) -> Dict[str, str]:
    """
    Extract metadata from the file path structure.
    Structure: ./hi-IN/year/source/channel_or_category/file.vtt
    
    Args:
        vtt_path: Path to the VTT file
        base_dir: Base directory (parent of language folders, e.g., '.')
        single_lang: If provided, treat base_dir as a single language directory
        
    Returns:
        Dictionary with lang, year, source, category
    """
    try:
        # Get relative path from base directory
        rel_path = vtt_path.relative_to(base_dir)
        parts = rel_path.parts
        
        # If single_lang is specified, use it as the language
        if single_lang:
            # Structure: year/source/channel_or_category/file.vtt (4 parts)
            if len(parts) >= 4:
                return {
                    'lang': single_lang,
                    'year': parts[0],
                    'source': parts[1],
                    'category': 'unknown' if parts[2].startswith('UC') else parts[2],
                    'filename': parts[-1]
                }
            elif len(parts) == 3:
                # Structure without category: year/source/file.vtt
                return {
                    'lang': single_lang,
                    'year': parts[0],
                    'source': parts[1],
                    'category': 'unknown',
                    'filename': parts[-1]
                }

        # Expected structure: hi-IN/year/source/channel_or_category/file.vtt (5 parts)
        if len(parts) >= 5:
            lang = parts[0]
            year = parts[1]
            source = parts[2]
            category_or_channel = parts[3]
            
            # If it's a YouTube channel ID (starts with UC), mark as unknown
            if category_or_channel.startswith('UC'):
                category = 'unknown'
            else:
                category = category_or_channel
            
            return {
                'lang': lang,
                'year': year,
                'source': source,
                'category': category,
                'filename': parts[-1]
            }
        elif len(parts) == 4:
            # Structure without category: lang/year/source/file.vtt
            return {
                'lang': parts[0],
                'year': parts[1],
                'source': parts[2],
                'category': 'unknown',
                'filename': parts[-1]
            }
        else:
            print(f"Warning: Unexpected path structure for {rel_path}")
            return {
                'lang': parts[0] if len(parts) > 0 else 'unknown',
                'year': parts[1] if len(parts) > 1 else 'unknown',
                'source': parts[2] if len(parts) > 2 else 'unknown',
                'category': 'unknown',
                'filename': vtt_path.name
            }
    except Exception as e:
        print(f"Error extracting metadata from {vtt_path}: {e}")
        return {
            'lang': 'unknown',
            'year': 'unknown',
            'source': 'unknown',
            'category': 'unknown',
            'filename': vtt_path.name
        }


def should_process_vtt(vtt_path: Path) -> bool:
    """
    Check if a VTT file should be processed.
    Only process base .vtt files, ignore language-specific ones like .en-US.vtt, .en-GB.vtt, etc.
    
    Args:
        vtt_path: Path to the VTT file
        
    Returns:
        True if file should be processed, False otherwise
    """
    filename = vtt_path.name
    
    # Pattern to match language codes before .vtt extension
    # Examples: .en-US.vtt, .en-GB.vtt, .de.vtt, .en.vtt, .es.vtt, .fr.vtt
    # We want to skip these and only process files ending with just .vtt
    import re
    
    # Check if filename has a language code pattern before .vtt
    # Language codes can be: en, en-US, en-GB, de, es, fr, etc.
    # Pattern: word boundary, 2-3 letter code, optional dash and 2 letter country code, then .vtt
    lang_code_pattern = r'\.[a-z]{2}(-[A-Z]{2})?\.vtt$'
    
    if re.search(lang_code_pattern, filename):
        return False  # Skip language-specific files
    
    return True  # Process base .vtt files


def process_vtt_directory(base_dir: str, output_dir: str = None, single_lang: str = None) -> None:
    """
    Process all VTT files in the directory structure and create JSON files per language.
    This version only includes word counts, not the actual transcript text.
    
    Args:
        base_dir: Base directory containing language folders or single language directory
        output_dir: Output directory for JSON files (defaults to base_dir)
        single_lang: If provided, treat base_dir as a single language directory with this language code
    """
    base_path = Path(base_dir)
    
    if output_dir is None:
        output_dir = base_dir
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store data per language
    lang_data: Dict[str, Dict[str, Any]] = {}
    
    # Count VTT files first (for progress reporting)
    print("Scanning for VTT files...", flush=True)
    all_vtt_files = list(base_path.rglob('*.vtt'))
    vtt_files_to_process = [f for f in all_vtt_files if should_process_vtt(f)]
    vtt_count = len(vtt_files_to_process)
    
    # Determine language for log filename
    if single_lang:
        lang_for_log = single_lang
    elif vtt_files_to_process:
        # Extract language from first file's path
        first_metadata = extract_metadata_from_path(vtt_files_to_process[0], base_path, single_lang)
        lang_for_log = first_metadata['lang']
    else:
        lang_for_log = 'unknown'
    
    # Setup logging to file with language-specific name
    log_file_path = output_path / f'{lang_for_log}_wordcount_processing.log'
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log(message):
        """Print to console and write to log file"""
        print(message, flush=True)
        log_file.write(message + '\n')
        log_file.flush()
    
    log(f"Found {len(all_vtt_files)} total VTT files, {vtt_count} to process (skipping {len(all_vtt_files) - vtt_count} language-specific files)")
    
    # Process files one at a time using generator (memory efficient)
    processed = 0
    for vtt_file in vtt_files_to_process:
        # Extract metadata from path
        metadata = extract_metadata_from_path(vtt_file, base_path, single_lang)
        
        # Count words and get text for script detection
        word_count, text = count_words_in_vtt(str(vtt_file))
        
        if word_count == 0:
            log(f"Warning: No words found in {vtt_file}")
            continue
        
        # Detect script from text
        script = detect_script(text, metadata['lang'])
        
        # Initialize language structure if not exists
        lang = metadata['lang']
        if lang not in lang_data:
            lang_data[lang] = {
                'lang': lang,
                'total_words': 0,
                'categories': {},
                'scripts': {}  # Track script types
            }
        
        # Update total word count
        lang_data[lang]['total_words'] += word_count
        
        # Update script word count
        if script not in lang_data[lang]['scripts']:
            lang_data[lang]['scripts'][script] = {
                'script': script,
                'word_count': 0
            }
        lang_data[lang]['scripts'][script]['word_count'] += word_count
        
        # Update category word count
        category = metadata['category']
        if category not in lang_data[lang]['categories']:
            lang_data[lang]['categories'][category] = {
                'category': category,
                'word_count': 0,
                'scripts': {}  # Track scripts per category
            }
        lang_data[lang]['categories'][category]['word_count'] += word_count
        
        # Update script count within category
        if script not in lang_data[lang]['categories'][category]['scripts']:
            lang_data[lang]['categories'][category]['scripts'][script] = 0
        lang_data[lang]['categories'][category]['scripts'][script] += word_count
        
        processed += 1
        
        # Yield to heartbeat thread every 10 files (GIL release)
        if processed % 10 == 0:
            time.sleep(0.01)
        
        if processed % 100 == 0:
            log(f"Processed {processed}/{vtt_count} files...")
    
    # Convert categories and scripts dicts to lists and write JSON files per language
    for lang, data in lang_data.items():
        # Convert categories dict to list
        data['categories'] = list(data['categories'].values())
        # Convert scripts dict to list
        data['scripts'] = list(data['scripts'].values())
        
        output_file = output_path / f"{lang}_wordcount.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Log script breakdown
        script_summary = ', '.join([f"{s['script']}: {s['word_count']}" for s in data['scripts']])
        log(f"Created {output_file} with {data['total_words']} total words across {len(data['categories'])} categories")
        log(f"  Script breakdown: {script_summary}")
    
    log(f"\nTotal processed: {processed} files")
    log(f"Languages: {', '.join(lang_data.keys())}")
    
    # Close log file
    log_file.close()
    print(f"\nLog saved to: {log_file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process VTT files and create JSON with word count statistics (no transcript text)'
    )
    parser.add_argument(
        'input_dir',
        help='Input directory containing language folders with VTT files (e.g., . or parent of hi-IN)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for JSON files (default: same as input_dir)',
        default=None
    )
    parser.add_argument(
        '--single-lang',
        help='Language code if input_dir contains a single language (e.g., hi-IN). Use when input_dir IS the language folder.',
        default=None
    )
    
    args = parser.parse_args()
    
    process_vtt_directory(args.input_dir, args.output_dir, args.single_lang)


if __name__ == '__main__':
    main()
