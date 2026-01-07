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


def count_words_in_vtt(vtt_path: str) -> int:
    """
    Count words in a VTT file without extracting the full text.
    
    Args:
        vtt_path: Path to the VTT file
        
    Returns:
        Word count
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
            
            # Check if line is a number (subtitle index) or contains only [Music] or similar
            if line.isdigit() or line in ['[संगीत]', '[Music]', '[music]']:
                continue
            
            # Remove timestamp tags like <00:00:00.480><c> and </c>
            clean_line = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', line)  # Remove timestamps
            clean_line = re.sub(r'</?c>', '', clean_line)  # Remove <c> and </c> tags
            clean_line = clean_line.strip()
            
            # Skip if line becomes empty after cleaning
            if not clean_line:
                continue
            
            # This is subtitle text
            text_parts.append(clean_line)
        
        # Join all text parts and count words
        combined_text = ' '.join(text_parts)
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        return len(combined_text.split()) if combined_text else 0
    
    except Exception as e:
        print(f"Error processing {vtt_path}: {e}")
        return 0


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
    
    # Setup logging to file
    log_file_path = output_path / 'processing.log'
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log(message):
        """Print to console and write to log file"""
        print(message, flush=True)
        log_file.write(message + '\n')
        log_file.flush()
    
    # Dictionary to store data per language
    lang_data: Dict[str, Dict[str, Any]] = {}
    
    # Count VTT files first (for progress reporting)
    log("Scanning for VTT files...")
    vtt_count = sum(1 for _ in base_path.rglob('*.vtt'))
    log(f"Found {vtt_count} VTT files to process")
    
    # Process files one at a time using generator (memory efficient)
    processed = 0
    for vtt_file in base_path.rglob('*.vtt'):
        # Extract metadata from path
        metadata = extract_metadata_from_path(vtt_file, base_path, single_lang)
        
        # Count words only (don't extract full text)
        word_count = count_words_in_vtt(str(vtt_file))
        
        if word_count == 0:
            log(f"Warning: No words found in {vtt_file}")
            continue
        
        # Initialize language structure if not exists
        lang = metadata['lang']
        if lang not in lang_data:
            lang_data[lang] = {
                'lang': lang,
                'total_words': 0,
                'categories': {}
            }
        
        # Update total word count
        lang_data[lang]['total_words'] += word_count
        
        # Update category word count
        category = metadata['category']
        if category not in lang_data[lang]['categories']:
            lang_data[lang]['categories'][category] = {
                'category': category,
                'word_count': 0
            }
        lang_data[lang]['categories'][category]['word_count'] += word_count
        
        processed += 1
        
        # Yield to heartbeat thread every 10 files (GIL release)
        if processed % 10 == 0:
            time.sleep(0.01)
        
        if processed % 100 == 0:
            log(f"Processed {processed}/{vtt_count} files...")
    
    # Convert categories dict to list and write JSON files per language
    for lang, data in lang_data.items():
        # Convert categories dict to list
        data['categories'] = list(data['categories'].values())
        
        output_file = output_path / f"{lang}_wordcount.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        log(f"Created {output_file} with {data['total_words']} total words across {len(data['categories'])} categories")
    
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
