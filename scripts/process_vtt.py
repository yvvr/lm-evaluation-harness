#!/usr/bin/env python3
"""
Process VTT subtitle files and create a JSON dataset.
Extracts text from VTT files organized by language, year, source, and category.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any


def parse_vtt_file(vtt_path: str) -> str:
    """
    Parse a VTT file and extract all subtitle text.
    Removes timestamps, tags, and metadata.
    
    Args:
        vtt_path: Path to the VTT file
        
    Returns:
        Combined text from all subtitles, joined with spaces
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
            print(f"Warning: Used fallback decoding for {vtt_path}")
        
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
            # Pattern: <timestamp><c>text</c> or just <c>text</c>
            clean_line = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', line)  # Remove timestamps
            clean_line = re.sub(r'</?c>', '', clean_line)  # Remove <c> and </c> tags
            clean_line = clean_line.strip()
            
            # Skip if line becomes empty after cleaning
            if not clean_line:
                continue
            
            # This is subtitle text
            text_parts.append(clean_line)
        
        # Join all text parts with spaces
        combined_text = ' '.join(text_parts)
        
        # Clean up multiple spaces
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        return combined_text
    
    except Exception as e:
        print(f"Error processing {vtt_path}: {e}")
        return ""


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
    print("Scanning for VTT files...")
    vtt_count = sum(1 for _ in base_path.rglob('*.vtt'))
    print(f"Found {vtt_count} VTT files to process")
    
    # Process files one at a time using generator (memory efficient)
    processed = 0
    for vtt_file in base_path.rglob('*.vtt'):
        # Extract metadata from path
        metadata = extract_metadata_from_path(vtt_file, base_path, single_lang)
        
        # Parse VTT content
        text = parse_vtt_file(str(vtt_file))
        
        if not text:
            print(f"Warning: No text extracted from {vtt_file}")
            continue
        
        # Count words
        word_count = len(text.split())
        
        # Create transcript entry
        transcript = {
            'year': metadata['year'],
            'source': metadata['source'],
            'category': metadata['category'],
            'filename': metadata['filename'],
            'word_count': word_count,
            'data': text
        }
        
        # Initialize language structure if not exists
        lang = metadata['lang']
        if lang not in lang_data:
            lang_data[lang] = {
                'lang': lang,
                'total_words': 0,
                'categories': {},
                'transcripts': []
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
        
        # Add transcript
        lang_data[lang]['transcripts'].append(transcript)
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{vtt_count} files...")
    
    # Convert categories dict to list and write JSON files per language
    for lang, data in lang_data.items():
        # Convert categories dict to list
        data['categories'] = list(data['categories'].values())
        
        output_file = output_path / f"{lang}_dataset.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Created {output_file} with {len(data['transcripts'])} transcripts, {data['total_words']} total words")
    
    print(f"\nTotal processed: {processed} files")
    print(f"Languages: {', '.join(lang_data.keys())}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process VTT files and create JSON dataset'
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
