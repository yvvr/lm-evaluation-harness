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
        with open(vtt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
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
            import re
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


def extract_metadata_from_path(vtt_path: Path, base_dir: Path) -> Dict[str, str]:
    """
    Extract metadata from the file path structure.
    Structure: ./hi-IN/year/source/channel_or_category/file.vtt
    
    Args:
        vtt_path: Path to the VTT file
        base_dir: Base directory (parent of language folders, e.g., '.')
        
    Returns:
        Dictionary with lang, year, source, category
    """
    try:
        # Get relative path from base directory
        rel_path = vtt_path.relative_to(base_dir)
        parts = rel_path.parts
        
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


def process_vtt_directory(base_dir: str, output_dir: str = None) -> None:
    """
    Process all VTT files in the directory structure and create JSON files per language.
    
    Args:
        base_dir: Base directory containing language folders (e.g., ./hi-IN-vtt)
        output_dir: Output directory for JSON files (defaults to base_dir)
    """
    base_path = Path(base_dir)
    
    if output_dir is None:
        output_dir = base_dir
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store data per language
    lang_data: Dict[str, Dict[str, Any]] = {}
    
    # Find all VTT files
    vtt_files = list(base_path.rglob('*.vtt'))
    
    print(f"Found {len(vtt_files)} VTT files to process")
    
    processed = 0
    for vtt_file in vtt_files:
        # Extract metadata from path
        metadata = extract_metadata_from_path(vtt_file, base_path)
        
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
            print(f"Processed {processed}/{len(vtt_files)} files...")
    
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
    
    args = parser.parse_args()
    
    process_vtt_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
