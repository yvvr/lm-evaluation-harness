#!/usr/bin/env python3
"""
Script to generate IGB XQuAD IN Gen language configuration files.

This script generates:
1. Individual YAML config files for each language
2. Updates the main group YAML file with all languages
3. Validates that all language classes exist in task.py

Note: Unlike the original igb_xquad_lm, we don't need to generate Python classes
since we use a simpler inheritance pattern with LANG class variables.
"""

import os
from pathlib import Path

# Language configurations (same as original igb_xquad_lm)
LANGUAGES = [
    ("as", "Assamese"),
    ("bn", "Bengali"), 
    ("en", "English"),
    ("gu", "Gujarati"),
    ("hi", "Hindi"),
    ("kn", "Kannada"),
    ("ml", "Malayalam"),
    ("mr", "Marathi"),
    ("or", "Odia"),
    ("pa", "Punjabi"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
]

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


def generate_yaml_config(lang_code):
    """Generate individual YAML config file for a language."""
    class_name = f"IGB_XQuad_IN_Gen_{lang_code.capitalize()}"
    content = f"""task: igb_xquad_in_gen_{lang_code}
class: !function igb_xquad_in_gen.{class_name}
"""
    return content


def generate_group_yaml():
    """Generate the main group YAML file."""
    tasks = [f"  - igb_xquad_in_gen_{lang_code}" for lang_code, _ in LANGUAGES]
    content = f"""group: igb_xquad_in_gen
task:
{chr(10).join(tasks)}
aggregate_metric_list:
  - aggregation: mean
    metric: contains
    weight_by_size: true
"""
    return content


def write_yaml_configs():
    """Write individual YAML config files for all languages."""
    print("Generating individual YAML config files...")
    for lang_code, lang_name in LANGUAGES:
        filename = SCRIPT_DIR / f"igb_xquad_in_gen_{lang_code}.yaml"
        content = generate_yaml_config(lang_code)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Created: {filename.name} ({lang_name})")


def write_group_yaml():
    """Write the main group YAML file."""
    print("\nGenerating main group YAML file...")
    filename = SCRIPT_DIR / "igb_xquad_in_gen.yaml"
    content = generate_group_yaml()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Created: {filename.name}")


def validate_task_classes():
    """Validate that all required language classes exist in igb_xquad_in_gen.py."""
    print("\nValidating language classes in igb_xquad_in_gen.py...")
    
    try:
        task_file = SCRIPT_DIR / "igb_xquad_in_gen.py"
        with open(task_file, 'r', encoding='utf-8') as f:
            task_content = f.read()
        
        missing_classes = []
        for lang_code, lang_name in LANGUAGES:
            class_name = f"IGB_XQuad_IN_Gen_{lang_code.capitalize()}"
            if f"class {class_name}" not in task_content:
                missing_classes.append((lang_code, class_name, lang_name))
        
        if missing_classes:
            print("  Missing language classes:")
            for lang_code, class_name, lang_name in missing_classes:
                print(f"    - {class_name} for {lang_name} ({lang_code})")
            return False
        else:
            print("  All language classes found in igb_xquad_in_gen.py")
            return True
            
    except Exception as e:
        print(f"  Error validating igb_xquad_in_gen.py: {e}")
        return False


def print_summary():
    """Print summary of generated files."""
    print("\n" + "="*80)
    print("GENERATED FILES SUMMARY:")
    print("="*80)
    
    print(f"\nDirectory: {SCRIPT_DIR}")
    print(f"Languages: {len(LANGUAGES)} supported")
    print(f"YAML files: {len(LANGUAGES)} individual + 1 group file")
    
    print("\nIndividual language files:")
    for lang_code, lang_name in LANGUAGES:
        print(f"  - igb_xquad_in_gen_{lang_code}.yaml ({lang_name})")
    
    print("\nGroup file:")
    print("  - igb_xquad_in_gen.yaml (contains all languages)")
    
    print("\nUsage:")
    print("  lm_eval --tasks igb_xquad_in_gen           # All languages")
    print("  lm_eval --tasks igb_xquad_in_gen_hi        # Hindi only")
    print("  lm_eval --tasks igb_xquad_in_gen_hi,igb_xquad_in_gen_en  # Multiple")


def main():
    """Main function to generate all configuration files."""
    print("IGB XQuAD IN Gen Configuration Generator")
    print("="*80)
    print(f"Generating configs for {len(LANGUAGES)} languages")
    print(f"Target directory: {SCRIPT_DIR}")
    print()
    
    # Validate task classes first
    if not validate_task_classes():
        print("\nValidation failed! Please check igb_xquad_in_gen.py")
        print("   Make sure all language classes are properly defined.")
        return 1
    
    # Generate individual YAML configs
    write_yaml_configs()
    
    # Generate group YAML
    write_group_yaml()
    
    # Print summary
    print_summary()
    
    print("\n" + "="*80)
    print("Configuration generation complete!")
    print("="*80)
    print("\nNote: Unlike the original igb_xquad_lm task, we don't need to")
    print("   manually add Python classes since our igb_xquad_in_gen.py uses")
    print("   a simpler inheritance pattern with LANG class variables.")
    
    return 0


if __name__ == "__main__":
    exit(main())