#!/usr/bin/env python3
"""
Quick regeneration script for IGB XQuAD IN Gen configurations.
This is a wrapper around generate_configs.py with error handling.
"""

def main():
    """Run the configuration generator with error handling."""
    try:
        # Import and run the main generator
        from generate_configs import main as generate_main
        
        print("Regenerating IGB XQuAD IN Gen configurations...")
        result = generate_main()
        
        if result == 0:
            print("\nRegeneration completed successfully!")
        else:
            print("\nRegeneration failed!")
            
        return result
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("   Make sure generate_configs.py is in the same directory.")
        return 1
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("   Check generate_configs.py for issues.")
        return 1


if __name__ == "__main__":
    exit(main())