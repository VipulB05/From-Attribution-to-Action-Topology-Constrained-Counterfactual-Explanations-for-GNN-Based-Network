"""
Inspect Abilene files (no extensions).
"""

from pathlib import Path


def inspect_no_extension_files():
    """Inspect files without extensions."""
    
    print("="*80)
    print("ABILENE RAW FILES INSPECTION")
    print("="*80)
    
    abilene_path = Path('data/raw/abilene')
    
    # Get all files (no extension filter)
    all_items = sorted(list(abilene_path.iterdir()))
    files = [f for f in all_items if f.is_file()]
    
    print(f"\nFound {len(files)} files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  📄 {f.name} ({size_kb:.1f} KB)")
    
    # Inspect each file
    for file_path in files:
        print(f"\n{'='*80}")
        print(f"FILE: {file_path.name}")
        print(f"{'='*80}")
        
        # Try to read as text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"\n✓ Readable as text")
            print(f"  Lines: {len(lines)}")
            print(f"  Size: {file_path.stat().st_size / 1024:.2f} KB")
            
            # Show first 15 lines
            print(f"\nFirst 15 lines:")
            print("-" * 80)
            for i, line in enumerate(lines[:15], 1):
                display_line = line.rstrip()
                if len(display_line) > 75:
                    display_line = display_line[:75] + "..."
                print(f"{i:3d}: {display_line}")
            
            if len(lines) > 15:
                print(f"... ({len(lines) - 15} more lines)")
            
            # Analyze format
            print(f"\nFormat analysis:")
            analyze_format(lines, file_path.name)
            
        except UnicodeDecodeError:
            print(f"\n✗ Not a text file (binary)")
            
            # Try to peek at binary
            with open(file_path, 'rb') as f:
                header = f.read(100)
            
            print(f"  First 100 bytes (hex): {header[:50].hex()}")
            print(f"  First 100 bytes (ascii): {header[:50]}")
        
        except Exception as e:
            print(f"\n✗ Error reading: {e}")
        
        print()


def analyze_format(lines, filename):
    """Analyze the format of the file."""
    
    # Skip empty lines and comments
    data_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
    
    if not data_lines:
        print("  No data lines (all empty/comments)")
        return
    
    first_line = data_lines[0]
    print(f"  First data line: {first_line[:80]}")
    
    # Detect delimiter
    if '\t' in first_line:
        delimiter = 'TAB'
        values = first_line.split('\t')
    elif ',' in first_line:
        delimiter = 'COMMA'
        values = first_line.split(',')
    else:
        delimiter = 'SPACE'
        values = first_line.split()
    
    print(f"  Delimiter: {delimiter}")
    print(f"  Values per line: {len(values)}")
    print(f"  Sample values: {values[:5]}")
    
    # Try to parse as numbers
    try:
        numbers = [float(v.strip()) for v in values if v.strip()]
        print(f"  ✓ Parseable as floats")
        print(f"  Number range: {min(numbers):.2f} to {max(numbers):.2f}")
        
        # Specific format detection based on filename
        if 'TM' in filename:
            print(f"\n  → This is likely TRAFFIC MATRIX data")
            if len(numbers) == 144:
                print(f"     Format: Flattened 12×12 matrix (one per line)")
            elif len(numbers) == 12:
                print(f"     Format: One row per line (12 matrices total)")
        
        elif 'shortest' in filename.lower():
            print(f"\n  → This is likely SHORTEST PATHS data")
        
        elif filename == 'Abilene':
            print(f"\n  → This is likely TOPOLOGY/METADATA")
    
    except:
        print(f"  ✗ Not numeric data")
        print(f"  Might be: topology info, metadata, or labels")
    
    # Check consistency
    line_lengths = []
    for line in data_lines[:100]:  # Check first 100
        if delimiter == 'TAB':
            vals = line.split('\t')
        elif delimiter == 'COMMA':
            vals = line.split(',')
        else:
            vals = line.split()
        line_lengths.append(len(vals))
    
    if len(set(line_lengths)) == 1:
        print(f"  ✓ Consistent: all {len(line_lengths)} lines have {line_lengths[0]} values")
    else:
        print(f"  ⚠ Inconsistent: found line lengths {set(line_lengths)}")


if __name__ == "__main__":
    inspect_no_extension_files()