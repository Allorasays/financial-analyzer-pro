#!/usr/bin/env python3
"""
Fix script to add show_advanced definition to prevent NameError
"""

import os
import re

def fix_show_advanced_error(file_path):
    """Add show_advanced definition to prevent NameError"""
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if show_advanced is used but not defined
    if 'show_advanced' in content and 'show_advanced =' not in content:
        print(f"Found show_advanced usage in {file_path}")
        
        # Find the function that uses show_advanced
        # Look for patterns like "if show_advanced:" or "show_advanced"
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'if show_advanced:' in line or 'show_advanced' in line:
                # Find the function this line belongs to
                func_start = -1
                for j in range(i-1, -1, -1):
                    if lines[j].strip().startswith('def '):
                        func_start = j
                        break
                
                if func_start != -1:
                    # Add the definition after the function definition line
                    func_line = lines[func_start]
                    indent = len(func_line) - len(func_line.lstrip())
                    
                    # Add the show_advanced definition
                    definition = ' ' * (indent + 4) + 'show_advanced = st.sidebar.checkbox("Show Advanced Indicators", value=False)'
                    
                    # Insert after the function definition
                    lines.insert(func_start + 1, definition)
                    print(f"Added show_advanced definition after line {func_start + 1}")
                    break
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Fixed {file_path}")
        return True
    
    return False

def main():
    """Apply fix to all app files"""
    app_files = [f for f in os.listdir('.') if f.startswith('app') and f.endswith('.py')]
    
    fixed_files = []
    for app_file in app_files:
        if fix_show_advanced_error(app_file):
            fixed_files.append(app_file)
    
    if fixed_files:
        print(f"\nFixed {len(fixed_files)} files:")
        for f in fixed_files:
            print(f"  - {f}")
    else:
        print("No files needed fixing")

if __name__ == "__main__":
    main()
