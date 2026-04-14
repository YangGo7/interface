
import os

fn = r'c:\interface\backend\services\pano_inference.py'
with open(fn, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
target_string = 'gap_cx = ((b[0]+b[2])/2) + (dir_factor * w * 1.1)'
fix_logic = """
                          # [FIX Q1/Q2]
                          if q in [1, 2] and dir_factor == 0:
                              if q == 1: df_fix = -1 if ref_lbl < fdi else 1
                              elif q == 2: df_fix = 1 if ref_lbl < fdi else -1
                              gap_cx = ((b[0]+b[2])/2) + (df_fix * w * 1.1)
"""

inserted = False
for line in lines:
    new_lines.append(line)
    if target_string in line and not inserted:
        # Determine indentation
        indent = line[:line.find('gap_cx')]
        # Add Fix Logic (indented)
        indented_fix = ""
        for fix_line in fix_logic.strip().split('\n'):
            indented_fix += indent + fix_line.strip() + '\n'
            
        new_lines.append(indented_fix)
        inserted = True

with open(fn, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Applied fix.")
