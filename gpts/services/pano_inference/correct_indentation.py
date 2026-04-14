
import os

fn = r'c:\interface\backend\services\pano_inference.py'
with open(fn, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
target_string_1 = 'if q == 1: df_fix = -1 if ref_lbl < fdi else 1'
target_string_2 = 'elif q == 2: df_fix = 1 if ref_lbl < fdi else -1'
target_string_3 = 'gap_cx = ((b[0]+b[2])/2) + (df_fix * w * 1.1)'

prev_indent = ""

for line in lines:
    lstrip = line.lstrip()
    if target_string_1 in line:
        # Check indentation
        current_indent = line[:line.find('if q == 1')]
        # Should be deeper than prev block (which was 'if q in [1, 2]')
        # Wait, I don't know prev block here.
        # But I know it should be 4 spaces deeper than 26 chars?
        # Let's assume 30 chars.
        new_lines.append(' ' * 30 + target_string_1 + '\n')
    elif target_string_2 in line:
        new_lines.append(' ' * 30 + target_string_2 + '\n')
    elif target_string_3 in line:
        # This one is tricky. It appears TWICE in the file?
        # Once at 1547 (inside if). Once at 1542 (original).
        # Original: gap_cx = ((b[0]+b[2])/2) + (dir_factor * w * 1.1)
        # Fix: gap_cx = ((b[0]+b[2])/2) + (df_fix * w * 1.1)
        # Note variable name change: dir_factor vs df_fix.
        # So target_string_3 is unique.
        new_lines.append(' ' * 30 + target_string_3 + '\n')
    else:
        new_lines.append(line)

with open(fn, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Applied Indentation Fix.")
