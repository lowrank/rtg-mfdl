import re
import os
import sys

def process_content(content):
    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    
    def is_list_item(l):
        return bool(re.match(r'^\s*([-*+]|\d+\.)\s+', l))

    def is_header(l):
        return bool(re.match(r'^#+\s+', l))

    def get_indent(l):
        m = re.match(r'^(\s*)', l)
        return m.group(1) if m else ""

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Rule 5: Headers (except the first)
        if is_header(line):
            has_content_before = False
            for prev in reversed(new_lines):
                if prev.strip():
                    has_content_before = True
                    break
            if has_content_before:
                if new_lines and new_lines[-1].strip() != "":
                    new_lines.append("\n")
        
        # Rule 6: Lists
        if is_list_item(line):
            if new_lines and new_lines[-1].strip() != "" and not is_list_item(new_lines[-1]):
                new_lines.append("\n")

        # Rule 1, 2, 3: Math Blocks
        if stripped.startswith('$$'):
            # Detect indentation requirement
            indent = ""
            # Look back for last non-empty line to see if it's a list item
            for j in range(len(new_lines) - 1, -1, -1):
                if new_lines[j].strip():
                    if is_list_item(new_lines[j]):
                        # Use list item indentation + 4 spaces
                        list_indent = get_indent(new_lines[j])
                        indent = list_indent + "    "
                    break
            
            # Rule 1: Isolation (before)
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("\n")
            
            # Opening delimiter
            new_lines.append(indent + "$$\n")
            
            # Content
            math_lines = []
            if stripped.endswith('$$') and len(stripped) > 2:
                content_inner = stripped[2:-2].strip()
                if content_inner:
                    math_lines.append(content_inner + "\n")
                i += 1
            else:
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('$$'):
                    math_lines.append(lines[i])
                    i += 1
                if i < len(lines) and lines[i].strip().startswith('$$'):
                    i += 1
            
            # Rule 2: Internal Newlines
            while math_lines and not math_lines[0].strip():
                math_lines.pop(0)
            while math_lines and not math_lines[-1].strip():
                math_lines.pop()
            
            for ml in math_lines:
                # No truncation: preserve internal content
                # But apply the block indentation to every line
                # If the line already has some indentation, we keep it relative?
                # Actually, user said "the ENTIRE block ... MUST be indented by 4 spaces"
                # If we lstrip and then add indent, it might lose internal relative indentation.
                # Let's try to be smarter: just ensure it starts with at least `indent`.
                l_stripped = ml.lstrip()
                new_lines.append(indent + l_stripped)
            
            # Closing delimiter
            new_lines.append(indent + "$$\n")
            
            # Rule 1: Isolation (after)
            if i < len(lines) and lines[i].strip() != "":
                new_lines.append("\n")
            
            continue

        new_lines.append(line)
        i += 1

    return "".join(new_lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_math.py <file1> <file2> ...")
        sys.exit(1)
        
    for filepath in sys.argv[1:]:
        if not os.path.isfile(filepath):
            continue
        print(f"Processing {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        new_content = process_content(original_content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

if __name__ == "__main__":
    main()
