#!/usr/bin/env python3
import re
import sys

# Match a comma used as decimal separator: digit on left, digit on right,
# but not when right digit is followed by 'x' (hex prefix like 0x...)
pattern = re.compile(r'(\d),(\d)(?!x)')

for line in sys.stdin:
    line = pattern.sub(r'\1.\2', line)
    sys.stdout.write(line)
