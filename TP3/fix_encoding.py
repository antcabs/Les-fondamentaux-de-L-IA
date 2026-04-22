# Script to replace special Unicode characters with ASCII equivalents
with open('tp3_complet.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    ('\u2192', '->'),
    ('\u2190', '<-'),
    ('\u2550', '='),
    ('\u2554', '+'),
    ('\u2557', '+'),
    ('\u2551', '|'),
    ('\u255a', '+'),
    ('\u255d', '+'),
    ('\u2014', '--'),
    ('\u2013', '-'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open('tp3_complet.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done - special characters replaced')
