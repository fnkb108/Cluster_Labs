"""
Утилиты форматирования для отображения результатов бенчмарков в виде аккуратных ASCII/Markdown таблиц.
"""

def format_table(headers, rows, alignments=None):
    """
    Формирует ASCII-таблицу.
    headers: список строк
    rows: список списков с данными
    alignments: список символов выравнивания ('<', '>' или '^')
    """
    if not rows:
        return "No data"

    str_rows = [[str(cell) for cell in row] for row in rows]
    num_cols = len(headers)

    if not alignments:
        alignments = ['<'] * num_cols

    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for i in range(num_cols):
            if i < len(row):
                col_widths[i] = max(col_widths[i], len(row[i]))

    # Разделительная линия
    def make_line(left, mid, right, fill='-'):
        return left + mid.join(fill * (w + 2) for w in col_widths) + right

    top_border = make_line('+', '+', '+')
    sep_border = make_line('+', '+', '+')
    bot_border = make_line('+', '+', '+')

    # Форматирование отдельной строки
    def render_row(cells):
        out = []
        for i, cell in enumerate(cells):
            align = alignments[i] if i < len(alignments) else '<'
            w = col_widths[i]
            if align == '>':
                out.append(f" {cell:>{w}} ")
            elif align == '^':
                out.append(f" {cell:^{w}} ")
            else:
                out.append(f" {cell:<{w}} ")
        return '|' + '|'.join(out) + '|'

    lines = [
        top_border,
        render_row(headers),
        sep_border,
    ]
    for row in str_rows:
        lines.append(render_row(row))
    lines.append(bot_border)

    return '\n'.join(lines)


def format_bytes(num_bytes):
    """Преобразует количество байт в читаемую строку (B, KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    else:
        return f"{num_bytes / (1024**3):.2f} GB"