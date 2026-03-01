"""
Generate engineering_report.pdf from ENGINEERING_REPORT.md using fpdf2.
Tables use smart proportional column widths based on content length.
Multi-line cells are drawn with fill rectangles for proper alignment.
"""

from pathlib import Path
from fpdf import FPDF
import re

# Page margins
LEFT_MARGIN = 15
RIGHT_MARGIN = 15
PAGE_WIDTH = 210
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN  # 180mm


def sanitize(text: str) -> str:
    """Replace non-latin-1 characters with ASCII equivalents."""
    table = {
        "\u2014": "--", "\u2013": "-",
        "\u2018": "'",  "\u2019": "'",
        "\u201c": '"',  "\u201d": '"',
        "\u2022": "-",  "\u2026": "...",
        "\u00d7": "x",  "\u2192": "->",
        "\u2190": "<-", "\u00b0": " deg",
        "\u00a0": " ",  "\u2265": ">=",
        "\u2264": "<=", "\u00e9": "e",
        "\u00e0": "a",  "\u2713": "[OK]",
        "\u274c": "[X]","\u2714": "[OK]",
        "\u2705": "[OK]",
    }
    for ch, rep in table.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def strip_md(text: str) -> str:
    """Strip bold/italic/code markdown markers."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    return text.strip()


def compute_col_widths(parsed: list, total_width: float) -> list:
    """
    Compute proportional column widths based on max cell content length.
    First column gets 1.5x weight since it usually holds labels/names.
    """
    if not parsed:
        return []
    col_count = max(len(r) for r in parsed)
    # Find max length of text in each column
    max_lens = [0] * col_count
    for row in parsed:
        for ci, cell in enumerate(row):
            if ci < col_count:
                clean = sanitize(strip_md(cell))
                max_lens[ci] = max(max_lens[ci], len(clean))

    # Give first column 1.5x weight
    weights = [max_lens[0] * 1.5] + [max_lens[i] for i in range(1, col_count)]
    total_weight = sum(weights) or col_count
    widths = [(w / total_weight) * total_width for w in weights]

    # Enforce minimum column width of 18mm
    min_w = 18.0
    for i in range(len(widths)):
        if widths[i] < min_w:
            widths[i] = min_w

    # Re-scale to fit total_width
    scale = total_width / sum(widths)
    widths = [w * scale for w in widths]
    return widths


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.set_xy(LEFT_MARGIN, 8)
        self.cell(CONTENT_WIDTH, 6, "Prime Lands RAG Platform - Engineering Report", align="R")
        self.set_draw_color(180, 180, 180)
        self.line(LEFT_MARGIN, 15, PAGE_WIDTH - RIGHT_MARGIN, 15)
        self.set_y(18)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def h1(self, text: str):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(25, 70, 150)
        self.ln(4)
        self.set_x(LEFT_MARGIN)
        self.multi_cell(CONTENT_WIDTH, 10, sanitize(text))
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def h2(self, text: str):
        self.ln(5)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(25, 70, 150)
        self.set_x(LEFT_MARGIN)
        self.multi_cell(CONTENT_WIDTH, 8, sanitize(text))
        self.set_text_color(0, 0, 0)
        self.set_draw_color(160, 190, 230)
        y = self.get_y()
        self.line(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y)
        self.ln(3)

    def h3(self, text: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.set_x(LEFT_MARGIN)
        self.multi_cell(CONTENT_WIDTH, 7, sanitize(text))
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def paragraph(self, text: str):
        self.set_font("Helvetica", size=10)
        self.set_x(LEFT_MARGIN)
        self.multi_cell(CONTENT_WIDTH, 5.5, sanitize(strip_md(text)))
        self.ln(1)

    def bullet(self, text: str):
        self.set_font("Helvetica", size=10)
        self.set_x(LEFT_MARGIN + 3)
        self.cell(5, 5.5, "-")
        self.set_x(LEFT_MARGIN + 8)
        self.multi_cell(CONTENT_WIDTH - 8, 5.5, sanitize(strip_md(text)))

    def blockquote(self, text: str):
        self.set_font("Helvetica", "I", 9)
        self.set_fill_color(255, 248, 210)
        self.set_x(LEFT_MARGIN + 3)
        self.multi_cell(CONTENT_WIDTH - 3, 5, sanitize(strip_md(text)), fill=True)
        self.ln(1)

    def hrule(self):
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        y = self.get_y()
        self.line(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y)
        self.ln(4)

    def _render_header_row(self, header: list, col_widths: list):
        """Render the table header row with blue fill."""
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(210, 225, 250)
        self.set_text_color(20, 20, 80)
        x = LEFT_MARGIN
        y = self.get_y()
        ROW_H = 7.0

        # Compute max lines in header
        max_lines = 1
        for ci, cell in enumerate(header):
            clean = sanitize(strip_md(cell))
            cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]
            chars_per_line = max(1, int(cw / 2.1))
            lines = max(1, (len(clean) + chars_per_line - 1) // chars_per_line)
            max_lines = max(max_lines, lines)

        row_h = ROW_H * max_lines

        for ci, cell in enumerate(header):
            clean = sanitize(strip_md(cell))
            cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]
            # Draw fill rectangle
            self.set_fill_color(210, 225, 250)
            self.rect(x, y, cw, row_h, style="F")
            # Draw border
            self.set_draw_color(150, 170, 220)
            self.rect(x, y, cw, row_h)
            # Write text
            if max_lines == 1:
                self.set_xy(x + 1, y + (row_h - ROW_H) / 2 + 0.5)
                self.cell(cw - 2, ROW_H, clean)
            else:
                self.set_xy(x + 1, y + 1)
                self.multi_cell(cw - 2, ROW_H - 0.5, clean, border=0, fill=False)
            x += cw

        self.set_text_color(0, 0, 0)
        self.set_y(y + row_h)

    def render_table(self, rows: list):
        """
        Render a markdown table with smart proportional column widths.
        Each column's width is proportional to its max content length.
        Multi-line cells use rect+multi_cell for correct fill and alignment.
        """
        # Parse rows — skip separator lines (|---|---|)
        parsed = []
        for row in rows:
            if re.match(r'^\|[\s\-:|]+\|?\s*$', row.strip()):
                continue
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            parsed.append(cells)

        if not parsed:
            return

        col_count = max(len(r) for r in parsed)
        # Pad all rows to col_count
        for row in parsed:
            while len(row) < col_count:
                row.append("")

        col_widths = compute_col_widths(parsed, CONTENT_WIDTH)
        ROW_H = 6.5

        # Estimate total table height to decide if we need a new page first
        def estimate_row_height(row, cws):
            ml = 1
            for ci, cell in enumerate(row):
                clean = sanitize(strip_md(cell))
                cw = cws[ci] if ci < len(cws) else cws[-1]
                cpp = max(1, int(cw / 2.1))
                lines = max(1, (len(clean) + cpp - 1) // cpp)
                ml = max(ml, lines)
            return ROW_H * ml

        total_h = sum(estimate_row_height(r, col_widths) for r in parsed) + 6
        available = 270 - self.get_y()

        # If the whole table won't fit and we're past 1/3 of the page, start fresh
        if total_h > available and self.get_y() > 100:
            self.add_page()

        self.ln(2)

        for ri, row in enumerate(parsed):
            is_header = (ri == 0)

            # Compute row height
            max_lines = 1
            for ci, cell in enumerate(row):
                clean = sanitize(strip_md(cell))
                cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]
                chars_per_line = max(1, int(cw / 2.1))
                lines_needed = max(1, (len(clean) + chars_per_line - 1) // chars_per_line)
                max_lines = max(max_lines, lines_needed)

            row_h = ROW_H * max_lines

            # Page break within table
            if self.get_y() + row_h > 270:
                self.add_page()
                self.ln(2)
                # Re-render header on new page
                self._render_header_row(parsed[0], col_widths)
                if ri == 0:
                    continue  # don't double-render header

            if is_header:
                self._render_header_row(row, col_widths)
                continue

            # Data row
            even = (ri % 2 == 1)
            fill_rgb = (245, 249, 255) if even else (255, 255, 255)
            self.set_fill_color(*fill_rgb)
            self.set_font("Helvetica", size=9)
            self.set_text_color(30, 30, 30)
            self.set_draw_color(190, 200, 220)

            y_row = self.get_y()
            x = LEFT_MARGIN
            for ci, cell in enumerate(row):
                clean = sanitize(strip_md(cell))
                cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]

                # Fill background
                self.set_fill_color(*fill_rgb)
                self.rect(x, y_row, cw, row_h, style="F")
                # Border
                self.set_draw_color(200, 210, 230)
                self.rect(x, y_row, cw, row_h)

                # Text
                if max_lines == 1:
                    self.set_xy(x + 1, y_row + (row_h - ROW_H) / 2 + 0.5)
                    self.cell(cw - 2, ROW_H, clean, border=0, fill=False)
                else:
                    self.set_xy(x + 1, y_row + 1)
                    self.multi_cell(cw - 2, ROW_H - 0.5, clean, border=0, fill=False)
                x += cw

            self.set_text_color(0, 0, 0)
            self.set_y(y_row + row_h)

        self.ln(4)


def parse_and_render(pdf: ReportPDF, md_path: Path):
    raw = md_path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    table_rows = []
    in_code = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Code block ──────────────────────────────────────────────
        if line.strip().startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue

        # ── Table ───────────────────────────────────────────────────
        if line.strip().startswith("|"):
            table_rows.append(line)
            i += 1
            continue
        else:
            if table_rows:
                pdf.render_table(table_rows)
                table_rows = []

        # ── Headings ────────────────────────────────────────────────
        stripped = line.rstrip()
        if re.match(r'^# [^#]', stripped):
            pdf.h1(stripped[2:].strip())
        elif re.match(r'^## ', stripped):
            pdf.h2(stripped[3:].strip())
        elif re.match(r'^### ', stripped):
            pdf.h3(stripped[4:].strip())
        elif re.match(r'^#### ', stripped):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(LEFT_MARGIN)
            pdf.multi_cell(CONTENT_WIDTH, 6, sanitize(strip_md(stripped[5:])))
            pdf.ln(1)

        # ── Blockquote ──────────────────────────────────────────────
        elif stripped.startswith("> "):
            pdf.blockquote(stripped[2:])

        # ── Bullet ──────────────────────────────────────────────────
        elif re.match(r'^[-*] ', stripped):
            pdf.bullet(stripped[2:])

        # ── Horizontal rule ─────────────────────────────────────────
        elif re.match(r'^---+$', stripped.strip()):
            pdf.hrule()

        # ── Blank line ──────────────────────────────────────────────
        elif stripped.strip() == "":
            pdf.ln(2)

        # ── Normal paragraph ────────────────────────────────────────
        else:
            pdf.paragraph(stripped)

        i += 1

    # Flush remaining table
    if table_rows:
        pdf.render_table(table_rows)


def main():
    root = Path.cwd()
    md_path = root / "ENGINEERING_REPORT.md"
    out_path = root / "report" / "engineering_report.pdf"
    out_path.parent.mkdir(exist_ok=True)

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(LEFT_MARGIN, 18, RIGHT_MARGIN)
    pdf.add_page()

    parse_and_render(pdf, md_path)

    pdf.output(str(out_path))
    size_kb = out_path.stat().st_size / 1024
    print(f"PDF generated: {out_path}")
    print(f"Pages: {pdf.page}  |  Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
