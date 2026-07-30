"""Shared PDF scaffolding for the project's reports."""
from __future__ import annotations

import os
from typing import Sequence

from fpdf import FPDF

INK, INK2, MUTED = (11, 11, 11), (82, 81, 78), (137, 135, 129)
BLUE, RULE = (42, 120, 214), (225, 224, 217)


class PdfReport(FPDF):
    """An A4 report with headings, body copy, tables and captioned figures."""

    def __init__(self, running_head: str = "", **kwargs):
        super().__init__(format="A4", **kwargs)
        self.running_head = running_head
        self.set_margins(18, 16, 18)
        self.set_auto_page_break(True, margin=16)

    @property
    def content_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def header(self) -> None:
        if self.page_no() > 1 and self.running_head:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*MUTED)
            self.cell(0, 5, self.running_head, align="R",
                      new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, str(self.page_no()), align="C")

    def h1(self, text: str) -> None:
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*INK)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text: str) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", 12.5)
        self.set_text_color(*INK)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2.5)

    def body(self, text: str, size: float = 10) -> None:
        self.set_font("Helvetica", "", size)
        self.set_text_color(*INK2)
        self.multi_cell(0, 5.2 if size > 10 else 5, text,
                        new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def mono(self, text: str) -> None:
        self.set_fill_color(246, 246, 244)
        self.set_font("Courier", "", 8.6)
        self.set_text_color(*INK)
        self.multi_cell(0, 4.6, text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def caption(self, text: str) -> None:
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.2, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def figure(self, path: str, caption: str, width: float | None = None) -> None:
        width = width or self.content_width
        self.image(path, x=(self.w - width) / 2, w=width)
        self.ln(1.5)
        self.caption(caption)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
              widths: Sequence[float], size: float = 9.5) -> None:
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*INK)
        for text, width in zip(headers, widths):
            self.cell(width, 6, text, border="B")
        self.ln()
        self.set_font("Helvetica", "", size)
        self.set_text_color(*INK2)
        for row in rows:
            for text, width in zip(row, widths):
                self.cell(width, 6, str(text), border="B")
            self.ln()
        self.ln(2)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.output(path)
        print(f"wrote {path}")
        return path
