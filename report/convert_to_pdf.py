#!/usr/bin/env python3
"""Convert SC5005_report.md to PDF with proper formatting."""

import markdown
from weasyprint import HTML
from pathlib import Path
import re

REPORT_DIR = Path(__file__).parent
MD_FILE = REPORT_DIR / "SC5005_report.md"
PDF_FILE = REPORT_DIR / "SC5005_report.pdf"

CSS = """
@page {
    size: A4;
    margin: 1.8cm 1.8cm 1.8cm 1.8cm;
}
body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 10pt;
    line-height: 1.3;
    color: #000;
}
h1 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 0;
    margin-bottom: 4pt;
    text-align: center;
}
h2 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 8pt;
    margin-bottom: 3pt;
}
h3 {
    font-size: 10pt;
    font-weight: bold;
    margin-top: 6pt;
    margin-bottom: 2pt;
}
p {
    margin-top: 2pt;
    margin-bottom: 2pt;
    text-align: justify;
}
table {
    border-collapse: collapse;
    width: auto;
    margin: 4pt auto;
    font-size: 9pt;
}
th, td {
    border: 1px solid #333;
    padding: 2pt 5pt;
    text-align: left;
}
th {
    background-color: #e8e8e8;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
img {
    max-width: 42%;
    height: auto;
    display: block;
    margin: 3pt auto;
}
em {
    font-style: italic;
}
strong {
    font-weight: bold;
}
blockquote {
    margin: 3pt 10pt;
    padding: 2pt 6pt;
    border-left: 2px solid #999;
    font-style: italic;
    font-size: 9pt;
}
ul, ol {
    margin-top: 1pt;
    margin-bottom: 1pt;
    padding-left: 16pt;
}
li {
    margin-bottom: 1pt;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 4pt 0;
}
/* Figure captions */
p > em:only-child {
    display: block;
    text-align: center;
    font-size: 8pt;
    margin-top: 1pt;
}
/* Page break before references */
.page-break {
    page-break-before: always;
}
"""


def convert():
    md_text = MD_FILE.read_text()

    # Fix image paths to be absolute
    md_text = md_text.replace("](../MAD.png)", f"]({REPORT_DIR.parent / 'MAD.png'})")
    md_text = md_text.replace("](../out/", f"]({REPORT_DIR.parent / 'out'}/")

    # Insert page break before References and Appendices
    md_text = md_text.replace("## References", '<div class="page-break"></div>\n\n## References')
    md_text = md_text.replace("## Appendix A", '<div class="page-break"></div>\n\n## Appendix A')

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "md_in_html"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html, base_url=str(REPORT_DIR)).write_pdf(str(PDF_FILE))
    print(f"PDF written to {PDF_FILE}")

    # Count pages
    try:
        import fitz
        doc = fitz.open(str(PDF_FILE))
        print(f"Total pages: {len(doc)}")
        # Find where References start by searching text
        for i, page in enumerate(doc):
            text = page.get_text()
            if "References" in text and "Bai, Y." in text:
                print(f"References start on page {i+1}")
                print(f"Main body: {i} pages")
                break
        doc.close()
    except ImportError:
        pass


if __name__ == "__main__":
    convert()
