"""XML plumbing shared by every reader and writer in the pipeline."""
from __future__ import annotations

import gzip
import os
import xml.etree.ElementTree as ET
from typing import IO, Iterator
from xml.sax.saxutils import escape, quoteattr

XML_HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n'


def open_maybe_gzip(path: str, mode: str = "rb") -> IO:
    """SUMO reads and writes .gz transparently; so do we."""
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def parse_root(path: str) -> ET.Element:
    with open_maybe_gzip(path) as handle:
        return ET.parse(handle).getroot()


def iter_elements(path: str, tags: tuple[str, ...]) -> Iterator[ET.Element]:
    """Stream elements of interest, clearing as we go: these files reach GBs."""
    with open_maybe_gzip(path) as handle:
        for _, element in ET.iterparse(handle, events=("start",)):
            if element.tag in tags:
                yield element
            element.clear()


def attrs(**values) -> str:
    return " ".join(f"{k}={quoteattr(str(v))}" for k, v in values.items()
                    if v is not None)


def ensure_parent(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def write_tree(root: ET.Element, path: str) -> str:
    ensure_parent(path)
    ET.indent(tree := ET.ElementTree(root))
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return path


__all__ = ["XML_HEADER", "attrs", "ensure_parent", "escape", "iter_elements",
           "open_maybe_gzip", "parse_root", "quoteattr", "write_tree"]
