#!/usr/bin/env python3

import io
from typing import Sequence
from typing import Any
import sys

def write_latex_table(content_rows: Sequence[Sequence[Any]], header_rows: Sequence[str], label: str = "tableX") -> str:
    buffer = io.StringIO()
    def _print(*args, **kwargs):
        print(*args, **kwargs, file=buffer)
    def _printrow(cells):
        _print(" & ".join(cells) + r"    \\")
    _print(r"""
\begin{table}[b]
\centering
%\resizebox{.95\columnwidth}{!}{
\begin{tabular}""", end="")
    num_cols = len(header_rows)
    _print("{|" + "|".join(["l"] * num_cols) + "|}")
    _print(r"    \firsthline""")
    _print("    ", end="")
    _printrow(header_rows)
    _print(r"    \hline")
    for row in content_rows:
        _print("    ", end="")
        _printrow(row)
    _print(r"    \lasthline")
    _print(r"\end{tabular}")
    _print("%}")
    _print(r"""\caption{INSERT CAPTION TEXT HERE}
\label{table1}
\end{table}
""")
    s = buffer.getvalue()
    return s



def arch_exp() -> int:
    header_row = [
        "Arch",
        "Val Acc"
    ]
    content_rows = [
        ["2-2-2", "86.56\\%"],
        ["2-4-2", "84.74\\%"],
        ["2-5-2", "84.98\\%"],
        ["2-5-3", "85.32\\%"],
        ["3-5-3", "87.00\\%"],
    ]
    print(write_latex_table(content_rows, header_row))
    return 0


def arch_lrs_exp() -> int:
    header_row = [
        "Arch/LRS",
        "Val Acc \\%"
    ]
    content_rows = [
    ["2-2-2 Step", "86.7"],
    ["3-5-3 Step", "87.3"],
    ["2-2-2 CosA", "92.1"],
    ["3-5-3 CosA", "92.0"],
    ["2-2-2 Plat", "92.2"],
    ["3-5-3 Plat", "91.7"],
    ]
    print(write_latex_table(content_rows, header_row))
    return 0

def main() -> int:

    #           0.9178  checkpoint/ckpt-a9f40a4b.pth  3-5-3;h=Hyperparametry(default);k=3;lr=0.1;opt=sgd;sch=plateau:factor=0.75;patience=5;threshold=0.05;threshold_mode=abs
    #           0.9142  checkpoint/ckpt-79af52d3.pth  3-5-3;h=Hyperparametry(k=3,ildr=0.2,dropout=0.0);k=3;lr=0.1;opt=sgd;sch=plateau:factor=0.75;patience=5;threshold=0.05;threshold_mode=abs
    #           0.912   checkpoint/ckpt-36ea8246.pth  3-5-3;h=Hyperparametry(k=3,ildr=0.2,dropout=0.2);k=3;lr=0.1;opt=sgd;sch=plateau:factor=0.75;patience=5;threshold=0.05;threshold_mode=abs
    #           0.8968  checkpoint/ckpt-6723d73e.pth  3-5-3;h=Hyperparametry(k=3,ildr=0.2,dropout=0.5);k=3;lr=0.1;opt=sgd;sch=plateau:factor=0.75;patience=5;threshold=0.05;threshold_mode=abs
    header_row = [
        "Input \\%", "Hidden \\%", "Val Acc \\%"
    ]
    content_rows = [
        ["0", "20", "91.8"],
        ["20", "0", "91.4"],
        ["20", "20", "91.2"],
        ["20", "50", "89.7"],
    ]
    print(write_latex_table(content_rows, header_row))
    return 0


if __name__ == '__main__':
    sys.exit(main())
