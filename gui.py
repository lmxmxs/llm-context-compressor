#!/usr/bin/env python3
"""
llm-context-compressor GUI
Dark minimalist interface for density-based LLM context compression.

Run:
    python gui.py

Requires: PyQt6 (pip install PyQt6)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from context_compressor import (
    compress_qa_pairs,
    compress_sources,
    smart_truncate_context,
    estimate_compression_ratio,
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QComboBox, QSpinBox,
    QSplitter, QStatusBar, QFileDialog, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QKeySequence, QShortcut

# ── Palette ───────────────────────────────────────────────────────────────────

_BG0      = "#080808"
_BG1      = "#0f0f0f"
_BG2      = "#161616"
_BG3      = "#232323"
_FG       = "#c8c8c8"
_FG_DIM   = "#555555"
_GREEN    = "#39ff14"
_GREEN_DK = "#1a5c00"
_AMBER    = "#ffaa00"
_RED      = "#ff4040"
_MONO     = "Fira Code, Cascadia Code, Consolas, Liberation Mono, monospace"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {_BG0};
    color: {_FG};
    font-family: {_MONO};
    font-size: 12px;
}}
QTextEdit {{
    background-color: {_BG2};
    color: {_FG};
    border: 1px solid {_BG3};
    selection-background-color: {_GREEN_DK};
    font-family: {_MONO};
    font-size: 12px;
}}
QTextEdit:focus {{ border: 1px solid {_GREEN_DK}; }}
QLineEdit {{
    background-color: {_BG2};
    color: {_FG};
    border: 1px solid {_BG3};
    padding: 4px 8px;
    font-family: {_MONO};
}}
QLineEdit:focus {{ border: 1px solid {_GREEN_DK}; }}
QSpinBox {{
    background-color: {_BG2};
    color: {_FG};
    border: 1px solid {_BG3};
    padding: 2px 6px;
    font-family: {_MONO};
}}
QSpinBox::up-button, QSpinBox::down-button {{ background: {_BG3}; border: none; width: 14px; }}
QComboBox {{
    background-color: {_BG2};
    color: {_FG};
    border: 1px solid {_BG3};
    padding: 4px 8px;
    font-family: {_MONO};
}}
QComboBox::drop-down {{ border: none; background: {_BG3}; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {_BG2};
    color: {_FG};
    selection-background-color: {_BG3};
    border: 1px solid {_BG3};
}}
QPushButton {{
    background-color: {_BG2};
    color: {_FG};
    border: 1px solid {_BG3};
    padding: 5px 14px;
    font-family: {_MONO};
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton:hover {{ background-color: {_BG3}; border-color: {_GREEN_DK}; color: {_GREEN}; }}
QPushButton:pressed {{ background-color: {_GREEN_DK}; color: {_BG0}; }}
QPushButton#run_btn {{
    background-color: {_GREEN_DK};
    color: {_BG0};
    border: 1px solid {_GREEN};
    font-weight: bold;
    font-size: 12px;
    padding: 7px 22px;
    letter-spacing: 2px;
    min-width: 130px;
}}
QPushButton#run_btn:hover {{ background-color: {_GREEN}; }}
QPushButton#run_btn:disabled {{ background-color: {_BG3}; color: {_FG_DIM}; border-color: {_BG3}; }}
QStatusBar {{
    background-color: {_BG1};
    color: {_FG_DIM};
    border-top: 1px solid {_BG3};
    font-size: 11px;
    padding: 0 6px;
}}
QSplitter::handle {{ background-color: {_BG3}; width: 2px; height: 2px; }}
QLabel {{ color: {_FG}; }}
QCheckBox {{ color: {_FG}; spacing: 8px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {_BG3}; background: {_BG2}; }}
QCheckBox::indicator:checked {{ background: {_GREEN_DK}; border-color: {_GREEN}; }}
QScrollBar:vertical {{ background: {_BG1}; width: 5px; }}
QScrollBar::handle:vertical {{ background: {_BG3}; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {_GREEN_DK}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {_BG1}; height: 5px; }}
QScrollBar::handle:horizontal {{ background: {_BG3}; }}
"""

ASCII_BANNER = (
    "  ██╗     ██╗     ███╗   ███╗   ██████╗████████╗██╗  ██╗\n"
    "  ██║     ██║     ████╗ ████║  ██╔════╝╚══██╔══╝╚██╗██╔╝\n"
    "  ██║     ██║     ██╔████╔██║  ██║        ██║    ╚███╔╝ \n"
    "  ██║     ██║     ██║╚██╔╝██║  ██║        ██║    ██╔██╗ \n"
    "  ███████╗███████╗██║ ╚═╝ ██║  ╚██████╗   ██║   ██╔╝ ██╗\n"
    "  ╚══════╝╚══════╝╚═╝     ╚═╝   ╚═════╝   ╚═╝   ╚═╝  ╚═╝\n"
    "  ───────────────────────────────────────────────────────────\n"
    "  LLM  CONTEXT  COMPRESSOR  ·  density-first  ·  v0.1.0\n"
    "  zero dependencies  ·  pure Python stdlib  ·  LLMLingua-inspired"
)


class _Worker(QThread):
    done  = pyqtSignal(str, str, int, int)
    error = pyqtSignal(str)

    def __init__(self, mode: str, text: str, params: dict):
        super().__init__()
        self._mode   = mode
        self._text   = text
        self._params = params

    def run(self):
        try:
            text = self._text.strip()
            p    = self._params
            if self._mode == "Q&A Pairs":
                qa = json.loads(text)
                result = compress_qa_pairs(
                    qa, max_chars=p["max_chars"], topic=p["topic"],
                    question_key=p["q_key"], answer_key=p["a_key"],
                )
                original = "\n\n".join(
                    f"Q: {a.get(p['q_key'], '')}\nA: {a.get(p['a_key'], '')}"
                    for a in qa
                )
            elif self._mode == "Smart Truncate":
                original = text
                result   = smart_truncate_context(text, max_chars=p["max_chars"], topic=p["topic"])
            else:
                sources  = json.loads(text)
                original = json.dumps(sources, indent=2)
                result   = compress_sources(
                    sources, max_items=p["max_items"],
                    include_content=p["include_content"],
                    max_content_chars=p["max_content_chars"],
                )
            ratio = estimate_compression_ratio(original, result)
            pct   = ratio * 100
            stats = (
                f"{len(original):,} → {len(result):,} chars  ·  "
                f"{pct:.1f}% of original  ·  {100-pct:.0f}% compressed"
            )
            self.done.emit(result, stats, len(original), len(result))
        except json.JSONDecodeError as e:
            self.error.emit(f"JSON parse error: {e}")
        except Exception as e:
            self.error.emit(f"Error: {e}")


class CompressorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("llm-context-compressor")
        self.resize(1140, 760)
        self._worker: _Worker | None = None
        self._build_ui()
        self._bind_shortcuts()
        QTimer.singleShot(800, lambda: self._status.showMessage(
            "Ctrl+Enter compress  ·  Ctrl+O open file  ·  Ctrl+S save output  ·  Ctrl+Shift+C copy"
        ))

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self._make_banner())
        vbox.addWidget(self._make_toolbar())
        vbox.addWidget(self._make_splitter(), 1)
        self._status = QStatusBar()
        self._status.setFixedHeight(22)
        self.setStatusBar(self._status)

    def _make_banner(self) -> QLabel:
        lbl = QLabel(ASCII_BANNER)
        lbl.setFont(QFont("Fira Code, Cascadia Code, Consolas", 9))
        lbl.setStyleSheet(f"color: {_GREEN}; background: {_BG0}; padding: 14px 18px 8px 18px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        return lbl

    def _make_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(f"background: {_BG1}; border-bottom: 1px solid {_BG3};")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(10)

        def _lbl(t):
            l = QLabel(t)
            l.setStyleSheet(f"color: {_FG_DIM}; font-size: 10px; letter-spacing: 2px;")
            return l

        row.addWidget(_lbl("MODE"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Q&A Pairs", "Smart Truncate", "Sources"])
        self._mode_combo.setFixedWidth(148)
        self._mode_combo.currentTextChanged.connect(self._on_mode_change)
        row.addWidget(self._mode_combo)
        row.addSpacing(18)

        row.addWidget(_lbl("TOPIC"))
        self._topic = QLineEdit()
        self._topic.setPlaceholderText("quantum computing, RAG, neural nets…")
        self._topic.setFixedWidth(280)
        row.addWidget(self._topic)
        row.addSpacing(18)

        self._limit_lbl = _lbl("MAX CHARS")
        row.addWidget(self._limit_lbl)
        self._max_spin = QSpinBox()
        self._max_spin.setRange(100, 500_000)
        self._max_spin.setValue(6_000)
        self._max_spin.setSingleStep(500)
        self._max_spin.setFixedWidth(88)
        row.addWidget(self._max_spin)
        row.addStretch()

        self._run_btn = QPushButton("▶  COMPRESS")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._run_btn)
        return bar

    def _make_splitter(self) -> QSplitter:
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(2)
        sp.addWidget(self._make_input_panel())
        sp.addWidget(self._make_output_panel())
        sp.setSizes([560, 540])
        return sp

    def _make_input_panel(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {_BG0};")
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 10, 6, 10)
        v.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        il = QLabel("INPUT")
        il.setStyleSheet(f"color: {_GREEN_DK}; letter-spacing: 3px; font-size: 10px; font-weight: bold;")
        hdr.addWidget(il)
        hdr.addStretch()

        ob = QPushButton("OPEN FILE")
        ob.setFixedWidth(88)
        ob.clicked.connect(self._open_file)
        hdr.addWidget(ob)

        cb = QPushButton("CLEAR")
        cb.setFixedWidth(62)
        cb.clicked.connect(lambda: (self._input.clear(), self._chars_lbl.setText("")))
        hdr.addWidget(cb)
        v.addLayout(hdr)

        self._input = QTextEdit()
        self._input.setPlaceholderText('[\n  {"question": "What is RAG?", "answer": "Retrieval-Augmented…"},\n  …\n]')
        self._input.textChanged.connect(self._on_input_changed)
        v.addWidget(self._input)

        self._chars_lbl = QLabel("")
        self._chars_lbl.setStyleSheet(f"color: {_FG_DIM}; font-size: 10px;")
        v.addWidget(self._chars_lbl)

        self._qa_opts = QWidget()
        qr = QHBoxLayout(self._qa_opts)
        qr.setContentsMargins(0, 0, 0, 0)
        qr.setSpacing(8)
        qr.addWidget(self._dim_lbl("Q KEY"))
        self._q_key = QLineEdit("question")
        self._q_key.setFixedWidth(96)
        qr.addWidget(self._q_key)
        qr.addWidget(self._dim_lbl("A KEY"))
        self._a_key = QLineEdit("answer")
        self._a_key.setFixedWidth(96)
        qr.addWidget(self._a_key)
        qr.addStretch()
        v.addWidget(self._qa_opts)

        self._src_opts = QWidget()
        self._src_opts.setVisible(False)
        sr = QHBoxLayout(self._src_opts)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(8)
        self._incl_content = QCheckBox("include content")
        sr.addWidget(self._incl_content)
        sr.addWidget(self._dim_lbl("MAX CONTENT CHARS"))
        self._max_content = QSpinBox()
        self._max_content.setRange(50, 2000)
        self._max_content.setValue(300)
        self._max_content.setFixedWidth(76)
        sr.addWidget(self._max_content)
        sr.addStretch()
        v.addWidget(self._src_opts)
        return w

    def _make_output_panel(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {_BG0};")
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 10, 12, 10)
        v.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        ol = QLabel("OUTPUT")
        ol.setStyleSheet(f"color: {_GREEN_DK}; letter-spacing: 3px; font-size: 10px; font-weight: bold;")
        hdr.addWidget(ol)
        hdr.addStretch()

        cpb = QPushButton("COPY")
        cpb.setFixedWidth(62)
        cpb.clicked.connect(self._copy_output)
        hdr.addWidget(cpb)

        svb = QPushButton("SAVE")
        svb.setFixedWidth(62)
        svb.clicked.connect(self._save_output)
        hdr.addWidget(svb)
        v.addLayout(hdr)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("Compressed output appears here…")
        v.addWidget(self._output)

        self._ratio_lbl = QLabel("ratio: —")
        self._ratio_lbl.setStyleSheet(f"color: {_FG_DIM}; font-size: 10px; letter-spacing: 1px;")
        v.addWidget(self._ratio_lbl)
        return w

    def _dim_lbl(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color: {_FG_DIM}; font-size: 10px; letter-spacing: 1px;")
        return l

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"),   self).activated.connect(self._run)
        QShortcut(QKeySequence("Ctrl+O"),        self).activated.connect(self._open_file)
        QShortcut(QKeySequence("Ctrl+S"),        self).activated.connect(self._save_output)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self).activated.connect(self._copy_output)

    def _on_mode_change(self, mode: str):
        self._qa_opts.setVisible(mode == "Q&A Pairs")
        self._src_opts.setVisible(mode == "Sources")
        if mode == "Sources":
            self._limit_lbl.setText("MAX ITEMS")
            self._max_spin.setRange(1, 500)
            self._max_spin.setValue(20)
        else:
            self._limit_lbl.setText("MAX CHARS")
            self._max_spin.setRange(100, 500_000)
            self._max_spin.setValue(6_000)
        placeholders = {
            "Q&A Pairs":     '[\n  {"question": "What is RAG?", "answer": "Retrieval-Augmented…"},\n  …\n]',
            "Smart Truncate": "Paste your long document text here…\n\n(50 000 chars → compressed to MAX CHARS budget,\nkeeping front + dense middle + back)",
            "Sources":       '[\n  {"title": "Attention Is All You Need", "url": "https://…"},\n  …\n]',
        }
        self._input.setPlaceholderText(placeholders.get(mode, ""))

    def _on_input_changed(self):
        n = len(self._input.toPlainText())
        self._chars_lbl.setText(f"{n:,} chars" if n else "")

    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        text = self._input.toPlainText().strip()
        if not text:
            self._status.showMessage("⚠  no input text")
            return
        mode   = self._mode_combo.currentText()
        params = {
            "max_chars":         self._max_spin.value(),
            "topic":             self._topic.text().strip(),
            "q_key":             self._q_key.text().strip() or "question",
            "a_key":             self._a_key.text().strip() or "answer",
            "max_items":         self._max_spin.value(),
            "include_content":   self._incl_content.isChecked(),
            "max_content_chars": self._max_content.value(),
        }
        self._run_btn.setEnabled(False)
        self._run_btn.setText("▶  running…")
        self._output.clear()
        self._ratio_lbl.setText("ratio: …")
        self._status.showMessage("compressing…")
        self._worker = _Worker(mode, text, params)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: str, stats: str, orig_len: int, cmp_len: int):
        self._output.setStyleSheet("")
        self._output.setPlainText(result)
        self._status.showMessage(stats)
        ratio = cmp_len / max(orig_len, 1)
        pct   = ratio * 100
        color = _GREEN if pct < 30 else (_AMBER if pct < 55 else _RED)
        grade = "excellent" if pct < 30 else ("good" if pct < 55 else "low")
        self._ratio_lbl.setText(
            f"ratio: {ratio:.3f}  ·  {100-pct:.0f}% reduction  ·  {cmp_len:,} chars out  [{grade}]"
        )
        self._ratio_lbl.setStyleSheet(f"color: {color}; font-size: 10px; letter-spacing: 1px;")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  COMPRESS")

    def _on_error(self, msg: str):
        self._output.setStyleSheet(f"color: {_RED};")
        self._output.setPlainText(f"ERROR\n\n{msg}")
        self._status.showMessage(f"✗  {msg}")
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  COMPRESS")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open file", "", "JSON / Text (*.json *.txt *.md);;All files (*)"
        )
        if path:
            self._input.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
            self._status.showMessage(f"loaded: {Path(path).name}")

    def _copy_output(self):
        text = self._output.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self._status.showMessage("copied to clipboard")

    def _save_output(self):
        text = self._output.toPlainText()
        if not text:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output", "compressed.txt", "Text (*.txt);;All files (*)"
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            self._status.showMessage(f"saved: {Path(path).name}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(_BG0))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(_FG))
    pal.setColor(QPalette.ColorRole.Base,            QColor(_BG2))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(_BG1))
    pal.setColor(QPalette.ColorRole.Text,            QColor(_FG))
    pal.setColor(QPalette.ColorRole.Button,          QColor(_BG2))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(_FG))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(_GREEN_DK))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(_BG0))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(_FG_DIM))
    app.setPalette(pal)
    win = CompressorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
