"""
Slunder Studio v0.0.2 — Lyrics Editor
Rich text editor with syntax highlighting for structure tags ([Verse], [Chorus], etc.),
right-click section regeneration, streaming token display, and export tools.
"""
import re
from typing import Optional

from PySide6.QtWidgets import (
    QPlainTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMenu, QApplication,
)
from PySide6.QtCore import Qt, Signal, QRegularExpression
from PySide6.QtGui import (
    QSyntaxHighlighter, QTextCharFormat, QColor, QFont,
    QTextCursor, QAction,
)

from ui.theme import Palette


# ── Syntax Highlighter ─────────────────────────────────────────────────────────

class LyricsHighlighter(QSyntaxHighlighter):
    """Highlights structure tags, ad-libs, and special markers in lyrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules = []

        # Structure tags: [Verse 1], [Chorus], [Bridge], etc.
        tag_fmt = QTextCharFormat()
        tag_fmt.setForeground(QColor(Palette.BLUE))
        tag_fmt.setFontWeight(QFont.Weight.Bold)
        tag_fmt.setFontPointSize(12)
        self._rules.append((
            QRegularExpression(r"\[(?:Verse|Chorus|Pre-Chorus|Post-Chorus|Bridge|Outro|Intro|Hook|Breakdown|Instrumental|Interlude|Spoken|Ad-lib|Refrain|Drop)(?:\s*\d*)?\]"),
            tag_fmt
        ))

        # Parenthetical directions: (whispered), (x2), (repeat)
        paren_fmt = QTextCharFormat()
        paren_fmt.setForeground(QColor(Palette.OVERLAY0))
        paren_fmt.setFontItalic(True)
        self._rules.append((
            QRegularExpression(r"\(.*?\)"),
            paren_fmt
        ))

        # Ad-libs in all caps at end of lines
        adlib_fmt = QTextCharFormat()
        adlib_fmt.setForeground(QColor(Palette.MAUVE))
        adlib_fmt.setFontItalic(True)
        self._rules.append((
            QRegularExpression(r"(?:yeah|hey|oh|uh|woo|skrrt|ayy|hmm|ooh|ah)(?:\s*!*\s*$)", QRegularExpression.PatternOption.CaseInsensitiveOption),
            adlib_fmt
        ))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            match_iter = pattern.globalMatch(text)
            while match_iter.hasNext():
                match = match_iter.next()
                start = match.capturedStart()
                length = match.capturedLength()
                self.setFormat(start, length, fmt)


# ── Lyrics Editor Widget ──────────────────────────────────────────────────────

class LyricsEditor(QWidget):
    """
    Rich lyrics editor with syntax highlighting, section detection,
    right-click regeneration, and streaming token display.

    Signals:
        text_changed(str)           - emitted when lyrics text changes
        section_regenerate(str)     - emitted when user requests section regeneration
        send_to_song_forge(str)     - emitted when user clicks "Send to Song Forge"
    """
    text_changed = Signal(str)
    section_regenerate = Signal(str)
    send_to_song_forge = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._streaming = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._word_count_label = QLabel("0 words \u2022 0 lines")
        self._word_count_label.setObjectName("caption")
        toolbar.addWidget(self._word_count_label)

        toolbar.addStretch()

        copy_btn = QPushButton("Copy")
        copy_btn.setObjectName("ghostBtn")
        copy_btn.setFixedHeight(28)
        copy_btn.clicked.connect(self._copy_to_clipboard)
        toolbar.addWidget(copy_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("ghostBtn")
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self._clear)
        toolbar.addWidget(clear_btn)

        self._forge_btn = QPushButton("\U0001f3b6 Send to Song Forge")
        self._forge_btn.setFixedHeight(30)
        self._forge_btn.clicked.connect(lambda: self.send_to_song_forge.emit(self.text))
        toolbar.addWidget(self._forge_btn)

        layout.addLayout(toolbar)

        # Editor
        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText(
            "Generated lyrics will appear here...\n\n"
            "You can also type or paste lyrics directly.\n"
            "Right-click on a section tag to regenerate just that section."
        )
        self._editor.setStyleSheet(f"""
            QPlainTextEdit {{
                font-family: "JetBrains Mono", "Cascadia Code", "Consolas", "Courier New", monospace;
                font-size: 13px;
                line-height: 1.6;
                padding: 16px;
            }}
        """)
        self._editor.setTabStopDistance(40)
        self._editor.textChanged.connect(self._on_text_changed)
        self._editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._editor.customContextMenuRequested.connect(self._show_context_menu)

        # Attach syntax highlighter
        self._highlighter = LyricsHighlighter(self._editor.document())

        layout.addWidget(self._editor, 1)

        # Status bar
        self._status = QLabel("")
        self._status.setObjectName("caption")
        self._status.setStyleSheet(f"color: {Palette.BLUE}; font-size: 11px;")
        layout.addWidget(self._status)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        return self._editor.toPlainText()

    @text.setter
    def text(self, value: str):
        self._editor.setPlainText(value)

    @property
    def is_empty(self) -> bool:
        return not self._editor.toPlainText().strip()

    # ── Streaming Support ──────────────────────────────────────────────────────

    def start_streaming(self):
        """Begin streaming mode — clear editor and prepare for token-by-token input."""
        self._streaming = True
        self._editor.clear()
        self._editor.setReadOnly(True)
        self._status.setText("Generating...")
        self._status.setStyleSheet(f"color: {Palette.BLUE}; font-size: 11px;")

    def append_token(self, token: str):
        """Append a single token during streaming generation."""
        if self._streaming:
            cursor = self._editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(token)
            self._editor.setTextCursor(cursor)
            self._editor.ensureCursorVisible()

    def stop_streaming(self):
        """End streaming mode — make editor editable again."""
        self._streaming = False
        self._editor.setReadOnly(False)
        self._status.setText("Generation complete")
        self._on_text_changed()

    # ── Section Detection ──────────────────────────────────────────────────────

    def get_sections(self) -> list[dict]:
        """
        Parse lyrics into sections.
        Returns list of dicts: {"tag": "[Chorus]", "start_line": 5, "end_line": 9, "content": "..."}
        """
        text = self.text
        lines = text.split("\n")
        sections = []
        current_tag = None
        current_start = 0
        current_lines = []

        tag_pattern = re.compile(r"^\[.+\]\s*$")

        for i, line in enumerate(lines):
            if tag_pattern.match(line.strip()):
                # Save previous section
                if current_tag:
                    sections.append({
                        "tag": current_tag,
                        "start_line": current_start,
                        "end_line": i - 1,
                        "content": "\n".join(current_lines).strip(),
                    })
                current_tag = line.strip()
                current_start = i
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_tag:
            sections.append({
                "tag": current_tag,
                "start_line": current_start,
                "end_line": len(lines) - 1,
                "content": "\n".join(current_lines).strip(),
            })

        return sections

    def get_section_at_cursor(self) -> Optional[str]:
        """Get the section tag that the cursor is currently inside."""
        cursor = self._editor.textCursor()
        line_number = cursor.blockNumber()
        sections = self.get_sections()

        for section in sections:
            if section["start_line"] <= line_number <= section["end_line"]:
                return section["tag"]
        return None

    def replace_section(self, tag: str, new_content: str):
        """Replace the content of a specific section."""
        text = self.text
        lines = text.split("\n")
        sections = self.get_sections()

        for section in sections:
            if section["tag"] == tag:
                # Find the content lines (after the tag line)
                start = section["start_line"] + 1
                end = section["end_line"] + 1

                new_lines = new_content.strip().split("\n")
                lines[start:end] = new_lines

                self.text = "\n".join(lines)
                return

    # ── Context Menu ───────────────────────────────────────────────────────────

    def _show_context_menu(self, position):
        menu = QMenu(self)

        # Standard edit actions
        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self._editor.undo)
        undo_action.setEnabled(self._editor.document().isUndoAvailable())
        menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.triggered.connect(self._editor.redo)
        redo_action.setEnabled(self._editor.document().isRedoAvailable())
        menu.addAction(redo_action)

        menu.addSeparator()

        cut_action = QAction("Cut", self)
        cut_action.triggered.connect(self._editor.cut)
        menu.addAction(cut_action)

        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self._editor.copy)
        menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self._editor.paste)
        menu.addAction(paste_action)

        menu.addSeparator()

        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self._editor.selectAll)
        menu.addAction(select_all_action)

        # Section-specific actions
        current_section = self.get_section_at_cursor()
        if current_section:
            menu.addSeparator()

            regen_action = QAction(f"\U0001f504 Regenerate {current_section}", self)
            regen_action.triggered.connect(lambda: self.section_regenerate.emit(current_section))
            menu.addAction(regen_action)

        menu.exec(self._editor.mapToGlobal(position))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_text_changed(self):
        text = self.text
        words = len(text.split()) if text.strip() else 0
        lines = text.count("\n") + 1 if text.strip() else 0
        sections = len(self.get_sections())
        self._word_count_label.setText(f"{words} words \u2022 {lines} lines \u2022 {sections} sections")
        self.text_changed.emit(text)

    def _copy_to_clipboard(self):
        QApplication.clipboard().setText(self.text)

    def _clear(self):
        self._editor.clear()
        self._status.setText("")

    def set_status(self, text: str, color: str = Palette.BLUE):
        """Set status text below the editor."""
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-size: 11px;")
