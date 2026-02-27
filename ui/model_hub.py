"""
Slunder Studio v0.1.0 — Model Hub UI
Grid view of all models with live download progress, speed tracking,
partial download detection, and one-click download/delete.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QProgressBar,
    QLineEdit, QComboBox, QCheckBox, QDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

from ui.theme import Palette
from core.model_manager import ModelManager, ModelInfo, ModelStatus, ModelCategory
from core.settings import Settings
from core.workers import DownloadWorker


class HFTokenDialog(QDialog):
    """Inline dialog to paste a HuggingFace token for gated model downloads."""

    def __init__(self, model_name: str, repo_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HuggingFace Token")
        self.setFixedSize(480, 240)
        self.token = ""
        self._build_ui(model_name, repo_id)

    def _build_ui(self, model_name: str, repo_id: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel(f"<b>{model_name}</b> requires a HuggingFace access token")
        title.setStyleSheet(f"font-size: 15px; color: {Palette.TEXT};")
        title.setWordWrap(True)
        layout.addWidget(title)

        link_row = QHBoxLayout()
        link_row.setSpacing(8)
        open_btn = QPushButton("Get Token from HuggingFace")
        open_btn.setFixedHeight(34)
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/settings/tokens"))
        )
        link_row.addWidget(open_btn)
        link_row.addStretch()
        layout.addLayout(link_row)

        self._token_input = QLineEdit()
        self._token_input.setPlaceholderText("Paste token here  (starts with hf_)")
        self._token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_input.setFixedHeight(38)
        layout.addWidget(self._token_input)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 36)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._save_btn = QPushButton("Save & Download")
        self._save_btn.setFixedSize(160, 36)
        self._save_btn.setObjectName("accentBtn")
        self._save_btn.clicked.connect(self._accept)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

    def _accept(self):
        t = self._token_input.text().strip()
        if t and t.startswith("hf_"):
            self.token = t
            self.accept()
        else:
            self._token_input.setPlaceholderText("Must start with hf_...")
            self._token_input.setStyleSheet(f"border: 1px solid {Palette.RED};")


class ModelCard(QFrame):
    """A single model card with integrated download panel."""

    download_requested = Signal(str)
    cancel_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, info: ModelInfo, parent=None):
        super().__init__(parent)
        self.model_id = info.model_id
        self.info = info
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMinimumHeight(140)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        # -- Header: name + badge --
        header = QHBoxLayout()
        header.setSpacing(8)

        self._name_label = QLabel(self.info.name)
        self._name_label.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: "
            f"{Palette.BLUE if self.info.is_core else Palette.TEXT};"
        )
        self._name_label.setWordWrap(True)
        header.addWidget(self._name_label, 1)

        self._status_badge = QLabel()
        self._status_badge.setFixedHeight(22)
        header.addWidget(self._status_badge)
        layout.addLayout(header)

        # -- Description --
        desc = QLabel(self.info.description)
        desc.setWordWrap(True)
        desc.setMaximumHeight(40)
        desc.setStyleSheet(f"font-size: 12px; color: {Palette.SUBTEXT0};")
        layout.addWidget(desc)

        # -- Stats row --
        stats = QHBoxLayout()
        stats.setSpacing(12)
        for text in [
            f"{self.info.vram_gb:.1f} GB VRAM",
            f"{self.info.disk_gb:.1f} GB disk",
            self.info.license,
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"font-size: 11px; color: {Palette.OVERLAY0};")
            stats.addWidget(lbl)
        if getattr(self.info, "gated", False):
            g = QLabel("Token Required")
            g.setStyleSheet(f"font-size: 11px; color: {Palette.OVERLAY0};")
            stats.addWidget(g)
        stats.addStretch()
        layout.addLayout(stats)

        # -- Download panel (hidden by default, expands inline) --
        self._dl_panel = QFrame()
        self._dl_panel.setVisible(False)
        dl_layout = QVBoxLayout(self._dl_panel)
        dl_layout.setContentsMargins(0, 6, 0, 2)
        dl_layout.setSpacing(4)

        # Progress bar — gradient fill, rounded
        self._progress = QProgressBar()
        self._progress.setFixedHeight(14)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {Palette.SURFACE0};
                border: none;
                border-radius: 7px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Palette.BLUE}, stop:1 {Palette.MAUVE}
                );
                border-radius: 7px;
            }}
        """)
        dl_layout.addWidget(self._progress)

        # Info row: percent | size | speed | cancel btn
        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        self._pct_label = QLabel("0%")
        self._pct_label.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Palette.BLUE};"
        )
        self._pct_label.setFixedWidth(40)
        info_row.addWidget(self._pct_label)

        self._size_label = QLabel("")
        self._size_label.setStyleSheet(
            f"font-size: 11px; color: {Palette.SUBTEXT0};"
        )
        info_row.addWidget(self._size_label)

        info_row.addStretch()

        self._speed_label = QLabel("")
        self._speed_label.setStyleSheet(
            f"font-size: 11px; color: {Palette.OVERLAY0};"
        )
        info_row.addWidget(self._speed_label)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedSize(60, 24)
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 11px; padding: 2px 8px;
                background: {Palette.SURFACE1}; color: {Palette.RED};
                border: 1px solid {Palette.SURFACE2}; border-radius: 4px;
            }}
            QPushButton:hover {{ background: {Palette.SURFACE2}; }}
        """)
        self._cancel_btn.clicked.connect(
            lambda: self.cancel_requested.emit(self.model_id)
        )
        info_row.addWidget(self._cancel_btn)

        dl_layout.addLayout(info_row)
        layout.addWidget(self._dl_panel)

        # -- Action button --
        self._action_btn = QPushButton("Download")
        self._action_btn.setFixedHeight(32)
        self._action_btn.clicked.connect(self._on_action)
        layout.addWidget(self._action_btn)

    def _on_action(self):
        mgr = ModelManager()
        status = mgr.get_status(self.model_id)
        if status in (
            ModelStatus.NOT_DOWNLOADED, ModelStatus.ERROR, ModelStatus.PARTIAL
        ):
            self.download_requested.emit(self.model_id)
        elif status == ModelStatus.DOWNLOADED:
            self.delete_requested.emit(self.model_id)

    def update_status(self, status: ModelStatus):
        """Update the card visual state based on model status."""
        btn = self._action_btn

        # Pip-managed models
        if getattr(self.info, "pip_managed", False):
            self._set_badge("Auto-managed", Palette.GREEN)
            btn.setText("Installed with engine")
            btn.setEnabled(False)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)
            return

        if status == ModelStatus.NOT_DOWNLOADED:
            self._set_badge("Not Downloaded", Palette.OVERLAY0)
            btn.setText("Download")
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

        elif status == ModelStatus.PARTIAL:
            self._set_badge("Incomplete", Palette.PEACH)
            btn.setText("Resume Download")
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

        elif status == ModelStatus.DOWNLOADING:
            self._set_badge("Downloading", Palette.BLUE)
            btn.setVisible(False)
            self._dl_panel.setVisible(True)
            self._progress.setValue(0)
            self._pct_label.setText("0%")
            self._size_label.setText("Starting...")
            self._speed_label.setText("")

        elif status == ModelStatus.DOWNLOADED:
            self._set_badge("Ready", Palette.GREEN)
            btn.setText("Delete")
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

        elif status == ModelStatus.LOADED:
            self._set_badge("Active", Palette.BLUE)
            btn.setText("In Use")
            btn.setEnabled(False)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

        elif status == ModelStatus.LOADING:
            self._set_badge("Loading...", Palette.YELLOW)
            btn.setEnabled(False)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

        elif status == ModelStatus.ERROR:
            self._set_badge("Error", Palette.RED)
            btn.setText("Retry Download")
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

    def _set_badge(self, text: str, color: str):
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(
            f"background: rgba({self._hex_to_rgba(color)},40); "
            f"color: {color}; padding: 2px 10px; border-radius: 11px; "
            f"font-size: 11px; font-weight: 600;"
        )

    @staticmethod
    def _hex_to_rgba(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        if len(h) == 6:
            return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
        return "128,128,128"

    def update_download_progress(self, pct: int, size_text: str = "",
                                 speed_text: str = ""):
        """Update the download panel with live metrics."""
        self._progress.setValue(pct)
        self._pct_label.setText(f"{pct}%")
        if size_text:
            self._size_label.setText(size_text)
        if speed_text:
            self._speed_label.setText(speed_text)

    def set_progress(self, pct: int):
        """Legacy compat — route through new method."""
        self.update_download_progress(pct)


class ModelHubView(QWidget):
    """Model Hub page with grid of model cards, search/filter, and disk usage."""

    def __init__(self, toast_mgr=None, parent=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._cards: dict[str, ModelCard] = {}
        self._workers: dict[str, DownloadWorker] = {}
        self._mgr = ModelManager()
        self._build_ui()
        self._connect_signals()
        self._refresh_all_cards()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Model Hub")
        title.setObjectName("heading")
        layout.addWidget(title)

        subtitle = QLabel(
            "Download and manage AI models. "
            "Only one large model is loaded at a time to fit within your GPU memory."
        )
        subtitle.setObjectName("caption")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # GPU status bar
        self._gpu_bar = QFrame()
        self._gpu_bar.setObjectName("accentCard")
        gpu_layout = QHBoxLayout(self._gpu_bar)
        gpu_layout.setContentsMargins(14, 10, 14, 10)

        self._gpu_label = QLabel("GPU: Detecting...")
        self._gpu_label.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {Palette.BLUE};"
        )
        gpu_layout.addWidget(self._gpu_label)
        gpu_layout.addStretch()

        self._disk_label = QLabel("Disk usage: calculating...")
        self._disk_label.setStyleSheet(
            f"font-size: 12px; color: {Palette.SUBTEXT0};"
        )
        gpu_layout.addWidget(self._disk_label)
        layout.addWidget(self._gpu_bar)

        # Filter bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(12)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search models...")
        self._search.setFixedHeight(36)
        self._search.textChanged.connect(self._filter_cards)
        filter_bar.addWidget(self._search, 1)

        self._category_filter = QComboBox()
        self._category_filter.addItem("All Categories", "all")
        for cat in ModelCategory:
            self._category_filter.addItem(
                cat.value.replace("_", " ").title(), cat.value
            )
        self._category_filter.setFixedHeight(36)
        self._category_filter.currentIndexChanged.connect(self._filter_cards)
        filter_bar.addWidget(self._category_filter)

        self._downloaded_only = QCheckBox("Downloaded only")
        self._downloaded_only.stateChanged.connect(self._filter_cards)
        filter_bar.addWidget(self._downloaded_only)
        layout.addLayout(filter_bar)

        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(16)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)

        col = 0
        row = 0
        for model_id, info in self._mgr.registry.items():
            card = ModelCard(info)
            card.download_requested.connect(self._start_download)
            card.cancel_requested.connect(self._cancel_download)
            card.delete_requested.connect(self._delete_model)
            self._cards[model_id] = card
            self._grid_layout.addWidget(card, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        self._grid_layout.setRowStretch(row + 1, 1)
        scroll.setWidget(self._grid_container)
        layout.addWidget(scroll, 1)

    def _connect_signals(self):
        self._mgr.status_changed.connect(self._on_status_changed)
        self._mgr.gpu_status_changed.connect(self._on_gpu_changed)

    def _refresh_all_cards(self):
        for model_id, card in self._cards.items():
            status = self._mgr.get_status(model_id)
            card.update_status(status)
        self._update_gpu_display()
        self._update_disk_display()

        # Alert user about partial downloads on startup
        partials = [
            mid for mid in self._mgr.registry
            if self._mgr.get_status(mid) == ModelStatus.PARTIAL
        ]
        if partials and self.toast_mgr:
            names = ", ".join(
                self._mgr.get_model_info(m).name for m in partials
            )
            self.toast_mgr.warning(
                f"Incomplete downloads detected: {names}. Click Resume to finish."
            )

    def _on_status_changed(self, model_id: str, status_str: str):
        if model_id in self._cards:
            status = ModelStatus(status_str)
            self._cards[model_id].update_status(status)
        self._update_disk_display()

    def _on_gpu_changed(self, gpu_info: dict):
        self._update_gpu_display(gpu_info)

    def _update_gpu_display(self, gpu_info: dict = None):
        if gpu_info is None:
            gpu_info = self._mgr.get_gpu_status()
        if gpu_info.get("available"):
            name = gpu_info["name"]
            total = gpu_info["total_gb"]
            used = gpu_info["used_gb"]
            current = gpu_info.get("current_model_name", "None")
            self._gpu_label.setText(
                f"{name}  |  {used:.1f} / {total:.1f} GB  |  "
                f"Active: {current or 'None'}"
            )
        else:
            self._gpu_label.setText(
                "No CUDA GPU detected — models will run on CPU (much slower)"
            )
            self._gpu_label.setStyleSheet(
                f"font-size: 13px; font-weight: 600; color: {Palette.YELLOW};"
            )

    def _update_disk_display(self):
        usage = self._mgr.get_total_disk_usage()
        downloaded = sum(
            1 for s in self._mgr._status.values()
            if s in (ModelStatus.DOWNLOADED, ModelStatus.LOADED)
        )
        total = len(self._mgr.registry)
        self._disk_label.setText(
            f"{usage:.1f} GB on disk  |  {downloaded}/{total} models ready"
        )

    def _filter_cards(self):
        search = self._search.text().lower()
        cat_filter = self._category_filter.currentData()
        downloaded_only = self._downloaded_only.isChecked()

        for model_id, card in self._cards.items():
            info = self._mgr.get_model_info(model_id)
            status = self._mgr.get_status(model_id)
            visible = True
            if search and search not in info.name.lower() \
                    and search not in info.description.lower():
                visible = False
            if cat_filter != "all" and info.category.value != cat_filter:
                visible = False
            if downloaded_only and status not in (
                ModelStatus.DOWNLOADED, ModelStatus.LOADED
            ):
                visible = False
            card.setVisible(visible)

    # -- Download Management -----------------------------------------------

    def _start_download(self, model_id: str):
        """Start or resume downloading a model in a background thread."""
        if model_id in self._workers:
            return

        info = self._mgr.get_model_info(model_id)

        # Gated model check
        if getattr(info, "gated", False):
            token = self._mgr._get_hf_token()
            if not token:
                dlg = HFTokenDialog(info.name, info.source, parent=self)
                if dlg.exec() == QDialog.DialogCode.Accepted and dlg.token:
                    Settings().set("model_hub.hf_token", dlg.token)
                    if self.toast_mgr:
                        self.toast_mgr.success("HuggingFace token saved!")
                else:
                    return

        # Wipe partial download files before fresh start
        if self._mgr.has_partial_download(model_id):
            import shutil
            cache_dir = self._mgr.get_cache_dir(model_id)
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)

        card = self._cards[model_id]
        card.update_status(ModelStatus.DOWNLOADING)

        worker = DownloadWorker(self._mgr.download_model, model_id)

        # Wire all progress signals into the card
        worker.progress.connect(
            lambda pct, mid=model_id: self._on_dl_progress(mid, pct)
        )
        worker.speed.connect(
            lambda s, mid=model_id: self._on_dl_speed(mid, s)
        )
        worker.downloaded.connect(
            lambda s, mid=model_id: self._on_dl_size(mid, s)
        )
        worker.finished.connect(self._on_download_finished)
        worker.error.connect(
            lambda err, mid=model_id: self._on_download_error(mid, err)
        )

        self._workers[model_id] = worker
        worker.start()

        if self.toast_mgr:
            self.toast_mgr.info(f"Downloading {info.name}...")

    def _on_dl_progress(self, model_id: str, pct: int):
        if model_id in self._cards:
            self._cards[model_id].update_download_progress(pct)

    def _on_dl_speed(self, model_id: str, speed: str):
        if model_id in self._cards:
            self._cards[model_id]._speed_label.setText(speed)

    def _on_dl_size(self, model_id: str, size: str):
        if model_id in self._cards:
            self._cards[model_id]._size_label.setText(size)

    def _cancel_download(self, model_id: str):
        """Cancel an active download and mark as partial."""
        if model_id in self._workers:
            self._workers[model_id].cancel()
            del self._workers[model_id]
        self._mgr._set_status(model_id, ModelStatus.PARTIAL)
        if model_id in self._cards:
            self._cards[model_id].update_status(ModelStatus.PARTIAL)
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.info(f"{info.name} download cancelled.")

    def _on_download_finished(self, model_id: str):
        if model_id in self._workers:
            del self._workers[model_id]
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.success(f"{info.name} downloaded successfully!")

    def _on_download_error(self, model_id: str, error: str):
        if model_id in self._workers:
            del self._workers[model_id]
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.error(f"Failed to download {info.name}: {error}")

    def _delete_model(self, model_id: str):
        """Delete a downloaded model's cache."""
        info = self._mgr.get_model_info(model_id)
        if info and getattr(info, "pip_managed", False):
            if self.toast_mgr:
                self.toast_mgr.info(
                    f"{info.name} is managed by pip — uninstall via pip if needed."
                )
            return

        import shutil
        cache_dir = self._mgr.get_cache_dir(model_id)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        self._mgr._set_status(model_id, ModelStatus.NOT_DOWNLOADED)
        self._cards[model_id].update_status(ModelStatus.NOT_DOWNLOADED)
        self._update_disk_display()
        if self.toast_mgr:
            self.toast_mgr.info(f"{info.name} removed from disk.")
