from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ricevision_qi.app.core.image_io import (
    ImageReadError,
    UnsupportedImageFormatError,
    load_image_pixmap,
)
from ricevision_qi.app.ui.image_viewer import ImageViewer
from ricevision_qi.app.ui.result_table import ResultTable


DETECTION_TYPES = [
    "Rice Defective Kernel Inspection",
    "Paddy Husked Rice Yield",
    "Head Rice Yield",
]

STAT_DEFAULTS = {
    "Total Grains": 0,
    "Normal": 0,
    "Broken": 0,
    "Defective": 0,
    "Impurity": 0,
}


class MainWindow(QMainWindow):
    WINDOW_TITLE = "RiceVision-QI: Rice Quality Inspection System"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1280, 820)
        self.current_image_path: Path | None = None
        self.stat_labels: dict[str, QLabel] = {}

        self.image_viewer = ImageViewer()
        self.result_table = ResultTable()
        self.detection_combo = QComboBox()
        self.detection_combo.addItems(DETECTION_TYPES)

        self._build_layout()
        self._reset_stats()

    def _build_layout(self) -> None:
        central = QWidget()
        root = QGridLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(self._build_control_panel(), 0, 0)
        root.addWidget(self.image_viewer, 0, 1)
        root.addWidget(self._build_stats_panel(), 0, 2)
        root.addWidget(self.result_table, 1, 0, 1, 3)

        root.setColumnStretch(0, 0)
        root.setColumnStretch(1, 1)
        root.setColumnStretch(2, 0)
        root.setRowStretch(0, 1)
        root.setRowStretch(1, 0)

        self.setCentralWidget(central)

    def _build_control_panel(self) -> QGroupBox:
        panel = QGroupBox("Control Panel")
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        import_button = QPushButton("Import Image")
        detect_button = QPushButton("Start Detection")
        clear_button = QPushButton("Clear Results")
        export_button = QPushButton("Export Report")

        import_button.clicked.connect(self.import_image)
        detect_button.clicked.connect(self.start_detection)
        clear_button.clicked.connect(self.clear_results)
        export_button.clicked.connect(self.export_report)

        layout.addWidget(QLabel("Detection Type"))
        layout.addWidget(self.detection_combo)
        layout.addSpacing(8)
        layout.addWidget(import_button)
        layout.addWidget(detect_button)
        layout.addWidget(clear_button)
        layout.addWidget(export_button)
        layout.addStretch(1)

        return panel

    def _build_stats_panel(self) -> QGroupBox:
        panel = QGroupBox("Statistics")
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for name in STAT_DEFAULTS:
            label = QLabel()
            label.setFrameShape(QFrame.Shape.StyledPanel)
            label.setMinimumHeight(36)
            self.stat_labels[name] = label
            layout.addWidget(label)

        layout.addStretch(1)
        return panel

    def import_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        try:
            pixmap = load_image_pixmap(Path(file_path))
        except UnsupportedImageFormatError as exc:
            QMessageBox.warning(self, "Unsupported Format", str(exc))
            return
        except ImageReadError as exc:
            QMessageBox.critical(self, "Image Read Failed", str(exc))
            return

        self.current_image_path = Path(file_path)
        self.image_viewer.set_image(pixmap)

    def start_detection(self) -> None:
        QMessageBox.information(self, "Detection", "Detection pipeline is not implemented yet.")

    def clear_results(self) -> None:
        self.current_image_path = None
        self.image_viewer.clear_image()
        self.result_table.clear_results()
        self._reset_stats()

    def export_report(self) -> None:
        QMessageBox.information(self, "Export Report", "Report export is not implemented yet.")

    def _reset_stats(self) -> None:
        for name, value in STAT_DEFAULTS.items():
            self.stat_labels[name].setText(f"{name}: {value}")
