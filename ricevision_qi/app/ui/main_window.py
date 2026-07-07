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
    pixmap_from_rgb,
    read_image,
)
from ricevision_qi.app.core.pipeline import run_detection
from ricevision_qi.app.core.visualization import draw_detection_overlay
from ricevision_qi.app.ui.image_viewer import ImageViewer
from ricevision_qi.app.ui.result_table import ResultTable


DETECTION_TYPES = [
    "稻米缺陷粒检测",
    "稻谷出糙率检测",
    "整精米率检测",
]

STAT_DEFAULTS = {
    "米粒总数": 0,
    "正常粒": 0,
    "碎米": 0,
    "缺陷粒": 0,
    "杂质": 0,
}


class MainWindow(QMainWindow):
    WINDOW_TITLE = "RiceVision-QI：稻米质量视觉检测系统"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1280, 820)
        self.current_image_path: Path | None = None
        self.current_image_rgb = None
        self.current_detection_result = None
        self.stat_labels: dict[str, QLabel] = {}

        self.image_viewer = ImageViewer()
        self.result_table = ResultTable()
        self.result_table.itemSelectionChanged.connect(self.highlight_selected_grain)
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
        panel = QGroupBox("控制面板")
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        import_button = QPushButton("导入图像")
        detect_button = QPushButton("开始检测")
        clear_button = QPushButton("清空结果")
        export_button = QPushButton("导出报告")

        import_button.clicked.connect(self.import_image)
        detect_button.clicked.connect(self.start_detection)
        clear_button.clicked.connect(self.clear_results)
        export_button.clicked.connect(self.export_report)

        layout.addWidget(QLabel("检测类型"))
        layout.addWidget(self.detection_combo)
        layout.addSpacing(8)
        layout.addWidget(import_button)
        layout.addWidget(detect_button)
        layout.addWidget(clear_button)
        layout.addWidget(export_button)
        layout.addStretch(1)

        return panel

    def _build_stats_panel(self) -> QGroupBox:
        panel = QGroupBox("统计信息")
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
            "导入图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        try:
            loaded = read_image(Path(file_path))
            pixmap = pixmap_from_rgb(loaded.image_rgb)
        except UnsupportedImageFormatError as exc:
            QMessageBox.warning(self, "格式不支持", str(exc))
            return
        except ImageReadError as exc:
            QMessageBox.critical(self, "图像读取失败", str(exc))
            return

        self.current_image_path = Path(file_path)
        self.current_image_rgb = loaded.image_rgb
        self.current_detection_result = None
        self.image_viewer.set_image(pixmap)
        self.result_table.clear_results()
        self._reset_stats()

    def start_detection(self) -> None:
        if self.current_image_rgb is None:
            QMessageBox.information(self, "检测", "请先导入图像。")
            return

        try:
            result = run_detection(self.current_image_rgb)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            QMessageBox.critical(self, "检测失败", f"检测失败：{exc}")
            return

        self.current_detection_result = result
        self.image_viewer.set_image(pixmap_from_rgb(result.overlay_image))
        self.result_table.set_results(result.grains)
        self._update_stats(result.summary)

    def clear_results(self) -> None:
        self.current_image_path = None
        self.current_image_rgb = None
        self.current_detection_result = None
        self.image_viewer.clear_image()
        self.result_table.clear_results()
        self._reset_stats()

    def export_report(self) -> None:
        QMessageBox.information(self, "导出报告", "报告导出功能尚未实现。")

    def _reset_stats(self) -> None:
        for name, value in STAT_DEFAULTS.items():
            self.stat_labels[name].setText(f"{name}: {value}")

    def _update_stats(self, summary: dict) -> None:
        values = {
            "米粒总数": summary.get("total_grains", 0),
            "正常粒": summary.get("normal_count", 0),
            "碎米": summary.get("broken_count", 0),
            "缺陷粒": summary.get("defective_count", 0),
            "杂质": summary.get("impurity_count", 0),
        }
        for name, value in values.items():
            self.stat_labels[name].setText(f"{name}: {value}")

    def highlight_selected_grain(self) -> None:
        if self.current_detection_result is None:
            return
        selected_id = self.result_table.selected_grain_id()
        overlay = draw_detection_overlay(
            self.current_detection_result.original_image,
            self.current_detection_result.grains,
            selected_grain_id=selected_id,
        )
        self.image_viewer.set_image(pixmap_from_rgb(overlay))
