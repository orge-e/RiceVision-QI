import csv
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ricevision_qi.app.core.config import PROFILE_DISPLAY_NAMES, load_profile_config
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

DEBUG_VIEW_OPTIONS = [
    ("原图", "original"),
    ("光照校正图", "illumination_corrected"),
    ("初始二值图", "initial_mask"),
    ("清理后 Mask", "cleaned_mask"),
    ("Watershed 标记图", "watershed_markers"),
    ("最终轮廓图", "overlay"),
]
DEBUG_VIEW_KEYS = dict(DEBUG_VIEW_OPTIONS)

SEGMENTATION_METHODS = [
    ("Lab 颜色阈值", "lab_threshold"),
    ("灰度 Otsu", "gray_otsu"),
    ("HSV 阈值", "hsv_threshold"),
    ("自适应阈值", "adaptive_threshold"),
]

PROFILE_OPTIONS = [
    ("Clean Background", "clean_background"),
    ("Plastic Bag Rice", "plastic_bag_rice"),
    ("Dense Rice Cluster", "dense_rice_cluster"),
    ("YOLOv8s Seg Rice（推荐）", "yolo_seg_rice_v8s"),
    ("YOLOv8m Seg Rice", "yolo_seg_rice_v8m"),
    ("YOLO Seg Rice", "yolo_seg_rice"),
    ("YOLO Seg Rice S", "yolo_seg_rice_s"),
    ("Debug Fast", "debug_fast"),
    ("Custom", "custom"),
]

STAT_DEFAULTS = {
    "米粒总数": 0,
    "正常粒": 0,
    "碎米": 0,
    "缺陷粒": 0,
    "杂质": 0,
    "未知": 0,
}

CLASS_OPTIONS = ["normal", "broken", "defective", "impurity", "unknown"]


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
        self.roi_label = QLabel("ROI: 未选择")
        self.profile_status_label = QLabel("当前 Profile: Custom")
        self.instance_info_label = QLabel("实例: 未选择")

        self.image_viewer = ImageViewer()
        self.image_viewer.roi_changed.connect(self._on_roi_changed)
        self.image_viewer.image_clicked.connect(self.select_grain_at_image_point)

        self.result_table = ResultTable()
        self.result_table.itemSelectionChanged.connect(self.highlight_selected_grain)

        self.detection_combo = QComboBox()
        self.detection_combo.addItems(DETECTION_TYPES)

        self.profile_combo = QComboBox()
        for label, value in PROFILE_OPTIONS:
            self.profile_combo.addItem(label, value)
        self.profile_combo.currentIndexChanged.connect(self.apply_profile_to_controls)

        self.debug_view_combo = QComboBox()
        for label, key in DEBUG_VIEW_OPTIONS:
            self.debug_view_combo.addItem(label, key)
        self.debug_view_combo.setCurrentText("最终轮廓图")
        self.debug_view_combo.currentTextChanged.connect(self.update_debug_view)

        self.segmentation_method_combo = QComboBox()
        for label, value in SEGMENTATION_METHODS:
            self.segmentation_method_combo.addItem(label, value)

        self.kernel_size_spin = _spin_box(1, 31, 3, 2)
        self.open_iter_spin = _spin_box(0, 10, 1)
        self.close_iter_spin = _spin_box(0, 10, 0)
        self.min_area_spin = _spin_box(1, 1_000_000, 120, 10)
        self.max_area_spin = _spin_box(1, 10_000_000, 100000, 100)
        self.watershed_ratio_spin = _double_spin_box(0.05, 0.95, 0.20, 0.05)
        self.fill_holes_check = QCheckBox("启用")
        self.fill_holes_check.setChecked(False)
        self.lab_l_min_spin = _spin_box(0, 255, 120, 5)
        self.lab_b_min_spin = _spin_box(0, 255, 140, 2)
        self.lab_a_min_spin = _spin_box(0, 255, 120, 1)
        self.lab_a_max_spin = _spin_box(0, 255, 145, 1)
        self.hsv_s_max_spin = _spin_box(0, 255, 70, 5)

        self.manual_class_combo = QComboBox()
        self.manual_class_combo.addItems(CLASS_OPTIONS)

        self._build_layout()
        self.apply_profile_to_controls()
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
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        import_button = QPushButton("导入图像")
        detect_button = QPushButton("开始/重新检测")
        clear_button = QPushButton("清空结果")
        clear_roi_button = QPushButton("清除 ROI")
        export_button = QPushButton("导出 CSV")
        import_button.clicked.connect(self.import_image)
        detect_button.clicked.connect(self.start_detection)
        clear_button.clicked.connect(self.clear_results)
        clear_roi_button.clicked.connect(self.image_viewer.clear_roi)
        export_button.clicked.connect(self.export_report)

        layout.addWidget(QLabel("检测类型"))
        layout.addWidget(self.detection_combo)
        layout.addWidget(QLabel("算法 Profile"))
        layout.addWidget(self.profile_combo)
        layout.addWidget(self.profile_status_label)
        layout.addWidget(QLabel("调试视图"))
        layout.addWidget(self.debug_view_combo)
        layout.addSpacing(8)
        layout.addWidget(import_button)
        layout.addWidget(detect_button)
        layout.addWidget(clear_roi_button)
        layout.addWidget(clear_button)
        layout.addWidget(export_button)
        layout.addWidget(self._build_parameter_panel())
        layout.addStretch(1)
        return panel

    def _build_parameter_panel(self) -> QGroupBox:
        panel = QGroupBox("分割参数")
        layout = QGridLayout(panel)
        rows = [
            ("分割方法", self.segmentation_method_combo),
            ("核大小", self.kernel_size_spin),
            ("开运算次数", self.open_iter_spin),
            ("闭运算次数", self.close_iter_spin),
            ("最小面积", self.min_area_spin),
            ("最大面积", self.max_area_spin),
            ("分水岭距离比例", self.watershed_ratio_spin),
            ("填洞", self.fill_holes_check),
            ("Lab 亮度下限", self.lab_l_min_spin),
            ("Lab 黄色下限", self.lab_b_min_spin),
            ("Lab 红绿下限", self.lab_a_min_spin),
            ("Lab 红绿上限", self.lab_a_max_spin),
            ("HSV 饱和度上限", self.hsv_s_max_spin),
        ]
        for row, (label, widget) in enumerate(rows):
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)
        hint = QLabel("修改参数或 ROI 后，请点击“开始/重新检测”刷新结果。")
        hint.setWordWrap(True)
        layout.addWidget(hint, len(rows), 0, 1, 2)
        return panel

    def _build_stats_panel(self) -> QGroupBox:
        panel = QGroupBox("统计信息")
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for name in STAT_DEFAULTS:
            label = QLabel()
            label.setFrameShape(QFrame.Shape.StyledPanel)
            label.setMinimumHeight(34)
            self.stat_labels[name] = label
            layout.addWidget(label)
        self.roi_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.roi_label.setWordWrap(True)
        layout.addWidget(self.roi_label)
        layout.addWidget(self._build_manual_review_panel())
        layout.addStretch(1)
        return panel

    def _build_manual_review_panel(self) -> QGroupBox:
        panel = QGroupBox("人工复核")
        layout = QVBoxLayout(panel)
        self.instance_info_label.setWordWrap(True)
        apply_button = QPushButton("应用人工类别")
        false_positive_button = QPushButton("标记为误检")
        apply_button.clicked.connect(self.apply_manual_class)
        false_positive_button.clicked.connect(self.mark_false_positive)
        layout.addWidget(self.instance_info_label)
        layout.addWidget(QLabel("人工类别"))
        layout.addWidget(self.manual_class_combo)
        layout.addWidget(apply_button)
        layout.addWidget(false_positive_button)
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
        self.image_viewer.clear_roi()
        self.result_table.clear_results()
        self._reset_stats()

    def start_detection(self) -> None:
        if self.current_image_rgb is None:
            QMessageBox.information(self, "检测", "请先导入图像。")
            return
        try:
            result = run_detection(self.current_image_rgb, self._segmentation_config())
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            QMessageBox.critical(self, "检测失败", f"检测失败：{exc}")
            return

        self.current_detection_result = result
        self.debug_view_combo.setCurrentText("最终轮廓图")
        self.update_debug_view()
        self.result_table.set_results(result.grains)
        self._update_stats(result.summary)
        self._on_roi_changed(result.roi)
        backend_message = (result.backend_stats or {}).get("message")
        if backend_message:
            QMessageBox.information(self, "YOLO-seg", backend_message)
        if not result.grains:
            QMessageBox.information(
                self,
                "检测结果",
                "未检测到米粒实例。请调整分割参数或图像采集条件。",
            )

    def clear_results(self) -> None:
        self.current_image_path = None
        self.current_image_rgb = None
        self.current_detection_result = None
        self.image_viewer.clear_image()
        self.result_table.clear_results()
        self._reset_stats()
        self.instance_info_label.setText("实例: 未选择")

    def export_report(self) -> None:
        if self.current_detection_result is None:
            QMessageBox.information(self, "导出 CSV", "当前没有检测结果。")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "ricevision_results.csv", "CSV (*.csv)")
        if not file_path:
            return
        with open(file_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "id",
                    "bbox",
                    "length_px",
                    "width_px",
                    "area_px",
                    "aspect_ratio",
                    "predicted_class",
                    "manual_class",
                    "final_class",
                    "status",
                    "confidence",
                    "roi",
                ],
            )
            writer.writeheader()
            for grain in self.current_detection_result.grains:
                writer.writerow(
                    {
                        "id": grain.id,
                        "bbox": grain.bbox,
                        "length_px": grain.features.get("length_px", 0.0),
                        "width_px": grain.features.get("width_px", 0.0),
                        "area_px": grain.features.get("area_px", 0.0),
                        "aspect_ratio": grain.features.get("aspect_ratio", 0.0),
                        "predicted_class": grain.predicted_class,
                        "manual_class": grain.manual_class or "",
                        "final_class": grain.final_class,
                        "status": grain.status,
                        "confidence": grain.confidence,
                        "roi": self.current_detection_result.roi,
                    }
                )
        QMessageBox.information(self, "导出 CSV", f"已导出：{file_path}")

    def update_debug_view(self) -> None:
        if self.current_detection_result is None:
            return
        key = self.debug_view_combo.currentData() or DEBUG_VIEW_KEYS.get(self.debug_view_combo.currentText(), "overlay")
        image = self.current_detection_result.debug_results.get(key)
        if image is None:
            return
        self.image_viewer.set_image(pixmap_from_rgb(_debug_image_to_rgb(image)))

    def highlight_selected_grain(self) -> None:
        grain = self._selected_grain()
        if self.current_detection_result is None or grain is None:
            return
        self._show_selected_grain(grain)

    def select_grain_at_image_point(self, image_x: int, image_y: int) -> None:
        if self.current_detection_result is None:
            return
        for grain in self.current_detection_result.grains:
            if cv2.pointPolygonTest(grain.contour, (float(image_x), float(image_y)), False) >= 0:
                self.result_table.select_grain_by_id(grain.id)
                self._show_selected_grain(grain)
                return

    def apply_manual_class(self) -> None:
        grain = self._selected_grain()
        if grain is None:
            return
        grain.manual_class = self.manual_class_combo.currentText()
        grain.final_class = grain.manual_class
        self._refresh_after_manual_update(grain)

    def mark_false_positive(self) -> None:
        grain = self._selected_grain()
        if grain is None:
            return
        grain.status = "false_positive"
        grain.final_class = grain.manual_class or grain.predicted_class
        self._refresh_after_manual_update(grain)

    def apply_profile_to_controls(self) -> None:
        profile = self.profile_combo.currentData() if hasattr(self, "profile_combo") else "custom"
        if profile == "custom":
            self.profile_status_label.setText("当前 Profile: Custom")
            return
        config = load_profile_config(profile)
        profile_name = PROFILE_DISPLAY_NAMES.get(config["profile_name"], config["profile_name"])
        self.profile_status_label.setText(f"当前 Profile: {profile_name}")
        self._set_combo_data(self.segmentation_method_combo, config.get("segmentation_method", "lab_threshold"))
        self.kernel_size_spin.setValue(int(config.get("kernel_size", 3)))
        self.open_iter_spin.setValue(int(config.get("open_iter", 1)))
        self.close_iter_spin.setValue(int(config.get("close_iter", 0)))
        self.min_area_spin.setValue(int(config.get("min_area", 120)))
        self.max_area_spin.setValue(int(config.get("max_area", 100000)))
        self.watershed_ratio_spin.setValue(float(config.get("watershed_dist_ratio", 0.2)))
        self.fill_holes_check.setChecked(bool(config.get("fill_holes", False)))

    def _segmentation_config(self) -> dict:
        min_area = self.min_area_spin.value()
        max_area = self.max_area_spin.value()
        method = self.segmentation_method_combo.currentData()
        profile = self.profile_combo.currentData()
        config = {
            "profile": profile if profile != "custom" else None,
            "segmentation_method": method,
            "method": method,
            "kernel_size": self.kernel_size_spin.value(),
            "morph_kernel_size": self.kernel_size_spin.value(),
            "open_iter": self.open_iter_spin.value(),
            "close_iter": self.close_iter_spin.value(),
            "min_area": min_area,
            "max_area": max_area,
            "min_grain_area": min_area,
            "max_grain_area": max_area,
            "watershed_dist_ratio": self.watershed_ratio_spin.value(),
            "fill_holes": self.fill_holes_check.isChecked(),
            "lab_l_min": self.lab_l_min_spin.value(),
            "lab_b_min": self.lab_b_min_spin.value(),
            "lab_a_min": self.lab_a_min_spin.value(),
            "lab_a_max": self.lab_a_max_spin.value(),
            "hsv_s_max": self.hsv_s_max_spin.value(),
            "roi": self.image_viewer.roi(),
        }
        return config

    def _selected_grain(self):
        if self.current_detection_result is None:
            return None
        selected_id = self.result_table.selected_grain_id()
        if selected_id is None:
            return None
        return next((grain for grain in self.current_detection_result.grains if grain.id == selected_id), None)

    def _show_selected_grain(self, grain) -> None:
        overlay = draw_detection_overlay(
            self.current_detection_result.original_image,
            self.current_detection_result.grains,
            selected_grain_id=grain.id,
        )
        self.debug_view_combo.setCurrentText("最终轮廓图")
        self.image_viewer.set_image(pixmap_from_rgb(overlay))
        self.instance_info_label.setText(
            "\n".join(
                [
                    f"ID: {grain.id}",
                    f"bbox: {grain.bbox}",
                    f"area: {grain.features.get('area_px', 0.0):.0f}",
                    f"length: {grain.features.get('length_px', 0.0):.1f}",
                    f"width: {grain.features.get('width_px', 0.0):.1f}",
                    f"aspect_ratio: {grain.features.get('aspect_ratio', 0.0):.2f}",
                    f"class: {grain.final_class}",
                    f"status: {grain.status}",
                ]
            )
        )
        self._set_combo_text(self.manual_class_combo, grain.manual_class or grain.final_class)

    def _refresh_after_manual_update(self, grain) -> None:
        self.result_table.update_grain(grain)
        self._show_selected_grain(grain)
        self._update_stats(self.current_detection_result.summary)

    def _reset_stats(self) -> None:
        for name, value in STAT_DEFAULTS.items():
            self.stat_labels[name].setText(f"{name}: {value}")
        self._on_roi_changed(self.image_viewer.roi() if hasattr(self, "image_viewer") else None)

    def _update_stats(self, summary: dict) -> None:
        if self.current_detection_result is not None:
            summary = self._summary_from_grains(self.current_detection_result.grains)
            self.current_detection_result.summary = summary
        values = {
            "米粒总数": summary.get("total_grains", 0),
            "正常粒": summary.get("normal_count", 0),
            "碎米": summary.get("broken_count", 0),
            "缺陷粒": summary.get("defective_count", 0),
            "杂质": summary.get("impurity_count", 0),
            "未知": summary.get("unknown_count", 0),
        }
        for name, value in values.items():
            self.stat_labels[name].setText(f"{name}: {value}")

    def _summary_from_grains(self, grains) -> dict:
        summary = {
            "total_grains": 0,
            "normal_count": 0,
            "broken_count": 0,
            "defective_count": 0,
            "impurity_count": 0,
            "unknown_count": 0,
        }
        for grain in grains:
            if grain.status == "false_positive":
                continue
            summary["total_grains"] += 1
            key = f"{grain.final_class}_count"
            if key in summary:
                summary[key] += 1
            else:
                summary["unknown_count"] += 1
        return summary

    def _on_roi_changed(self, roi) -> None:
        if roi is None:
            self.roi_label.setText("ROI: 未选择")
            return
        x, y, w, h = roi
        self.roi_label.setText(f"ROI:\nx={x}\ny={y}\nwidth={w}\nheight={h}")

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def _set_combo_text(self, combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)


def _spin_box(minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setSingleStep(step)
    return spin


def _double_spin_box(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(2)
    spin.setValue(value)
    spin.setSingleStep(step)
    return spin


def _debug_image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return np.ascontiguousarray(image.astype(np.uint8))
    if image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    normalized = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.dtype == np.int32:
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
