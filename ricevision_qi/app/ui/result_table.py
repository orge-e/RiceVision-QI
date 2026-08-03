from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


RESULT_COLUMNS = [
    "编号",
    "长度(px)",
    "宽度(px)",
    "面积(px²)",
    "BBox",
    "长宽比",
    "预测类别",
    "人工类别",
    "最终类别",
    "状态",
    "置信度",
]


class ResultTable(QTableWidget):
    def __init__(self) -> None:
        super().__init__(0, len(RESULT_COLUMNS))
        self.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def clear_results(self) -> None:
        self.setRowCount(0)

    def set_results(self, grains) -> None:
        self.setSortingEnabled(False)
        self.setRowCount(0)
        for row, grain in enumerate(grains):
            self.insertRow(row)
            self._set_row(row, grain)
        self.setSortingEnabled(True)

    def update_grain(self, grain) -> None:
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item is not None and item.text() == str(grain.id):
                self._set_row(row, grain)
                return

    def select_grain_by_id(self, grain_id: int) -> None:
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item is not None and item.text() == str(grain_id):
                self.selectRow(row)
                return

    def selected_grain_id(self) -> int | None:
        selected_items = self.selectedItems()
        if not selected_items:
            return None
        row = selected_items[0].row()
        item = self.item(row, 0)
        if item is None:
            return None
        try:
            return int(item.text())
        except ValueError:
            return None

    def _set_row(self, row: int, grain) -> None:
        features = grain.features
        values = [
            str(grain.id),
            _format_px(features.get("length_px", 0.0)),
            _format_px(features.get("width_px", 0.0)),
            f"{features.get('area_px', 0.0):.0f}",
            _format_bbox(grain.bbox),
            f"{features.get('aspect_ratio', 0.0):.2f}",
            getattr(grain, "predicted_class", grain.classification),
            getattr(grain, "manual_class", None) or "",
            getattr(grain, "final_class", grain.classification),
            getattr(grain, "status", "valid"),
            f"{grain.confidence:.2f}",
        ]
        for column, value in enumerate(values):
            self.setItem(row, column, QTableWidgetItem(value))


def _format_px(value: float) -> str:
    return f"{value:.1f}"


def _format_bbox(bbox: tuple[int, int, int, int]) -> str:
    x, y, w, h = bbox
    return f"{x},{y},{w},{h}"
