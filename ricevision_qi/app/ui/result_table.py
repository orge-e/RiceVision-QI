from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


RESULT_COLUMNS = [
    "编号",
    "长度",
    "宽度",
    "面积",
    "长宽比",
    "类别",
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
            features = grain.features
            self.insertRow(row)
            values = [
                str(grain.id),
                _format_px(features.get("length_px", 0.0)),
                _format_px(features.get("width_px", 0.0)),
                f"{features.get('area_px', 0.0):.0f}",
                f"{features.get('aspect_ratio', 0.0):.2f}",
                grain.classification,
                f"{grain.confidence:.2f}",
            ]
            for column, value in enumerate(values):
                self.setItem(row, column, QTableWidgetItem(value))
        self.setSortingEnabled(True)

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


def _format_px(value: float) -> str:
    return f"{value:.1f} px"
