from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


RESULT_COLUMNS = [
    "ID",
    "Length",
    "Width",
    "Area",
    "Aspect Ratio",
    "Class",
    "Confidence",
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


def _format_px(value: float) -> str:
    return f"{value:.1f} px"
