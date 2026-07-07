from PySide6.QtWidgets import QHeaderView, QTableWidget


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
