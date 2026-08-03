from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class ImageViewer(QLabel):
    """QLabel image viewer with ROI drawing and image-coordinate click events."""

    roi_changed = Signal(object)
    image_clicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__("未导入图像")
        self._source_pixmap: QPixmap | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(520, 360)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #f5f7fa; border: 1px solid #c8d0dc; color: #6b7280; }"
        )

    def set_image(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._update_scaled_pixmap()

    def clear_image(self) -> None:
        self._source_pixmap = None
        self.clear_roi()
        self.clear()
        self.setText("未导入图像")

    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def clear_roi(self) -> None:
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self.roi_changed.emit(None)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() == Qt.MouseButton.LeftButton and self._source_pixmap is not None:
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            end = event.position().toPoint()
            drag_rect = QRect(self._drag_start, end).normalized()
            self._drag_start = None
            self._drag_current = None
            if drag_rect.width() < 5 and drag_rect.height() < 5:
                image_point = self._widget_to_image(end)
                if image_point is not None:
                    self.image_clicked.emit(image_point[0], image_point[1])
            else:
                self._roi = self._widget_rect_to_image_roi(drag_rect)
                self.roi_changed.emit(self._roi)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = None
        if self._drag_start is not None and self._drag_current is not None:
            rect = QRect(self._drag_start, self._drag_current).normalized()
        elif self._roi is not None:
            rect = self._image_roi_to_widget_rect(self._roi)
        if rect is not None and rect.isValid():
            painter.fillRect(rect, QColor(37, 99, 235, 45))
            painter.setPen(QPen(QColor(37, 99, 235), 2, Qt.PenStyle.SolidLine))
            painter.drawRect(rect)

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def _pixmap_rect(self) -> QRect | None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        x = (self.width() - pixmap.width()) // 2
        y = (self.height() - pixmap.height()) // 2
        return QRect(x, y, pixmap.width(), pixmap.height())

    def _widget_to_image(self, point: QPoint) -> tuple[int, int] | None:
        if self._source_pixmap is None:
            return None
        rect = self._pixmap_rect()
        if rect is None or not rect.contains(point):
            return None
        x_ratio = self._source_pixmap.width() / rect.width()
        y_ratio = self._source_pixmap.height() / rect.height()
        image_x = int((point.x() - rect.x()) * x_ratio)
        image_y = int((point.y() - rect.y()) * y_ratio)
        image_x = max(0, min(image_x, self._source_pixmap.width() - 1))
        image_y = max(0, min(image_y, self._source_pixmap.height() - 1))
        return image_x, image_y

    def _widget_rect_to_image_roi(self, rect: QRect) -> tuple[int, int, int, int] | None:
        pixmap_rect = self._pixmap_rect()
        if pixmap_rect is None:
            return None
        clipped = rect.intersected(pixmap_rect)
        if clipped.width() <= 0 or clipped.height() <= 0:
            return None
        top_left = self._widget_to_image(clipped.topLeft())
        bottom_right = self._widget_to_image(clipped.bottomRight())
        if top_left is None or bottom_right is None:
            return None
        x1, y1 = top_left
        x2, y2 = bottom_right
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1) + 1
        height = abs(y2 - y1) + 1
        return x, y, width, height

    def _image_roi_to_widget_rect(self, roi: tuple[int, int, int, int]) -> QRect | None:
        if self._source_pixmap is None:
            return None
        pixmap_rect = self._pixmap_rect()
        if pixmap_rect is None:
            return None
        x, y, width, height = roi
        x_ratio = pixmap_rect.width() / self._source_pixmap.width()
        y_ratio = pixmap_rect.height() / self._source_pixmap.height()
        return QRect(
            pixmap_rect.x() + int(x * x_ratio),
            pixmap_rect.y() + int(y * y_ratio),
            max(1, int(width * x_ratio)),
            max(1, int(height * y_ratio)),
        )
