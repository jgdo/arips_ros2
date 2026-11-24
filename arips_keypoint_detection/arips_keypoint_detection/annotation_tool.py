"""Keypoint annotation tool for RGB-D datasets created by extract_rgbd_dataset.

usage:
  ros2 run arips_keypoint_detection keypoint_annotator [--dataset DIR]

Controls:
  left mouse button     set keypoint 0 (on RGB or depth image)
  right mouse button    set keypoint 1
  left / right arrow    previous / next image (sorted by name)
  ctrl + left / right   previous / next image that has no annotation for the tag yet
  space                 save the current state (also with no keypoints = none visible) and go to the
                        next unannotated image
  "Clear" button        remove all keypoints of the current image

The annotation is saved as <rgb file name>_<tag>.json next to the image:
- when pressing space (always, an empty keypoint set marks the image as "nothing visible"),
- when leaving the image with the arrow keys or closing the window, but only if it was changed
  (click or Clear). Just browsing with the arrow keys never writes anything.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from python_qt_binding.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from python_qt_binding.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from python_qt_binding.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import dataset_io as dio
from .learning.viz import colorize_depth

KEYPOINT_COLORS = [QColor(255, 40, 40), QColor(40, 220, 40), QColor(60, 120, 255), QColor(255, 200, 0),
                   QColor(255, 0, 255), QColor(0, 230, 230), QColor(255, 128, 0), QColor(160, 80, 255)]
MOUSE_KEYPOINTS = {Qt.MouseButton.LeftButton: 0, Qt.MouseButton.RightButton: 1}


def keypoint_color(index: int) -> QColor:
    return KEYPOINT_COLORS[index % len(KEYPOINT_COLORS)]


def rgb_array_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------------------------------------
# setup dialog
# ---------------------------------------------------------------------------------------------

class SetupDialog(QDialog):
    """Choose an existing annotation config of the dataset or create a new one."""

    def __init__(self, dataset_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Keypoint annotation - {dataset_dir}')
        self.dataset_dir = dataset_dir
        self.result_tag = None
        self.result_names = None
        self._configs = dio.list_configs(dataset_dir)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f'Dataset: {dataset_dir}  ({len(dio.list_rgb_images(dataset_dir))} images)'))

        # continue existing
        existing = QGroupBox('Continue existing annotation')
        ex_layout = QVBoxLayout(existing)
        self._list = QListWidget()
        for tag, names in self._configs:
            item = QListWidgetItem(f'{tag}    [{", ".join(names)}]')
            item.setData(Qt.ItemDataRole.UserRole, tag)
            self._list.addItem(item)
        self._list.itemDoubleClicked.connect(lambda _item: self._continue())
        ex_layout.addWidget(self._list)
        continue_button = QPushButton('Continue selected')
        continue_button.clicked.connect(self._continue)
        ex_layout.addWidget(continue_button)
        if not self._configs:
            self._list.addItem('(no annotation configs in this dataset yet)')
            self._list.setEnabled(False)
            continue_button.setEnabled(False)
        else:
            self._list.setCurrentRow(0)
        layout.addWidget(existing)

        # new annotation
        new = QGroupBox('Start new annotation')
        form = QFormLayout(new)
        self._tag_edit = QLineEdit()
        self._tag_edit.setPlaceholderText('e.g. door_handle  (letters, digits, _ and -)')
        form.addRow('Tag:', self._tag_edit)
        self._count = QSpinBox()
        self._count.setRange(1, 64)
        self._count.setValue(2)
        self._count.valueChanged.connect(self._resize_names)
        form.addRow('Number of keypoints:', self._count)
        self._names = QTableWidget(0, 1)
        self._names.setHorizontalHeaderLabels(['Keypoint name'])
        self._names.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._names.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        form.addRow(self._names)
        self._resize_names(self._count.value())
        start_button = QPushButton('Start new annotation')
        start_button.clicked.connect(self._start_new)
        form.addRow(start_button)
        layout.addWidget(new)

        self.resize(520, 560)

    def _resize_names(self, count: int):
        old = self._names.rowCount()
        self._names.setRowCount(count)
        self._names.setVerticalHeaderLabels([str(i) for i in range(count)])
        for i in range(old, count):
            self._names.setItem(i, 0, QTableWidgetItem(f'kp{i}'))

    def _continue(self):
        item = self._list.currentItem()
        if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
            return
        tag = item.data(Qt.ItemDataRole.UserRole)
        self.result_tag, self.result_names = next((t, n) for t, n in self._configs if t == tag)
        self.accept()

    def _start_new(self):
        tag = self._tag_edit.text().strip()
        if not dio.is_valid_tag(tag):
            QMessageBox.warning(self, 'Invalid tag', 'The tag may only contain letters, digits, _ and -.')
            return
        if any(t == tag for t, _ in self._configs) or dio.config_path(self.dataset_dir, tag).exists():
            QMessageBox.warning(self, 'Tag exists', f'An annotation with tag "{tag}" already exists, '
                                'select it under "Continue existing annotation".')
            return
        self._names.setCurrentItem(None)  # commit a running cell edit
        names = []
        for i in range(self._names.rowCount()):
            item = self._names.item(i, 0)
            name = item.text().strip() if item is not None else ''
            names.append(name or f'kp{i}')
        if len(set(names)) != len(names):
            QMessageBox.warning(self, 'Duplicate names', 'Keypoint names must be unique.')
            return
        dio.save_config(self.dataset_dir, tag, names)
        self.result_tag, self.result_names = tag, names
        self.accept()


# ---------------------------------------------------------------------------------------------
# image view
# ---------------------------------------------------------------------------------------------

class ImageView(QWidget):
    """Shows one image scaled to fit, draws keypoints (given in reference/RGB pixel coordinates)."""

    clicked = Signal(object, float, float)   # mouse button, x, y in reference coordinates
    hovered = Signal(float, float)           # x, y in this image's own pixel coordinates

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.pixmap = None
        self.placeholder = 'no image'
        self.ref_size = (1, 1)               # (w, h) of the coordinate frame of keypoints
        self.keypoints: dict[int, tuple[float, float]] = {}
        self.names: list[str] = []
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_image(self, qimage: QImage | None, ref_size, placeholder: str = 'no image'):
        self.pixmap = QPixmap.fromImage(qimage) if qimage is not None else None
        self.ref_size = ref_size
        self.placeholder = placeholder
        self.update()

    def set_keypoints(self, keypoints, names):
        self.keypoints = dict(keypoints)
        self.names = names
        self.update()

    def _target_rect(self) -> QRectF | None:
        if self.pixmap is None:
            return None
        pw, ph = self.pixmap.width(), self.pixmap.height()
        scale = min(self.width() / pw, self.height() / ph)
        w, h = pw * scale, ph * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _widget_to_ref(self, pos) -> tuple[float, float] | None:
        rect = self._target_rect()
        if rect is None or not rect.contains(pos):
            return None
        u = (pos.x() - rect.x()) / rect.width()
        v = (pos.y() - rect.y()) / rect.height()
        return u * self.ref_size[0], v * self.ref_size[1]

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        rect = self._target_rect()
        if rect is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        for idx, (x, y) in sorted(self.keypoints.items()):
            p = QPointF(rect.x() + x / self.ref_size[0] * rect.width(),
                        rect.y() + y / self.ref_size[1] * rect.height())
            color = keypoint_color(idx)
            painter.setPen(QPen(QColor(0, 0, 0), 4))
            painter.drawEllipse(p, 7, 7)
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(p, 7, 7)
            painter.drawLine(p + QPointF(-3, 0), p + QPointF(3, 0))
            painter.drawLine(p + QPointF(0, -3), p + QPointF(0, 3))
            label = self.names[idx] if idx < len(self.names) else str(idx)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(p + QPointF(11, -7), label)
            painter.setPen(color)
            painter.drawText(p + QPointF(10, -8), label)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QPointF(rect.x() + 6, rect.y() + 16), self.title)

    def mousePressEvent(self, event):
        ref = self._widget_to_ref(event.position() if hasattr(event, 'position') else event.pos())
        if ref is not None:
            self.clicked.emit(event.button(), ref[0], ref[1])

    def mouseMoveEvent(self, event):
        ref = self._widget_to_ref(event.position() if hasattr(event, 'position') else event.pos())
        if ref is not None and self.pixmap is not None:
            self.hovered.emit(ref[0] / self.ref_size[0] * self.pixmap.width(),
                              ref[1] / self.ref_size[1] * self.pixmap.height())


# ---------------------------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------------------------

class AnnotationWindow(QMainWindow):

    def __init__(self, dataset_dir: Path, tag: str, names: list[str]):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.tag = tag
        self.names = names
        self.images = dio.list_rgb_images(dataset_dir)
        self.depth_index = dio.DepthIndex(dataset_dir)
        self.annotated = {p for p in self.images if dio.has_annotation(p, tag)}
        self.index = -1
        self.keypoints: dict[int, tuple[float, float]] = {}
        self.dirty = False   # keypoints changed since loading the image
        self.depth_m = None

        self.setWindowTitle(f'Keypoint annotation [{tag}] - {dataset_dir}')
        central = QWidget()
        layout = QVBoxLayout(central)

        top = QHBoxLayout()
        clear_button = QPushButton('Clear')
        clear_button.setToolTip('Remove all keypoints of this image')
        clear_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear_button.clicked.connect(self.clear_keypoints)
        top.addWidget(clear_button)
        legend = '   '.join(
            f'<span style="color:{keypoint_color(i).name()}">&#9679;</span> {b}: {names[i]}'
            for b, i in (('left click', 0), ('right click', 1)) if i < len(names))
        top.addWidget(QLabel(legend + '&nbsp;&nbsp;&nbsp;&larr;/&rarr;: prev/next, '
                             'Ctrl+&larr;/&rarr;: prev/next unannotated, Space: save + next unannotated'))
        top.addStretch(1)
        self.message = QLabel()
        self.message.setStyleSheet('color: #d03030; font-weight: bold;')
        top.addWidget(self.message)
        layout.addLayout(top)

        views = QHBoxLayout()
        self.rgb_view = ImageView('RGB')
        self.depth_view = ImageView('Depth')
        for view in (self.rgb_view, self.depth_view):
            view.clicked.connect(self.on_click)
            views.addWidget(view, 1)
        self.rgb_view.hovered.connect(lambda x, y: self.on_hover(x, y, rgb=True))
        self.depth_view.hovered.connect(lambda x, y: self.on_hover(x, y, rgb=False))
        layout.addLayout(views, 1)

        bottom = QHBoxLayout()
        self.state_label = QLabel()
        bottom.addWidget(self.state_label)
        self.info = QLabel()
        bottom.addWidget(self.info, 1)
        layout.addLayout(bottom)
        self.setCentralWidget(central)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self.message.clear)

        self.resize(1400, 620)
        if self.images:
            first = next((i for i, p in enumerate(self.images) if p not in self.annotated), 0)
            self.load(first)
        else:
            self.show_message('no RGB images in dataset')

    # --- data -------------------------------------------------------------------------------

    @property
    def current(self) -> Path | None:
        return self.images[self.index] if 0 <= self.index < len(self.images) else None

    def load(self, index: int):
        self.index = index
        path = self.current
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            self.rgb_view.set_image(None, (1, 1), f'could not read {path.name}')
            ref = (1, 1)
        else:
            ref = (bgr.shape[1], bgr.shape[0])
            self.rgb_view.set_image(rgb_array_to_qimage(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ref)
        depth_path = self.depth_index.find(path)
        self.depth_m = None
        if depth_path is not None:
            try:
                self.depth_m = dio.load_depth_m(depth_path)
            except OSError:
                pass
        if self.depth_m is not None:
            self.depth_view.set_image(rgb_array_to_qimage(colorize_depth(self.depth_m)), ref)
        else:
            self.depth_view.set_image(None, ref, 'no depth image')
        self.keypoints = dio.load_annotation(path, self.tag) or {}
        self.dirty = False
        self.refresh()

    def save_current(self):
        path = self.current
        if path is None:
            return
        try:
            dio.save_annotation(path, self.tag, self.keypoints)
        except OSError as e:
            QMessageBox.critical(self, 'Save failed', f'Could not save annotation of {path.name}:\n{e}')
            return
        self.annotated.add(path)
        self.dirty = False

    def save_if_changed(self):
        if self.dirty:
            self.save_current()

    def refresh(self):
        for view in (self.rgb_view, self.depth_view):
            view.set_keypoints(self.keypoints, self.names)
        path = self.current
        if path is None:
            return
        annotated = path in self.annotated
        state = 'annotated' if annotated else 'not annotated yet'
        if self.dirty:
            state += ' (unsaved changes)'
        self.state_label.setText(state)
        self.state_label.setStyleSheet(
            'QLabel { color: white; font-weight: bold; padding: 2px 8px; border-radius: 4px; '
            f'background-color: {"#3c8d40" if annotated else "#c0392b"}; }}')
        kps = ', '.join(f'{self.names[i] if i < len(self.names) else i}=({x:.0f}, {y:.0f})'
                        for i, (x, y) in sorted(self.keypoints.items())) or 'no keypoints'
        self.info.setText(f'{self.index + 1}/{len(self.images)}   {path.name}   '
                          f'tag: {self.tag}   annotated: {len(self.annotated)}/{len(self.images)}   |   {kps}')

    # --- interaction ------------------------------------------------------------------------

    def on_click(self, button, x: float, y: float):
        idx = MOUSE_KEYPOINTS.get(button)
        if idx is None or idx >= len(self.names) or self.current is None:
            return
        self.keypoints[idx] = (x, y)
        self.dirty = True
        self.refresh()

    def on_hover(self, x: float, y: float, rgb: bool):
        if self.depth_m is None:
            return
        h, w = self.depth_m.shape
        if rgb:
            x, y = x / self.rgb_view.ref_size[0] * w, y / self.rgb_view.ref_size[1] * h
        u, v = min(int(x), w - 1), min(int(y), h - 1)
        d = self.depth_m[v, u]
        self.statusBar().showMessage(f'depth ({u}, {v}): ' + (f'{d:.3f} m' if d > 0 else 'invalid'))

    def clear_keypoints(self):
        self.keypoints = {}
        self.dirty = True
        self.refresh()

    def go_to(self, index: int):
        if self.current is None or index == self.index:
            return
        self.save_if_changed()
        self.load(index)

    def step(self, direction: int):
        if self.current is None:
            return
        target = self.index + direction
        if 0 <= target < len(self.images):
            self.go_to(target)
        else:
            self.show_message('first image' if direction < 0 else 'last image')

    def step_unannotated(self, direction: int):
        if self.current is None:
            return
        rng = range(self.index + 1, len(self.images)) if direction > 0 else range(self.index - 1, -1, -1)
        target = next((i for i in rng if self.images[i] not in self.annotated), None)
        if target is None:
            self.show_message('No unannotated image left' + (' after this one' if direction > 0
                                                             else ' before this one'))
            return
        self.go_to(target)

    def save_and_next(self):
        """Space: always save the current state (empty = nothing visible), then go to the next
        unannotated image."""
        if self.current is None:
            return
        self.save_current()
        self.refresh()
        self.step_unannotated(1)

    def show_message(self, text: str, ms: int = 3000):
        self.message.setText(text)
        self._message_timer.start(ms)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            direction = -1 if key == Qt.Key.Key_Left else 1
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.step_unannotated(direction)
            else:
                self.step(direction)
            return
        if key == Qt.Key.Key_Space:
            self.save_and_next()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.save_if_changed()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, help='dataset folder (asked for if not given)')
    args, qt_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    app = QApplication([sys.argv[0]] + qt_args)
    dataset_dir = args.dataset
    if dataset_dir is None:
        selected = QFileDialog.getExistingDirectory(None, 'Select dataset folder', str(Path.cwd()))
        if not selected:
            return 0
        dataset_dir = Path(selected)
    if not dataset_dir.is_dir():
        QMessageBox.critical(None, 'Error', f'{dataset_dir} is not a folder')
        return 1
    if not dio.list_rgb_images(dataset_dir):
        QMessageBox.critical(None, 'Error', f'No RGB images (*_rgb.jpg/png) found in {dataset_dir}')
        return 1

    setup = SetupDialog(dataset_dir)
    if setup.exec() != QDialog.DialogCode.Accepted:
        return 0
    window = AnnotationWindow(dataset_dir, setup.result_tag, setup.result_names)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
