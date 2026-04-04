import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from activity_features import (
    clean_signal,
    load_classifier_artifacts,
    load_csv_as_array,
    merged_label_ranges,
    predict_windows,
    ranges_to_dataframe,
)

WINDOW_SIZE = 50


def segment_signal(data, window_size):
    segments = []
    for i in range(0, len(data) - window_size, window_size):
        segments.append(data[i : i + window_size])
    return np.array(segments)


def main():
    app = QApplication(sys.argv)

    scaler, clf = load_classifier_artifacts(ROOT)

    inp = QLineEdit()
    out = QLineEdit()
    w = QWidget()
    w.setWindowTitle("Activity labeler — walking / jumping")
    w.resize(520, 210)

    blurb = QLabel(
        "Load accelerometer data from a CSV (time plus x, y, z values). The app classifies the signal as "
        "walking or jumping using the trained model, then writes a CSV of merged time ranges and labels."
    )
    blurb.setWordWrap(True)
    blurb.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    blurb.setStyleSheet("color: #ffffff;")

    def pick_input():
        path, _ = QFileDialog.getOpenFileName(w, "Open CSV", str(ROOT), "CSV (*.csv)")
        if path:
            inp.setText(path)
            if not out.text().strip():
                p = Path(path)
                out.setText(str(p.with_name(p.stem + "_labeled.csv")))

    def pick_output():
        path, _ = QFileDialog.getSaveFileName(w, "Save CSV", str(ROOT), "CSV (*.csv)")
        if path:
            out.setText(path)

    def run_labeling():
        in_path = inp.text().strip()
        out_path = out.text().strip()
        if not in_path or not out_path:
            QMessageBox.warning(w, "Paths", "Set input and output CSV paths.")
            return

        data = clean_signal(load_csv_as_array(in_path))

        starts, labels = predict_windows(data, scaler, clf, WINDOW_SIZE, segment_signal)
        df = ranges_to_dataframe(merged_label_ranges(data, starts, labels, WINDOW_SIZE))
        df.to_csv(out_path, index=False)

        QMessageBox.information(w, "Done", f"Saved {len(df)} range(s).")

    binp = QPushButton("…")
    binp.clicked.connect(pick_input)
    bout = QPushButton("…")
    bout.clicked.connect(pick_output)
    brun = QPushButton("Label and save")
    brun.clicked.connect(run_labeling)

    row_in = QHBoxLayout()
    row_in.addWidget(QLabel("Input:"))
    row_in.addWidget(inp)
    row_in.addWidget(binp)

    row_out = QHBoxLayout()
    row_out.addWidget(QLabel("Output:"))
    row_out.addWidget(out)
    row_out.addWidget(bout)

    lay = QVBoxLayout(w)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(10)
    lay.addWidget(blurb)
    lay.addLayout(row_in)
    lay.addLayout(row_out)
    lay.addWidget(brun)

    w.show()
    sys.exit(app.exec())

main()