import json
import sys
import threading
import time
import urllib.request
from pathlib import Path
from queue import Empty, SimpleQueue

import numpy as np

ROOT = Path(__file__).resolve().parent

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from activity_features import (
    LABEL_NAMES,
    clean_signal,
    extract_from_array,
    load_classifier_artifacts,
    load_csv_as_array,
    merged_label_ranges,
    predict_windows,
    ranges_to_dataframe,
)

WINDOW_SIZE = 50


class InfoHintLabel(QLabel):
    def __init__(self, tooltip, parent):
        super().__init__("ℹ", parent)
        self._tooltip = tooltip
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(20, 20)
        self.setAutoFillBackground(False)
        self.setStyleSheet("QLabel { color: #ffffff; background: transparent; border: none; font-size: 13px; }")

    def enterEvent(self, event):
        QToolTip.showText(event.globalPosition().toPoint(), self._tooltip, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


def info_hint(parent, tooltip):
    return InfoHintLabel(tooltip, parent)


def segment_signal(data, window_size):
    segments = []
    for i in range(0, len(data) - window_size, window_size):
        segments.append(data[i : i + window_size])
    return np.array(segments)


def phyphox_polling_loop(host, scaler, classifier, window_size, stop_event, out_q):
    time_buf = "acc_time"
    axes = ("accX", "accY", "accZ")
    cols = (time_buf, "accX", "accY", "accZ")
    last_t = 0.0
    row_buffer = []
    recent_preds = []
    buf_cap = window_size * 10
    base_url = "http://" + host.strip().lstrip("/")

    while not stop_event.is_set():
        query = f"{time_buf}={last_t:.4f}&" + "&".join(f"{c}={last_t:.4f}|{time_buf}" for c in cols)
        url = f"{base_url}/get?{query}"
        try:
            with urllib.request.urlopen(url) as r:
                data = json.loads(r.read().decode())
        except Exception as e:
            out_q.put(("error", str(e) or "failed to fetch data"))
            time.sleep(0.1)
            continue

        buf = data.get("buffer", {})
        ts = buf.get(time_buf, {}).get("buffer", [])
        axis_cols = [buf.get(a, {}).get("buffer", []) for a in axes]
        for row in zip(ts, *axis_cols):
            try:
                t, x, y, z = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            except:
                continue
            row_buffer.append([t, x, y, z])
        if ts:
            try:
                last_t = float(ts[-1])
            except:
                pass

        if len(row_buffer) > buf_cap:
            del row_buffer[:-buf_cap]

        while len(row_buffer) >= window_size:
            chunk = np.asarray(row_buffer[:window_size], dtype=float)
            del row_buffer[:window_size]
            feats = extract_from_array(chunk, clean_first=True)
            X = np.asarray(feats, dtype=float).reshape(1, -1)
            X = scaler.transform(X)
            X = np.nan_to_num(X, nan=0.0)
            pred = int(classifier.predict(X)[0])
            recent_preds.append(pred)
            recent_preds[:] = recent_preds[-3:]


            if len(recent_preds) >= 3:
                avg = sum(recent_preds[-3:]) / 3
                out_q.put(("activity", LABEL_NAMES[1] if avg >= 0.5 else LABEL_NAMES[0]))
            else:
                out_q.put(("activity", "Collecting…"))

        time.sleep(0.1)


def main():
    app = QApplication(sys.argv)

    scaler, classifier = load_classifier_artifacts(ROOT)

    inp = QLineEdit()
    out = QLineEdit()
    w = QWidget()
    w.setWindowTitle("Activity labeler: walking / jumping")
    w.resize(520, 340)
    w.live_thread = None
    w.live_stop_event = None
    w.live_queue = None

    blurb = QLabel(
        "Runs the trained classifier on your accelerometer recording: the signal is cleaned like in training, "
        "split into windows, and each window is scored as walking or jumping. Neighboring windows with the "
        "same label are merged into ranges with row indices and timestamps for the summary file."
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

        starts, labels = predict_windows(data, scaler, classifier, WINDOW_SIZE, segment_signal)
        df = ranges_to_dataframe(merged_label_ranges(data, starts, labels, WINDOW_SIZE))
        df.to_csv(out_path, index=False)

        QMessageBox.information(w, "Done", f"Saved {len(df)} range(s).")

    ip_phyphox = QLineEdit()
    ip_phyphox.setStyleSheet("QLineEdit { background: #2a2a2a; color: #ffffff; padding: 4px; }")

    lbl_live_activity = QLabel("----")
    lbl_live_activity.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
    lbl_live_err = QLabel("")
    lbl_live_err.setStyleSheet("color: #ff9999; font-size: 11px;")
    lbl_live_err.setWordWrap(True)

    live_timer = QTimer(w)
    live_timer.setInterval(50)

    def poll_live_queue():
        q = w.live_queue
        if q is None:
            return
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "activity":
                    lbl_live_err.clear()
                    lbl_live_activity.setText(payload)
                elif kind == "error":
                    lbl_live_err.setText(payload[:200])
        except Empty:
            pass

    live_timer.timeout.connect(poll_live_queue)

    def toggle_live():
        if w.live_thread is not None and w.live_thread.is_alive():
            w.live_stop_event.set()
            w.live_thread.join(5.0)
            w.live_thread = None
            w.live_stop_event = None
            w.live_queue = None
            live_timer.stop()
            btn_live.setText("Go live")
            ip_phyphox.setEnabled(True)
            lbl_live_activity.setText("----")
            lbl_live_err.clear()
            return

        host = ip_phyphox.text().strip()
        if not host:
            QMessageBox.warning(w, "Live", "Enter the Phyphox IP address (and port if needed, e.g. 192.168.1.5:8080).")
            return

        w.live_queue = SimpleQueue()
        w.live_stop_event = threading.Event()
        w.live_thread = threading.Thread(target=phyphox_polling_loop, args=(host, scaler, classifier, WINDOW_SIZE, w.live_stop_event, w.live_queue), daemon=True)
        btn_live.setText("Stop live")
        ip_phyphox.setEnabled(False)
        lbl_live_activity.setText("Collecting…")
        lbl_live_err.clear()
        live_timer.start()
        w.live_thread.start()

    binp = QPushButton("…")
    binp.clicked.connect(pick_input)
    bout = QPushButton("…")
    bout.clicked.connect(pick_output)
    brun = QPushButton("Label and save")
    brun.clicked.connect(run_labeling)

    btn_live = QPushButton("Go live")
    btn_live.clicked.connect(toggle_live)

    row_live = QHBoxLayout()
    row_live.addWidget(QLabel("Phyphox IP:"))
    row_live.addWidget(ip_phyphox, stretch=1)
    row_live.addWidget(btn_live)

    row_live_result = QHBoxLayout()
    row_live_result.addWidget(QLabel("Live activity:"))
    row_live_result.addWidget(lbl_live_activity, stretch=1)

    row_in = QHBoxLayout()
    row_in.addWidget(QLabel("Input:"))
    row_in.addWidget(info_hint(w, "Upload or select a CSV file of accelerometer data (time column plus x, y, z)."))
    row_in.addWidget(inp, stretch=1)
    row_in.addWidget(binp)

    row_out = QHBoxLayout()
    row_out.addWidget(QLabel("Output:"))
    row_out.addWidget(info_hint(w, "Choose where to save the labeled activity summary as a CSV (ranges, times, walking/jumping)."))
    row_out.addWidget(out, stretch=1)
    row_out.addWidget(bout)

    lay = QVBoxLayout(w)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(10)
    lay.addWidget(blurb)
    lay.addLayout(row_in)
    lay.addLayout(row_out)
    lay.addWidget(brun)
    lay.addSpacing(8)
    lay.addLayout(row_live)
    lay.addLayout(row_live_result)
    lay.addWidget(lbl_live_err)

    w.show()
    sys.exit(app.exec())

main()