import sys
import os
import cv2
import math
from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from rmn import RMN  # Residual Masking Network for sentiment analysis

# ---------------------------
# Revised Threat Calculation Function
# ---------------------------
def compute_threat_level_sentiment(results, scaling, bias, smoothing, prev_threat=None):
    """
    Computes a threat level based on sentiment analysis results.

    Key Adjustments:
      • Lower face factor (face_count * 0.2) to avoid inflating threat for multiple happy faces.
      • Adjust thresholds for Low/Medium/High/Critical so borderline values are more logical.

    Returns:
      (threat_level, category, breakdown)
    """
    # Define weights for the 7 sentiment classes (happy is small to reduce threat when happy dominates)
    weights = {
        'angry': 1.5,
        'disgust': 1.2,
        'fear': 1.3,
        'happy': 0.1,   # Lower weight for happy
        'sad': 0.8,
        'surprise': 0.7,
        'neutral': 0.5
    }

    contributions = {emo: 0.0 for emo in weights}
    total_contrib = 0.0
    negative_sum = 0.0  # For synergy bonus (angry, disgust, fear, sad)
    negative_emotions = ['angry', 'disgust', 'fear', 'sad']

    # Sum up contributions from each detected face
    for face in results:
        for emo_dict in face.get('proba_list', []):
            for emo, proba in emo_dict.items():
                contr = proba * weights.get(emo, 0)
                contributions[emo] += contr
                total_contrib += contr
                if emo in negative_emotions:
                    negative_sum += contr

    # Factor in the number of faces (lower factor than before)
    face_count = len(results)
    face_factor = face_count * 0.2  # Reduced from 0.5

    # Synergy bonus if multiple faces have negative emotions
    synergy = 0.0
    if face_count > 1 and negative_sum > 0:
        synergy = 0.5 * negative_sum

    base_threat = total_contrib + face_factor + synergy

    # Compute adjusted threat
    adjusted = scaling * base_threat - bias
    try:
        threat_level = 1 / (1 + math.exp(-adjusted))
    except OverflowError:
        threat_level = 0 if adjusted < 0 else 1

    # Temporal smoothing
    if prev_threat is not None:
        threat_level = smoothing * prev_threat + (1 - smoothing) * threat_level

    # New thresholds for more intuitive categorization
    if threat_level < 0.40:
        category = "Low"
    elif threat_level < 0.60:
        category = "Medium"
    elif threat_level < 0.80:
        category = "High"
    else:
        category = "Critical"

    # Calculate per-frame percentages for this call
    percentages = {}
    if total_contrib > 0:
        for emo, val in contributions.items():
            percentages[emo] = (val / total_contrib) * 100
    else:
        for emo in contributions:
            percentages[emo] = 0.0

    breakdown = {
        'absolute': contributions,
        'percentages': percentages,
        'face_count': face_count,
        'synergy_bonus': synergy
    }

    return threat_level, category, breakdown

# ---------------------------
# Plot Canvas for Real-time Plot
# ---------------------------
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Threat Level Over Time", fontsize=12)
        self.ax.set_xlabel("Frame Number", fontsize=10)
        self.ax.set_ylabel("Threat Level (0-1)", fontsize=10)
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], marker='o', linestyle='-')
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()

    def update_plot(self, frames, threat_levels):
        if frames:
            self.line.set_data(frames, threat_levels)
            self.ax.set_xlim(0, max(10, max(frames)))
        self.ax.set_ylim(0, 1)
        self.draw()

# ---------------------------
# Main Application Window
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentiment & Threat Assessment")
        self.resize(1000, 800)

        # Initialize RMN sentiment model
        self.model = RMN()

        # Threat calculation parameters
        self.scaling = 1.0
        self.bias = 1.0
        self.smoothing = 0.9
        self.prev_threat = None

        # Video capture and timer for video/webcam feed
        self.video_capture = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False

        # History for threat plot
        self.threat_history = []
        self.frame_history = []
        self.current_frame_number = 0

        # Accumulated contributions over frames
        self.total_contributions = {emo: 0.0 for emo in ['angry','disgust','fear','happy','sad','surprise','neutral']}
        self.total_frames = 0

        # Mode: "image", "video", or "webcam"
        self.mode = None
        self.loaded_filename = None

        self.init_ui()

    def init_ui(self):
        # Overall style
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border-radius: 5px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #005F9E;
            }
            QLabel {
                color: #333;
            }
            QTabWidget::pane {
                border: 1px solid #C2C7CB;
            }
            QTabBar::tab {
                background: #E0E0E0;
                padding: 10px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                border-bottom: 2px solid #007ACC;
            }
        """)

        # Create tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Sentiment Analysis (live feed)
        self.tab_analysis = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_analysis, "Sentiment Analysis")

        # Tab 2: Threat Report
        self.tab_report = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_report, "Threat Report")

        self.init_analysis_tab()
        self.init_report_tab()

    def init_analysis_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.tab_analysis.setLayout(layout)

        # Top control buttons
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.btn_load_file = QtWidgets.QPushButton("Load File")
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_open_webcam = QtWidgets.QPushButton("Open Webcam")
        self.btn_open_webcam.clicked.connect(self.open_webcam)
        self.btn_stop_feed = QtWidgets.QPushButton("Stop Feed")
        self.btn_stop_feed.clicked.connect(self.stop_feed)
        control_layout.addWidget(self.btn_load_file)
        control_layout.addWidget(self.btn_open_webcam)
        control_layout.addWidget(self.btn_stop_feed)
        layout.addLayout(control_layout)

        # File path label
        self.lbl_filepath = QtWidgets.QLabel("No file loaded")
        self.lbl_filepath.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_filepath)

        # Display area for video/image feed
        self.display_label = QtWidgets.QLabel()
        self.display_label.setFixedSize(640, 480)
        self.display_label.setStyleSheet("background-color: black;")
        self.display_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.display_label)

        # Threat level display
        self.lbl_threat = QtWidgets.QLabel("Threat Level: N/A")
        self.lbl_threat.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_threat)

        # Real-time plot
        self.plot_canvas = PlotCanvas(self, width=5, height=3)
        layout.addWidget(self.plot_canvas)

    def init_report_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.tab_report.setLayout(layout)

        self.lbl_report = QtWidgets.QLabel("Threat report summary will appear here once processing is complete.")
        self.lbl_report.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_report.setWordWrap(True)
        layout.addWidget(self.lbl_report)

        self.btn_export = QtWidgets.QPushButton("Export Report")
        self.btn_export.clicked.connect(self.export_report)
        layout.addWidget(self.btn_export)

    def load_file(self):
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image or Video File", "",
            "All Files (*);;Video Files (*.mp4 *.avi *.mov *.mkv);;Image Files (*.jpg *.jpeg *.png *.bmp *.webp)",
            options=options)
        if filename:
            self.loaded_filename = filename
            self.lbl_filepath.setText(filename)
            ext = os.path.splitext(filename)[1].lower()
            self.reset_accumulation()
            if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                self.mode = "video"
                self.start_video(filename)
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                self.mode = "image"
                self.stop_feed()
                self.process_image(filename)
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported File", "The selected file type is not supported.")

    def open_webcam(self):
        self.mode = "webcam"
        self.lbl_filepath.setText("Webcam Feed")
        self.reset_accumulation()
        self.start_video(0)

    def stop_feed(self):
        self.is_playing = False
        self.timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = None
        self.update_summary_report()  # Final summary when feed stops

    def reset_accumulation(self):
        self.threat_history.clear()
        self.frame_history.clear()
        self.current_frame_number = 0
        self.prev_threat = None
        self.total_frames = 0
        for emo in self.total_contributions:
            self.total_contributions[emo] = 0.0

    def process_image(self, filename):
        frame = cv2.imread(filename)
        if frame is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Unable to open image file.")
            return
        results = self.model.detect_emotion_for_single_frame(frame)
        threat, category, breakdown = compute_threat_level_sentiment(results, self.scaling, self.bias, self.smoothing)
        self.lbl_threat.setText(f"Threat Level: {threat:.2f} ({category})")

        # Update accumulation for a single image
        for emo, val in breakdown['absolute'].items():
            self.total_contributions[emo] += val
        self.total_frames += 1
        self.threat_history.append(threat)
        self.frame_history.append(1)

        annotated_frame = self.model.draw(frame, results)
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = 3 * w
        q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.display_label.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(self.display_label.size(), QtCore.Qt.KeepAspectRatio))

        # Immediately update the report for an image
        self.update_report_image(threat, category, breakdown)

    def start_video(self, source):
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            QtWidgets.QMessageBox.warning(self, "Error", "Unable to open video source.")
            return
        self.is_playing = True
        self.timer.start(30)

    def update_frame(self):
        if self.video_capture is None:
            return
        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_feed()
            self.tabs.setCurrentWidget(self.tab_report)
            return
        self.current_frame_number += 1

        # Run inference
        results = self.model.detect_emotion_for_single_frame(frame)
        threat, category, breakdown = compute_threat_level_sentiment(
            results, self.scaling, self.bias, self.smoothing, self.prev_threat
        )
        self.prev_threat = threat

        # Accumulate for summary
        for emo, val in breakdown['absolute'].items():
            self.total_contributions[emo] += val
        self.total_frames += 1

        self.threat_history.append(threat)
        self.frame_history.append(self.current_frame_number)

        # Annotate frame
        annotated_frame = self.model.draw(frame, results)
        overlay_text = f"Threat: {threat:.2f} ({category})"
        cv2.putText(annotated_frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.display_label.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(self.display_label.size(), QtCore.Qt.KeepAspectRatio))

        # Update real-time plot
        self.plot_canvas.update_plot(self.frame_history, self.threat_history)

        # Real-time summary update for webcam
        if self.mode == "webcam":
            self.update_summary_report()

    def update_summary_report(self):
        """
        Provides a comprehensive summary after the feed stops or in real time for webcam.
        Includes average threat, peak threat, and descending emotion contributions.
        """
        if self.total_frames == 0:
            summary_text = "No threat data available."
        else:
            avg_threat = sum(self.threat_history) / self.total_frames
            peak_threat = max(self.threat_history)
            peak_frame = self.frame_history[self.threat_history.index(peak_threat)]

            # Determine category based on avg_threat
            if avg_threat < 0.40:
                category = "Low"
            elif avg_threat < 0.60:
                category = "Medium"
            elif avg_threat < 0.80:
                category = "High"
            else:
                category = "Critical"

            # Compute overall percentages
            total_abs = sum(self.total_contributions.values())
            if total_abs > 0:
                overall_percentages = {
                    emo: (val / total_abs) * 100 for emo, val in self.total_contributions.items()
                }
            else:
                overall_percentages = {emo: 0.0 for emo in self.total_contributions}

            # Sort in descending order
            sorted_contrib = sorted(overall_percentages.items(), key=lambda x: x[1], reverse=True)
            contrib_lines = [f"{emo.capitalize()}: {pct:.1f}%" for emo, pct in sorted_contrib]

            summary_text = (f"Threat Report Summary:\n\n"
                            f"Average Threat Level: {avg_threat:.2f} ({category})\n"
                            f"Peak Threat Level: {peak_threat:.2f} (Frame {peak_frame})\n"
                            f"Total Frames Processed: {self.total_frames}\n\n"
                            f"Overall Emotion Contributions (Descending):\n" +
                            "\n".join(contrib_lines))

        self.lbl_report.setText(summary_text)

    def update_report_image(self, threat, category, breakdown):
        """
        For a single image, show immediate breakdown info.
        """
        sorted_breakdown = sorted(breakdown['percentages'].items(), key=lambda x: x[1], reverse=True)
        breakdown_lines = [f"{emo.capitalize()}: {pct:.1f}%" for emo, pct in sorted_breakdown]

        summary_text = (f"Image Processed.\n\n"
                        f"Threat Level: {threat:.2f} ({category})\n"
                        f"Face Count: {breakdown['face_count']}\n"
                        f"Synergy Bonus: {breakdown['synergy_bonus']:.2f}\n\n"
                        f"Emotion Contributions (Descending):\n" + "\n".join(breakdown_lines))

        self.lbl_report.setText(summary_text)

    def export_report(self):
        export_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Export Directory", os.path.expanduser("~")
        )
        if not export_folder:
            return
        summary_text = self.lbl_report.text()
        report_path = os.path.join(export_folder, "threat_report.txt")
        with open(report_path, "w") as f:
            f.write(summary_text)
        QtWidgets.QMessageBox.information(self, "Export Successful", f"Threat report exported to:\n{report_path}")

    def closeEvent(self, event):
        if self.video_capture is not None:
            self.video_capture.release()
        event.accept()

# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
