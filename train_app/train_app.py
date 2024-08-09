import sys
import json
import os

import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QVBoxLayout, QWidget, QCheckBox,
    QProgressBar
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from trainer import train_model  # Assuming the trainer module is available

CONFIG_FILE = 'config.json'
SETTINGS_FILE = 'settings.json'

class TrainingThread(QThread):
    update_image = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    training_finished = pyqtSignal()

    def __init__(self, input_folder, output_folder, var, size, epochs, batch_size, learning_rate, n_cycles, use_folder, opt_name):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.var = var
        self.size = size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_cycles = n_cycles
        self.use_folder = use_folder
        self.optimizer_name = opt_name
        self.should_run = True

    def run(self):
        for epoch, image_path in train_model(
                self.input_folder, self.output_folder, self.var, self.size,
                self.epochs, self.batch_size, self.learning_rate, self.n_cycles, self.use_folder, self.optimizer_name):
            if self.should_run:
                self.update_progress.emit(epoch + 1)
                self.update_image.emit(image_path)
            else:
                break
        self.training_finished.emit()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.settings = self.load_settings()
        self.widgets = {}
        self.initUI()
        self.training_thread = None


    def load_config(self):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_settings(self):
        settings = {}
        for widget_id, widget in self.widgets.items():
            if isinstance(widget, QLineEdit):
                settings[widget_id] = widget.text()
            elif isinstance(widget, QComboBox):
                settings[widget_id] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                settings[widget_id] = widget.isChecked()
            elif isinstance(widget, QPushButton) and widget_id.endswith("_folder"):
                settings[f"{widget_id}_path"] = widget.text()  # Correctly save folder paths
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f)

    def initUI(self):
        self.setWindowTitle(self.config['window']['title'])
        self.resize(*self.config['window']['size'])

        layout = QVBoxLayout()

        # Dynamically create widgets based on config
        for widget_config in self.config['widgets']:
            widget_id = widget_config['id']
            if widget_config['type'] == 'checkbox':
                widget = QCheckBox(widget_config['label'])
                widget.setChecked(self.settings.get(widget_id, False))
                layout.addWidget(widget)
            elif widget_config['type'] == 'combobox':
                widget = QComboBox(self)
                widget.addItems(widget_config['options'])
                widget.setCurrentText(self.settings.get(widget_id, widget_config.get('default', '')))
                layout.addWidget(QLabel(widget_config['label']))
                layout.addWidget(widget)
            elif widget_config['type'] == 'lineedit':
                widget = QLineEdit(self)
                widget.setText(self.settings.get(widget_id, widget_config.get('default', '')))
                layout.addWidget(QLabel(widget_config['label']))
                layout.addWidget(widget)
            elif widget_config['type'] == 'folder_select':
                widget = QPushButton("Select Folder")
                widget.clicked.connect(lambda checked, id=widget_id: self.select_folder(id))
                widget.setText(self.settings.get(f"{widget_id}_path", "Select Folder"))  # Load folder paths
                layout.addWidget(QLabel(widget_config['label']))
                layout.addWidget(widget)

            self.widgets[widget_id] = widget

        # Start training button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.toggle_training)  # Connect to the new toggle function
        layout.addWidget(self.start_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Image preview
        self.image_preview = QLabel(self)
        self.image_preview.setFixedSize(512, 512)
        self.image_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_preview)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Main widget
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def select_folder(self, widget_id):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.widgets[widget_id].setText(folder)
            self.settings[f"{widget_id}_path"] = folder  # Update settings immediately
    def toggle_training(self):
        if self.training_thread is None or not self.training_thread.isRunning():
            self.start_training()
        else:

            self.stop_training()
    def start_training(self):
        input_folder = self.widgets['input_folder'].text() if self.widgets['use_folder'].isChecked() else ""
        output_folder = self.widgets['output_folder'].text() if 'output_folder' in self.widgets else ""
        var = self.widgets['channels_variant'].currentText()
        size = 512
        epochs = int(self.widgets['epochs'].text())
        batch_size = int(self.widgets['batch_size'].text())
        learning_rate = float(self.widgets['learning_rate'].text())
        n_cycles = int(self.widgets['cycles'].text())
        use_folder = self.widgets['use_folder'].isChecked()
        optimizer_name = self.widgets['optimizer_name'].currentText()

        self.status_label.setText("Training started...")
        self.progress_bar.setMaximum(epochs)

        self.training_thread = TrainingThread(input_folder, output_folder, var, size, epochs, batch_size, learning_rate, n_cycles, use_folder, optimizer_name)
        self.training_thread.update_progress.connect(self.update_progress)
        self.training_thread.update_image.connect(self.show_image_preview)
        self.training_thread.training_finished.connect(self.training_finished)
        self.training_thread.start()

        self.start_button.setText("Stop Training")  # Change button text to "Stop Training"

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():

            self.training_thread.should_run = False  # Terminate the training thread
            torch.cuda.empty_cache()
            # self.training_thread.wait()
            self.training_finished()  # Call the finish function to reset the UI

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_image_preview(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(self.image_preview.size(), Qt.KeepAspectRatio)
        self.image_preview.setPixmap(pixmap)

    @pyqtSlot()  # Ensure this method is a slot
    def training_finished(self):
        self.status_label.setText("Training finished.")
        self.start_button.setText("Start Training")  # Reset button text to "Start Training"
        # self.training_thread = None  # Clear the thread reference to allow restart


    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
