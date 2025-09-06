# gui.py

import os
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QPlainTextEdit,
    QCheckBox, QSpinBox, QGroupBox, QMessageBox, QScrollArea, QTabWidget,
    QGridLayout, QComboBox, QDoubleSpinBox, QApplication, QRadioButton
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from utils import ModelDownloader, get_model_path, save_config, load_config
from processing import ProcessingWorker

class MainWindow(QMainWindow):
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Hyper-Efficient Image Captioner")
        self.setGeometry(100, 100, 1200, 900)
        
        self.config = load_config()
        self.processing_worker = None
        self.downloader = None

        self._init_ui()
        self._update_model_status()

        self.log_message.connect(self._append_log)

    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        self.processing_tab = QWidget()
        self.tab_widget.addTab(self.processing_tab, "Processing")
        self.processing_layout = QVBoxLayout(self.processing_tab)

        self._create_directory_selection_group()
        self.processing_layout.addWidget(self.directory_group_box)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Progress: %p%")
        self.processing_layout.addWidget(self.progress_bar)

        self.current_image_label = QLabel("Current Task: Idle")
        self.processing_layout.addWidget(self.current_image_label)

        self._create_control_buttons()
        self.processing_layout.addWidget(self.control_buttons_layout_widget)

        self.log_text_edit = QPlainTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFont(QFont("Consolas", 10))
        self.processing_layout.addWidget(self.log_text_edit)

        self.settings_tab = QWidget()
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.settings_layout = QVBoxLayout(self.settings_tab)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.settings_content_widget = QWidget()
        self.settings_content_layout = QVBoxLayout(self.settings_content_widget)
        self.scroll_area.setWidget(self.settings_content_widget)
        self.settings_layout.addWidget(self.scroll_area)

        self._create_model_download_settings()
        self.settings_content_layout.addWidget(self.model_download_group_box)
        
        self._create_clip_settings()
        self.settings_content_layout.addWidget(self.clip_settings_group_box)

        self._create_processing_settings()
        self.settings_content_layout.addWidget(self.processing_settings_group_box)
        
        self._create_florence2_settings()
        self.settings_content_layout.addWidget(self.florence2_settings_group_box)
        
        self._create_moondream_settings()
        self.settings_content_layout.addWidget(self.moondream_settings_group_box)

        self._create_model_selection_settings()
        self.settings_content_layout.addWidget(self.model_selection_group_box)
        
        self._create_vqa_model_selection_settings()
        self.settings_content_layout.addWidget(self.vqa_model_selection_group_box)

        self.apply_settings_button = QPushButton("Apply Settings")
        self.apply_settings_button.clicked.connect(self._apply_settings)
        self.settings_content_layout.addWidget(self.apply_settings_button)
        self.settings_content_layout.addStretch(1)

        self._set_ui_from_config()

    def _create_directory_selection_group(self):
        self.directory_group_box = QGroupBox("Directories")
        layout = QGridLayout(self.directory_group_box)
        layout.addWidget(QLabel("Model Directory:"), 0, 0)
        self.model_dir_input = QLineEdit(self.config['model_dir'])
        self.model_dir_input.setReadOnly(True)
        layout.addWidget(self.model_dir_input, 0, 1)
        self.browse_model_dir_button = QPushButton("Browse")
        self.browse_model_dir_button.clicked.connect(lambda: self._browse_directory(self.model_dir_input, 'model_dir'))
        layout.addWidget(self.browse_model_dir_button, 0, 2)
        layout.addWidget(QLabel("Image Directory:"), 1, 0)
        self.image_dir_input = QLineEdit(self.config['image_dir'])
        self.image_dir_input.setReadOnly(True)
        layout.addWidget(self.image_dir_input, 1, 1)
        self.browse_image_dir_button = QPushButton("Browse")
        self.browse_image_dir_button.clicked.connect(lambda: self._browse_directory(self.image_dir_input, 'image_dir'))
        layout.addWidget(self.browse_image_dir_button, 1, 2)
        layout.addWidget(QLabel("Output Directory:"), 2, 0)
        self.output_dir_input = QLineEdit(self.config['output_dir'])
        self.output_dir_input.setReadOnly(True)
        layout.addWidget(self.output_dir_input, 2, 1)
        self.browse_output_dir_button = QPushButton("Browse")
        self.browse_output_dir_button.clicked.connect(lambda: self._browse_directory(self.output_dir_input, 'output_dir'))
        layout.addWidget(self.browse_output_dir_button, 2, 2)

    def _create_control_buttons(self):
        self.control_buttons_layout_widget = QWidget()
        layout = QHBoxLayout(self.control_buttons_layout_widget)
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self._start_processing)
        layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        self.download_models_button = QPushButton("Download Models")
        self.download_models_button.clicked.connect(self._start_download)
        layout.addWidget(self.download_models_button)
        self.stop_download_button = QPushButton("Stop Download")
        self.stop_download_button.clicked.connect(self._stop_download)
        self.stop_download_button.setEnabled(False)
        layout.addWidget(self.stop_download_button)

    def _create_model_download_settings(self):
        self.model_download_group_box = QGroupBox("Model Download")
        layout = QVBoxLayout(self.model_download_group_box)
        self.model_status_label = QLabel("Model Status: Checking...")
        layout.addWidget(self.model_status_label)

    def _create_clip_settings(self):
        self.clip_settings_group_box = QGroupBox("CLIP Interrogator Settings")
        layout = QHBoxLayout(self.clip_settings_group_box)
        layout.addWidget(QLabel("CLIP Model (laion2B Finetunes):"))
        self.clip_light_radio = QRadioButton("Light (laion/CLIP-ViT-B-32-laion2B-s34B-b79K)")
        self.clip_heavy_radio = QRadioButton("Heavy (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)")
        if self.config['clip_model_variant'] == 'light':
            self.clip_light_radio.setChecked(True)
        else:
            self.clip_heavy_radio.setChecked(True)
        layout.addWidget(self.clip_light_radio)
        layout.addWidget(self.clip_heavy_radio)

    def _create_processing_settings(self):
        self.processing_settings_group_box = QGroupBox("General Processing Settings")
        layout = QGridLayout(self.processing_settings_group_box)
        row = 0
        self.use_system_image_limits_checkbox = QCheckBox("Use System Image Limits (Recommended)")
        self.use_system_image_limits_checkbox.stateChanged.connect(self._toggle_image_limits_inputs)
        layout.addWidget(self.use_system_image_limits_checkbox, row, 0, 1, 2); row += 1
        layout.addWidget(QLabel("Max Image Width (pixels):"), row, 0)
        self.max_width_spinbox = QSpinBox()
        self.max_width_spinbox.setRange(128, 65535)
        layout.addWidget(self.max_width_spinbox, row, 1); row += 1
        layout.addWidget(QLabel("Max Image Height (pixels):"), row, 0)
        self.max_height_spinbox = QSpinBox()
        self.max_height_spinbox.setRange(128, 65535)
        layout.addWidget(self.max_height_spinbox, row, 1); row += 1
        self.use_cuda_graphs_checkbox = QCheckBox("Enable CUDA Graphs (Requires ALL images to have same resolution)")
        layout.addWidget(self.use_cuda_graphs_checkbox, row, 0, 1, 2); row += 1
        self.concurrent_loading_checkbox = QCheckBox("Enable Concurrent Model Loading (Very High VRAM, Disables Batching)")
        layout.addWidget(self.concurrent_loading_checkbox, row, 0, 1, 2); row += 1
        self.resume_processing_checkbox = QCheckBox("Resume Processing (Skip already processed images)")
        layout.addWidget(self.resume_processing_checkbox, row, 0, 1, 2); row += 1
        self.use_question_file_checkbox = QCheckBox("Use General Question File (for all models except Moondream)")
        self.use_question_file_checkbox.stateChanged.connect(self._toggle_question_inputs)
        layout.addWidget(self.use_question_file_checkbox, row, 0, 1, 2); row += 1
        self.common_question_label = QLabel("Common VQA Question:")
        layout.addWidget(self.common_question_label, row, 0)
        self.common_question_input = QLineEdit()
        layout.addWidget(self.common_question_input, row, 1); row += 1
        self.question_file_label = QLabel("General Question JSON File:")
        layout.addWidget(self.question_file_label, row, 0)
        self.question_file_input = QLineEdit()
        self.question_file_input.setReadOnly(True)
        layout.addWidget(self.question_file_input, row, 1)
        self.browse_question_file_button = QPushButton("Browse")
        self.browse_question_file_button.clicked.connect(self._browse_question_file)
        layout.addWidget(self.browse_question_file_button, row, 2); row += 1
        layout.addWidget(QLabel("WDv3 Tagger General Threshold:"), row, 0)
        self.waifu_diffusion_general_threshold_spinbox = QDoubleSpinBox()
        self.waifu_diffusion_general_threshold_spinbox.setRange(0.0, 1.0)
        self.waifu_diffusion_general_threshold_spinbox.setSingleStep(0.05)
        layout.addWidget(self.waifu_diffusion_general_threshold_spinbox, row, 1); row += 1
        layout.addWidget(QLabel("WDv3 Tagger Character Threshold:"), row, 0)
        self.waifu_diffusion_character_threshold_spinbox = QDoubleSpinBox()
        self.waifu_diffusion_character_threshold_spinbox.setRange(0.0, 1.0)
        self.waifu_diffusion_character_threshold_spinbox.setSingleStep(0.05)
        layout.addWidget(self.waifu_diffusion_character_threshold_spinbox, row, 1); row += 1
        layout.addWidget(QLabel("LLaVA GPU Layers (-1 for all):"), row, 0)
        self.llava_n_gpu_layers_spinbox = QSpinBox()
        self.llava_n_gpu_layers_spinbox.setRange(-1, 999)
        layout.addWidget(self.llava_n_gpu_layers_spinbox, row, 1); row += 1
        layout.addWidget(QLabel("LLaVA Context Window (n_ctx):"), row, 0)
        self.llava_n_ctx_spinbox = QSpinBox()
        self.llava_n_ctx_spinbox.setRange(512, 16384)
        self.llava_n_ctx_spinbox.setSingleStep(512)
        layout.addWidget(self.llava_n_ctx_spinbox, row, 1)

    def _create_florence2_settings(self):
        self.florence2_settings_group_box = QGroupBox("Florence-2 Task Settings")
        layout = QGridLayout(self.florence2_settings_group_box)
        layout.addWidget(QLabel("Caption Style:"), 0, 0)
        self.florence_caption_style_combo = QComboBox()
        self.florence_caption_style_combo.addItems(["Normal", "Detailed", "More Detailed"])
        layout.addWidget(self.florence_caption_style_combo, 0, 1, 1, 2)
        self.florence_vqa_checkbox = QCheckBox("Enable VQA")
        layout.addWidget(self.florence_vqa_checkbox, 1, 0)
        self.florence_od_checkbox = QCheckBox("Enable Object Detection")
        layout.addWidget(self.florence_od_checkbox, 1, 1)
        self.florence_dense_caption_checkbox = QCheckBox("Enable Dense Region Caption")
        layout.addWidget(self.florence_dense_caption_checkbox, 1, 2)
        self.florence_ocr_checkbox = QCheckBox("Enable OCR")
        layout.addWidget(self.florence_ocr_checkbox, 2, 0)
        self.florence_ocr_with_region_checkbox = QCheckBox("Enable OCR with Region")
        layout.addWidget(self.florence_ocr_with_region_checkbox, 2, 1)
        self.florence_ocr_filter_checkbox = QCheckBox("Filter OCR Results")
        self.florence_ocr_filter_checkbox.setToolTip("Removes junk characters from OCR results.")
        layout.addWidget(self.florence_ocr_filter_checkbox, 2, 2)
        self.florence_region_proposal_checkbox = QCheckBox("Enable Region Proposal")
        layout.addWidget(self.florence_region_proposal_checkbox, 3, 0)
        self.florence_caption_grounding_checkbox = QCheckBox("Enable Caption to Phrase Grounding")
        layout.addWidget(self.florence_caption_grounding_checkbox, 3, 1, 1, 2)

    def _create_moondream_settings(self):
        self.moondream_settings_group_box = QGroupBox("Moondream Task Settings")
        layout = QGridLayout(self.moondream_settings_group_box)
        self.moondream_vqa_checkbox = QCheckBox("Enable VQA")
        layout.addWidget(self.moondream_vqa_checkbox, 0, 0)
        self.use_moondream_question_file_checkbox = QCheckBox("Use Moondream-Specific Question File")
        layout.addWidget(self.use_moondream_question_file_checkbox, 1, 0, 1, 2)
        self.moondream_question_file_label = QLabel("Moondream Question JSON File:")
        layout.addWidget(self.moondream_question_file_label, 2, 0)
        self.moondream_question_file_input = QLineEdit()
        self.moondream_question_file_input.setReadOnly(True)
        layout.addWidget(self.moondream_question_file_input, 2, 1)
        self.browse_moondream_question_file_button = QPushButton("Browse")
        self.browse_moondream_question_file_button.clicked.connect(self._browse_moondream_question_file)
        layout.addWidget(self.browse_moondream_question_file_button, 2, 2)

    def _create_model_selection_settings(self):
        self.model_selection_group_box = QGroupBox("Enabled Models & Parameters")
        layout = QGridLayout(self.model_selection_group_box)
        layout.addWidget(QLabel("Model"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel("Enable"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel("Batch Size"), 0, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel("Max Words"), 0, 3, Qt.AlignmentFlag.AlignCenter)
        
        self.model_ui_map = {
            "CLIP_Interrogator": "clip_interrogator", "BLIP": "blip",
            "Florence-2": "florence", "JoyCaption": "llava", "GIT": "git",
            "WD_Tagger": "wd_tagger", "Moondream": "moondream",
            "SmolVLM": "smolvlm",
        }
        self.model_checkboxes, self.model_batch_spinboxes, self.model_max_words_spinboxes = {}, {}, {}
        for i, (key, json_key) in enumerate(self.model_ui_map.items()):
            row = i + 1
            layout.addWidget(QLabel(f"{key}:"), row, 0)
            cb = QCheckBox(); self.model_checkboxes[json_key] = cb; layout.addWidget(cb, row, 1, Qt.AlignmentFlag.AlignCenter); cb.stateChanged.connect(self._update_model_status)
            bs = QSpinBox(); bs.setRange(1, 999); self.model_batch_spinboxes[json_key] = bs; layout.addWidget(bs, row, 2)
            if json_key in ['blip', 'florence', 'llava', 'git', 'moondream', 'smolvlm']:
                mw = QSpinBox(); mw.setRange(1, 9999999); self.model_max_words_spinboxes[json_key] = mw; layout.addWidget(mw, row, 3)

    def _create_vqa_model_selection_settings(self):
        self.vqa_model_selection_group_box = QGroupBox("VQA Enabled Models")
        layout = QHBoxLayout(self.vqa_model_selection_group_box)
        self.vqa_ui_map = { 
            "BLIP": "blip", "Florence-2": "florence", "JoyCaption": "llava", 
            "GIT": "git", "Moondream": "moondream", "SmolVLM": "smolvlm"
        }
        self.vqa_model_checkboxes = {}
        for name, key in self.vqa_ui_map.items():
            cb = QCheckBox(name); self.vqa_model_checkboxes[key] = cb; layout.addWidget(cb)
    
    def _set_ui_from_config(self):
        self.model_dir_input.setText(self.config['model_dir'])
        self.image_dir_input.setText(self.config['image_dir'])
        self.output_dir_input.setText(self.config['output_dir'])
        self.use_system_image_limits_checkbox.setChecked(self.config['use_system_image_limits'])
        self.max_width_spinbox.setValue(self.config['max_width'])
        self.max_height_spinbox.setValue(self.config['max_height'])
        self._toggle_image_limits_inputs(self.config['use_system_image_limits'])
        self.use_cuda_graphs_checkbox.setChecked(self.config.get('use_cuda_graphs', False))
        self.concurrent_loading_checkbox.setChecked(self.config['concurrent_loading'])
        self.resume_processing_checkbox.setChecked(self.config['resume_processing'])
        self.use_question_file_checkbox.setChecked(self.config['use_question_file'])
        self.question_file_input.setText(self.config['question_json_path'])
        self.common_question_input.setText(self.config['common_question'])
        self._toggle_question_inputs(self.config['use_question_file'])
        self.waifu_diffusion_general_threshold_spinbox.setValue(self.config['waifu_diffusion_general_threshold'])
        self.waifu_diffusion_character_threshold_spinbox.setValue(self.config['waifu_diffusion_character_threshold'])
        self.llava_n_gpu_layers_spinbox.setValue(self.config['llava_n_gpu_layers'])
        self.llava_n_ctx_spinbox.setValue(self.config['llava_n_ctx'])
        self.florence_caption_style_combo.setCurrentText(self.config['florence_caption_style'])
        self.florence_vqa_checkbox.setChecked(self.config['florence_enable_vqa'])
        self.florence_od_checkbox.setChecked(self.config['florence_enable_od'])
        self.florence_dense_caption_checkbox.setChecked(self.config['florence_enable_dense_caption'])
        self.florence_ocr_checkbox.setChecked(self.config['florence_enable_ocr'])
        self.florence_ocr_with_region_checkbox.setChecked(self.config.get('florence_enable_ocr_with_region', False))
        self.florence_region_proposal_checkbox.setChecked(self.config.get('florence_enable_region_proposal', False))
        self.florence_caption_grounding_checkbox.setChecked(self.config.get('florence_enable_caption_grounding', False))
        self.florence_ocr_filter_checkbox.setChecked(self.config.get('florence_filter_ocr', False))
        
        self.moondream_vqa_checkbox.setChecked(self.config.get('moondream_enable_vqa', True))
        self.use_moondream_question_file_checkbox.setChecked(self.config.get('use_moondream_question_file', False))
        self.moondream_question_file_input.setText(self.config.get('moondream_question_json_path', ''))

        for key, cb in self.model_checkboxes.items(): cb.setChecked(self.config['models_enabled'].get(key, False))
        for key, sb in self.model_batch_spinboxes.items(): sb.setValue(self.config['model_specific_batch_sizes'].get(key, 8))
        for key, sb in self.model_max_words_spinboxes.items(): sb.setValue(self.config['model_specific_max_words'].get(key, 100))
        for key, cb in self.vqa_model_checkboxes.items(): cb.setChecked(self.config['models_vqa_enabled'].get(key, False))

    def _get_current_config(self):
        self.config['model_dir'] = self.model_dir_input.text()
        self.config['image_dir'] = self.image_dir_input.text()
        self.config['output_dir'] = self.output_dir_input.text()
        self.config['use_system_image_limits'] = self.use_system_image_limits_checkbox.isChecked()
        self.config['max_width'] = self.max_width_spinbox.value()
        self.config['max_height'] = self.max_height_spinbox.value()
        self.config['use_cuda_graphs'] = self.use_cuda_graphs_checkbox.isChecked()
        self.config['concurrent_loading'] = self.concurrent_loading_checkbox.isChecked()
        self.config['resume_processing'] = self.resume_processing_checkbox.isChecked()
        self.config['use_question_file'] = self.use_question_file_checkbox.isChecked()
        self.config['question_json_path'] = self.question_file_input.text()
        self.config['common_question'] = self.common_question_input.text()
        self.config['waifu_diffusion_general_threshold'] = self.waifu_diffusion_general_threshold_spinbox.value()
        self.config['waifu_diffusion_character_threshold'] = self.waifu_diffusion_character_threshold_spinbox.value()
        self.config['llava_n_gpu_layers'] = self.llava_n_gpu_layers_spinbox.value()
        self.config['llava_n_ctx'] = self.llava_n_ctx_spinbox.value()
        self.config['clip_model_variant'] = 'light' if self.clip_light_radio.isChecked() else 'heavy'
        self.config['florence_caption_style'] = self.florence_caption_style_combo.currentText()
        self.config['florence_enable_vqa'] = self.florence_vqa_checkbox.isChecked()
        self.config['florence_enable_od'] = self.florence_od_checkbox.isChecked()
        self.config['florence_enable_dense_caption'] = self.florence_dense_caption_checkbox.isChecked()
        self.config['florence_enable_ocr'] = self.florence_ocr_checkbox.isChecked()
        self.config['florence_enable_ocr_with_region'] = self.florence_ocr_with_region_checkbox.isChecked()
        self.config['florence_enable_region_proposal'] = self.florence_region_proposal_checkbox.isChecked()
        self.config['florence_enable_caption_grounding'] = self.florence_caption_grounding_checkbox.isChecked()
        self.config['florence_filter_ocr'] = self.florence_ocr_filter_checkbox.isChecked()
        
        self.config['moondream_enable_vqa'] = self.moondream_vqa_checkbox.isChecked()
        self.config['use_moondream_question_file'] = self.use_moondream_question_file_checkbox.isChecked()
        self.config['moondream_question_json_path'] = self.moondream_question_file_input.text()

        for key, cb in self.model_checkboxes.items(): self.config['models_enabled'][key] = cb.isChecked()
        for key, sb in self.model_batch_spinboxes.items(): self.config['model_specific_batch_sizes'][key] = sb.value()
        for key, sb in self.model_max_words_spinboxes.items(): self.config['model_specific_max_words'][key] = sb.value()
        for key, cb in self.vqa_model_checkboxes.items(): self.config['models_vqa_enabled'][key] = cb.isChecked()
            
        save_config(self.config)
        self.log_message.emit("Configuration saved.")

    def _apply_settings(self):
        self._get_current_config()
        self.log_message.emit("Settings applied and saved.")
        self.config = load_config()
        self._update_model_status()

    def _toggle_image_limits_inputs(self, state):
        self.max_width_spinbox.setEnabled(not bool(state))
        self.max_height_spinbox.setEnabled(not bool(state))

    def _toggle_question_inputs(self, state):
        use_file = bool(state)
        self.common_question_label.setEnabled(not use_file)
        self.common_question_input.setEnabled(not use_file)
        self.question_file_label.setEnabled(use_file)
        self.question_file_input.setEnabled(use_file)
        self.browse_question_file_button.setEnabled(use_file)

    def _browse_directory(self, line_edit, config_key):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text())
        if directory:
            line_edit.setText(directory)
            self.config[config_key] = directory
            save_config(self.config)
            self.log_message.emit(f"Updated {config_key} to: {directory}")
            self._update_model_status()

    def _browse_question_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select General Question File", "", "JSON Files (*.json)")
        if filepath:
            self.question_file_input.setText(filepath)

    def _browse_moondream_question_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Moondream Question File", "", "JSON Files (*.json)")
        if filepath:
            self.moondream_question_file_input.setText(filepath)

    def _start_download(self):
        if self.downloader and self.downloader.isRunning(): return
        self._get_current_config()
        self.downloader = ModelDownloader(self.config['model_dir'], self.config)
        self.downloader.progress.connect(self._update_download_progress)
        self.downloader.finished.connect(self._download_finished)
        self.downloader.log.connect(self._append_log)
        self.set_controls_enabled(False, is_downloading=True)
        self.downloader.start()

    def _stop_download(self):
        if self.downloader and self.downloader.isRunning(): self.downloader.stop()

    def _download_finished(self, message):
        self.log_message.emit(message)
        self.set_controls_enabled(True)
        self._update_model_status()

    def _update_download_progress(self, message, value):
        self.model_status_label.setText(f"Download Status: {message}")

    def _update_model_status(self):
        self.log_message.emit("Checking local model status...")
        all_present, missing = True, []
        model_dir = self.config['model_dir']
        for key, json_key in ModelDownloader.JSON_KEY_MAP.items():
            if self.config['models_enabled'].get(json_key, False):
                if "CLIP" in key:
                    if self.config['clip_model_variant'] == 'heavy' and key != 'CLIP_HEAVY': continue
                    if self.config['clip_model_variant'] == 'light' and key != 'CLIP_LIGHT': continue
                details = ModelDownloader.MODEL_REGISTRY[key]
                path = os.path.join(model_dir, key)
                is_present = os.path.isdir(path) and (not details.get("filenames") or all(os.path.exists(os.path.join(path, f)) for f in details["filenames"]))
                if not is_present:
                    all_present, missing = False, missing + [key]
        self.model_status_label.setText("Model Status: All enabled models are present." if all_present else f"Model Status: Missing: {', '.join(missing)}.")
        self.start_button.setEnabled(all_present)
        self.log_message.emit("Local model status checked.")

    def _start_processing(self):
        if self.processing_worker and self.processing_worker.isRunning(): return
        self._get_current_config()
        if not all(os.path.isdir(self.config[d]) for d in ['image_dir', 'output_dir']):
            QMessageBox.warning(self, "Error", "Image or Output directory not found."); return
        if not any(self.config['models_enabled'].values()):
            QMessageBox.warning(self, "Warning", "No models are enabled."); return
        if self.config['use_question_file'] and not os.path.exists(self.config['question_json_path']):
            QMessageBox.warning(self, "Error", "General question file is enabled but the path is invalid."); return
        if self.config.get('use_moondream_question_file') and not os.path.exists(self.config.get('moondream_question_json_path')):
            QMessageBox.warning(self, "Error", "Moondream question file is enabled but the path is invalid."); return
        if not self.start_button.isEnabled():
             QMessageBox.warning(self, "Error", "Missing enabled models. Please download them first."); return
        
        self.processing_worker = ProcessingWorker(self.config)
        self.processing_worker.progress.connect(self._update_processing_progress)
        self.processing_worker.finished.connect(self._processing_finished)
        self.processing_worker.log.connect(self._append_log)
        self.set_controls_enabled(False)
        self.log_text_edit.clear()
        self.processing_worker.start()

    def stop_processing(self):
        if self.processing_worker and self.processing_worker.isRunning():
            self.processing_worker.stop()
            self.stop_button.setEnabled(False)

    def _update_processing_progress(self, current, total, task_name):
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_bar.setFormat(f"Progress: {current}/{total} ({percentage}%)")
        self.current_image_label.setText(f"Current Task: {task_name}")

    def _processing_finished(self, message):
        self.log_message.emit(message)
        self.set_controls_enabled(True)
        self.progress_bar.setValue(100)
        self.current_image_label.setText("Processing Complete!")

    def _append_log(self, message):
        self.log_text_edit.appendPlainText(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def set_controls_enabled(self, enabled, is_processing=None, is_downloading=False):
        is_processing = not enabled if is_processing is None else is_processing
        self.start_button.setEnabled(enabled)
        self.stop_button.setEnabled(is_processing)
        self.download_models_button.setEnabled(enabled)
        self.stop_download_button.setEnabled(is_downloading)
        self.settings_tab.setEnabled(enabled)

    def closeEvent(self, event):
        self.log_message.emit("Stopping active processes...")
        if self.processing_worker and self.processing_worker.isRunning():
            self.stop_processing(); self.processing_worker.wait(5000)
        if self.downloader and self.downloader.isRunning():
            self._stop_download(); self.downloader.wait(5000)
        self._get_current_config()
        self.log_message.emit("Configuration saved. Exiting.")
        event.accept()