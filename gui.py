# gui.py

import os
import sys
import shlex
import re
import tempfile
import getpass
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QPlainTextEdit,
    QCheckBox, QSpinBox, QGroupBox, QMessageBox, QScrollArea, QTabWidget,
    QGridLayout, QComboBox, QDoubleSpinBox, QApplication, QRadioButton
)
import subprocess
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
from urllib.parse import urlparse

from utils import ModelDownloader, get_model_path, resolve_model_dir, save_config, load_config
from processing import ProcessingWorker
from standalone_workers import (
    SGLangStandaloneWorker,
    QwenFastInferenceWorker,
    QwenLatentExtractWorker,
    QwenVocabCacheWorker,
)

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

        # Debug/Workstation tabs (standalone pipelines)
        self._setup_qwen_standalone()
        self._setup_sglang_standalone()

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

        self._create_qwen_settings()
        self.settings_content_layout.addWidget(self.qwen_settings_group_box)

        self._create_sglang_settings()
        self.settings_content_layout.addWidget(self.sglang_settings_group_box)

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

    def _create_qwen_settings(self):
        self.qwen_settings_group_box = QGroupBox("Fast Qwen Task Settings")
        layout = QGridLayout(self.qwen_settings_group_box)
        row = 0

        self.qwen_legacy_support_checkbox = QCheckBox("Legacy Support (Write Output as 'wd_tagger')")
        layout.addWidget(self.qwen_legacy_support_checkbox, row, 0, 1, 4); row += 1

        layout.addWidget(QLabel("Precision:"), row, 0)
        self.qwen_precision_combo = QComboBox()
        self.qwen_precision_combo.addItems(["bf16", "fp16", "fp32", "int8", "int4"])
        layout.addWidget(self.qwen_precision_combo, row, 1)

        layout.addWidget(QLabel("Embedding Dim:"), row, 2)
        self.qwen_output_dim_spinbox = QSpinBox()
        self.qwen_output_dim_spinbox.setRange(64, 2048)
        self.qwen_output_dim_spinbox.setSingleStep(64)
        layout.addWidget(self.qwen_output_dim_spinbox, row, 3); row += 1

        self.qwen_use_json_cache_checkbox = QCheckBox("Use Vocab JSON Cache")
        layout.addWidget(self.qwen_use_json_cache_checkbox, row, 0, 1, 4); row += 1

        self.qwen_tf32_checkbox = QCheckBox("TF32 Math")
        layout.addWidget(self.qwen_tf32_checkbox, row, 0)
        self.qwen_quant_checkbox = QCheckBox("AWQ/4-bit Quant")
        layout.addWidget(self.qwen_quant_checkbox, row, 1)

        layout.addWidget(QLabel("Prefetch:"), row, 2)
        self.qwen_prefetch_spinbox = QSpinBox()
        self.qwen_prefetch_spinbox.setRange(1, 64)
        layout.addWidget(self.qwen_prefetch_spinbox, row, 3); row += 1

        layout.addWidget(QLabel("Threshold:"), row, 0)
        self.qwen_threshold_spinbox = QDoubleSpinBox()
        self.qwen_threshold_spinbox.setRange(0.01, 1.0)
        self.qwen_threshold_spinbox.setSingleStep(0.05)
        layout.addWidget(self.qwen_threshold_spinbox, row, 1); row += 1

        layout.addWidget(QLabel("Max Tags:"), row, 0)
        self.qwen_max_tags_spinbox = QSpinBox()
        self.qwen_max_tags_spinbox.setRange(1, 9999)
        layout.addWidget(self.qwen_max_tags_spinbox, row, 1); row += 1

        example_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_embedding", "D+eng", "vocab_hybrid_meta.json")
        layout.addWidget(QLabel("Vocab JSON Cache Path:"), row, 0)
        self.qwen_json_cache_input = QLineEdit()
        self.qwen_json_cache_input.setPlaceholderText(example_cache)
        layout.addWidget(self.qwen_json_cache_input, row, 1, 1, 2)
        self.browse_qwen_cache_button = QPushButton("Browse")
        self.browse_qwen_cache_button.clicked.connect(lambda: self.qwen_json_cache_input.setText(QFileDialog.getOpenFileName(self, "Select JSON Cache", "", "JSON Files (*.json)")[0]))
        layout.addWidget(self.browse_qwen_cache_button, row, 3); row += 1

        example_inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(
            tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
        )
        layout.addWidget(QLabel("Inductor Cache Dir:"), row, 0)
        self.qwen_inductor_cache_dir_input = QLineEdit()
        self.qwen_inductor_cache_dir_input.setPlaceholderText(example_inductor_dir)
        layout.addWidget(self.qwen_inductor_cache_dir_input, row, 1, 1, 2)
        self.browse_qwen_inductor_cache_button = QPushButton("Browse")
        self.browse_qwen_inductor_cache_button.clicked.connect(
            lambda: self.qwen_inductor_cache_dir_input.setText(
                QFileDialog.getExistingDirectory(
                    self,
                    "Select Inductor Cache Directory",
                    self.qwen_inductor_cache_dir_input.text() or example_inductor_dir,
                )
            )
        )
        layout.addWidget(self.browse_qwen_inductor_cache_button, row, 3); row += 1

        layout.addWidget(QLabel("Inductor Compile Threads (0=default):"), row, 0)
        self.qwen_inductor_threads_spinbox = QSpinBox()
        self.qwen_inductor_threads_spinbox.setRange(0, 64)
        layout.addWidget(self.qwen_inductor_threads_spinbox, row, 1); row += 1

        self.qwen_compile_checkbox = QCheckBox("torch.compile")
        layout.addWidget(self.qwen_compile_checkbox, row, 0)
        self.qwen_dynamic_checkbox = QCheckBox("Dynamic Shapes")
        layout.addWidget(self.qwen_dynamic_checkbox, row, 1)
        self.qwen_cuda_graphs_checkbox = QCheckBox("CUDA Graphs")
        layout.addWidget(self.qwen_cuda_graphs_checkbox, row, 2)
        self.qwen_pinned_mem_checkbox = QCheckBox("Pinned Memory")
        layout.addWidget(self.qwen_pinned_mem_checkbox, row, 3)

    def _create_sglang_settings(self):
        self.sglang_settings_group_box = QGroupBox("SGLang API Settings")
        layout = QGridLayout(self.sglang_settings_group_box)
        
        self.sglang_legacy_support_checkbox = QCheckBox("Legacy Support (Write Output as 'smolvlm')")
        layout.addWidget(self.sglang_legacy_support_checkbox, 0, 0, 1, 1)

        self.sglang_disable_reasoning_checkbox = QCheckBox("Disable Reasoning Engine (Qwen3.5)")
        layout.addWidget(self.sglang_disable_reasoning_checkbox, 0, 1, 1, 1)

        self.sglang_format_reasoning_checkbox = QCheckBox("Format Reasoning as <reasoning> Tags")
        layout.addWidget(self.sglang_format_reasoning_checkbox, 0, 2, 1, 2)
        
        layout.addWidget(QLabel("Endpoint URL:"), 1, 0)
        self.sglang_url_input = QLineEdit()
        layout.addWidget(self.sglang_url_input, 1, 1, 1, 3)

        layout.addWidget(QLabel("System Prompt:"), 2, 0)
        self.sglang_system_prompt_input = QPlainTextEdit()
        self.sglang_system_prompt_input.setMaximumHeight(80)
        layout.addWidget(self.sglang_system_prompt_input, 2, 1, 1, 3)

        layout.addWidget(QLabel("Max New Tokens:"), 3, 0)
        self.sglang_max_tokens_spinbox = QSpinBox()
        self.sglang_max_tokens_spinbox.setRange(1, 99999)
        layout.addWidget(self.sglang_max_tokens_spinbox, 3, 1)

        layout.addWidget(QLabel("Max Image Res:"), 3, 2)
        self.sglang_max_res_spinbox = QSpinBox()
        self.sglang_max_res_spinbox.setRange(64, 4096)
        layout.addWidget(self.sglang_max_res_spinbox, 3, 3)

        layout.addWidget(QLabel("Max Concurrency:"), 4, 0)
        self.sglang_concurrency_spinbox = QSpinBox()
        self.sglang_concurrency_spinbox.setRange(1, 128)
        layout.addWidget(self.sglang_concurrency_spinbox, 4, 1)

        self.sglang_use_native_generate_checkbox = QCheckBox("Use native /generate (Faster)")
        layout.addWidget(self.sglang_use_native_generate_checkbox, 4, 2, 1, 2)

        self.sglang_auto_wsl_checkbox = QCheckBox("Auto-Start SGLang Server via WSL")
        layout.addWidget(self.sglang_auto_wsl_checkbox, 5, 0, 1, 2)

        self.sglang_shutdown_wsl_checkbox = QCheckBox("Shutdown WSL on unload (Frees VRAM)")
        layout.addWidget(self.sglang_shutdown_wsl_checkbox, 5, 2, 1, 2)

        layout.addWidget(QLabel("WSL Venv Activate:"), 6, 0)
        self.sglang_wsl_activate_input = QLineEdit()
        self.sglang_wsl_activate_input.setPlaceholderText("Example: ~/your-venv/bin/activate")
        layout.addWidget(self.sglang_wsl_activate_input, 6, 1, 1, 2)
        self.sglang_wsl_activate_browse_button = QPushButton("Browse (Win)")
        self.sglang_wsl_activate_browse_button.clicked.connect(self._browse_sglang_wsl_activate_windows)
        layout.addWidget(self.sglang_wsl_activate_browse_button, 6, 3)

        layout.addWidget(QLabel("WSL Model Path:"), 7, 0)
        self.sglang_wsl_model_path_input = QLineEdit()
        self.sglang_wsl_model_path_input.setPlaceholderText("Example: /path/to/model or /mnt/c/...")
        layout.addWidget(self.sglang_wsl_model_path_input, 7, 1, 1, 2)
        self.sglang_wsl_model_path_browse_button = QPushButton("Browse (Win)")
        self.sglang_wsl_model_path_browse_button.clicked.connect(self._browse_sglang_wsl_model_windows)
        layout.addWidget(self.sglang_wsl_model_path_browse_button, 7, 3)

        layout.addWidget(QLabel("Launch Command:"), 8, 0)
        self.sglang_wsl_launch_input = QLineEdit()
        self.sglang_wsl_launch_input.setPlaceholderText("Example: python3 -m sglang.launch_server")
        layout.addWidget(self.sglang_wsl_launch_input, 8, 1, 1, 3)

        layout.addWidget(QLabel("Extra Args:"), 9, 0)
        self.sglang_wsl_extra_args_input = QLineEdit()
        self.sglang_wsl_extra_args_input.setPlaceholderText("Example: --context-length 65536 --mem-fraction-static 0.6")
        layout.addWidget(self.sglang_wsl_extra_args_input, 9, 1, 1, 2)

        self.sglang_build_wsl_cmd_button = QPushButton("Build WSL Command")
        self.sglang_build_wsl_cmd_button.clicked.connect(self._build_sglang_wsl_command)
        layout.addWidget(self.sglang_build_wsl_cmd_button, 9, 3)

        layout.addWidget(QLabel("WSL Init Command:"), 10, 0)
        self.sglang_wsl_cmd_input = QLineEdit()
        self.sglang_wsl_cmd_input.setPlaceholderText(
            "Example: source ~/your-venv/bin/activate && python3 -m sglang.launch_server --model-path /path/to/model --port 30000"
        )
        layout.addWidget(self.sglang_wsl_cmd_input, 10, 1, 1, 3)

        self.sglang_manual_mode_button = QPushButton("Use Manual Server Mode")
        self.sglang_manual_mode_button.clicked.connect(self._use_sglang_manual_mode)
        layout.addWidget(self.sglang_manual_mode_button, 11, 1, 1, 2)

    def _windows_path_to_wsl(self, win_path: str) -> str:
        """
        Best-effort conversion: `C:\\foo\\bar` -> `/mnt/c/foo/bar`

        This intentionally only handles Windows drive-letter paths, since WSL-local paths
        (like `~/venv/bin/activate`) must be provided by the user.
        """
        p = (win_path or "").strip().strip("\"'")
        if not p:
            return ""
        p = p.replace("\\", "/")
        m = re.match(r"^([A-Za-z]):/(.*)$", p)
        if not m:
            return p
        drive = m.group(1).lower()
        rest = m.group(2)
        return f"/mnt/{drive}/{rest}"

    def _browse_sglang_wsl_activate_windows(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select WSL Venv Activate Script (Windows Path)", "", "All Files (*)")
        if not path:
            return
        self.sglang_wsl_activate_input.setText(self._windows_path_to_wsl(path))

    def _browse_sglang_wsl_model_windows(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Model Directory (Windows Path)", self.model_dir_input.text())
        if not directory:
            return
        self.sglang_wsl_model_path_input.setText(self._windows_path_to_wsl(directory))

    def _endpoint_port_from_url(self, url_text: str) -> int:
        try:
            parsed = urlparse((url_text or "").strip())
            if parsed.port:
                return int(parsed.port)
        except Exception:
            pass
        return 30000

    def _wsl_path_expr(self, path_text: str) -> str:
        p = (path_text or "").strip().strip("'\"")
        if not p:
            return ""
        if p == "~":
            return '"$HOME"'
        if p.startswith("~/"):
            rest = p[2:].replace('"', '\\"')
            return f'"$HOME/{rest}"'
        if p.startswith("$HOME"):
            rest = p[len("$HOME"):].replace('"', '\\"')
            return f'"$HOME{rest}"'
        if p.startswith("${HOME}"):
            rest = p[len("${HOME}"):].replace('"', '\\"')
            return f'"$HOME{rest}"'
        return shlex.quote(p)

    def _build_sglang_wsl_command(self):
        activate = (self.sglang_wsl_activate_input.text() or "").strip()
        model_path = (self.sglang_wsl_model_path_input.text() or "").strip()
        launch = (self.sglang_wsl_launch_input.text() or "").strip() or "python3 -m sglang.launch_server"
        extra_args = (self.sglang_wsl_extra_args_input.text() or "").strip()

        if not model_path:
            QMessageBox.warning(self, "SGLang WSL Builder", "Please set a WSL Model Path before building the command.")
            return

        activate_expr = self._wsl_path_expr(activate)
        model_expr = self._wsl_path_expr(model_path)

        if activate_expr:
            cmd = f"source {activate_expr} && {launch} --model-path {model_expr}"
        else:
            cmd = f"{launch} --model-path {model_expr}"

        if extra_args:
            cmd = cmd + " " + extra_args

        if "--port" not in cmd:
            port = self._endpoint_port_from_url(self.sglang_url_input.text())
            cmd = cmd + f" --port {port}"

        self.sglang_wsl_cmd_input.setText(cmd)

    def _use_sglang_manual_mode(self):
        self.sglang_auto_wsl_checkbox.setChecked(False)
        QMessageBox.information(
            self,
            "SGLang Manual Server Mode",
            "Auto-Start via WSL is now OFF.\n\nStart the SGLang server manually (WSL/Linux) and set the Endpoint URL to match the server port.",
        )

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
            "SmolVLM": "smolvlm", "Qwen": "qwen", "SGLang": "sglang"
        }
        self.model_checkboxes, self.model_batch_spinboxes, self.model_max_words_spinboxes = {}, {}, {}
        for i, (key, json_key) in enumerate(self.model_ui_map.items()):
            row = i + 1
            layout.addWidget(QLabel(f"{key}:"), row, 0)
            cb = QCheckBox(); self.model_checkboxes[json_key] = cb; layout.addWidget(cb, row, 1, Qt.AlignmentFlag.AlignCenter); cb.stateChanged.connect(self._update_model_status)
            bs = QSpinBox(); bs.setRange(1, 999); self.model_batch_spinboxes[json_key] = bs; layout.addWidget(bs, row, 2)
            if json_key in ['blip', 'florence', 'llava', 'git', 'moondream', 'smolvlm', 'qwen', 'sglang']:
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

        self.qwen_legacy_support_checkbox.setChecked(self.config.get('qwen_legacy_support', False))
        self.qwen_precision_combo.setCurrentText(self.config.get('qwen_precision', 'bf16'))
        self.qwen_output_dim_spinbox.setValue(int(self.config.get("qwen_output_dim", 512) or 512))
        self.qwen_use_json_cache_checkbox.setChecked(self.config.get('qwen_use_json_cache', True))
        self.qwen_tf32_checkbox.setChecked(self.config.get('qwen_tf32', True))
        self.qwen_quant_checkbox.setChecked(self.config.get('qwen_quant', False))
        self.qwen_prefetch_spinbox.setValue(self.config.get('qwen_prefetch', 2))
        self.qwen_threshold_spinbox.setValue(self.config.get('qwen_threshold', 0.30))
        self.qwen_max_tags_spinbox.setValue(self.config.get('qwen_max_tags', 50))
        cache_path = self.config.get('qwen_json_cache_path', '') or ''
        if not cache_path:
            example_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_embedding", "D+eng", "vocab_hybrid_meta.json")
            if os.path.exists(example_cache):
                cache_path = example_cache
        self.qwen_json_cache_input.setText(cache_path)
        self.qwen_inductor_cache_dir_input.setText((self.config.get("qwen_inductor_cache_dir") or "").strip())
        self.qwen_inductor_threads_spinbox.setValue(int(self.config.get("qwen_inductor_compile_threads", 0) or 0))
        self.qwen_compile_checkbox.setChecked(self.config.get('qwen_compile', True))
        self.qwen_dynamic_checkbox.setChecked(self.config.get('qwen_dynamic', True))
        self.qwen_cuda_graphs_checkbox.setChecked(self.config.get('qwen_cuda_graphs', False))
        self.qwen_pinned_mem_checkbox.setChecked(self.config.get('qwen_pinned_mem', True))

        # Latent Preprocessing Tab Restorations
        self.qwen_latent_compile_cb.setChecked(self.config.get('qwen_latent_compile', True))
        self.qwen_latent_dynamic_cb.setChecked(self.config.get('qwen_latent_dynamic', True))
        self.qwen_latent_cuda_graphs_cb.setChecked(self.config.get('qwen_latent_cuda_graphs', False))
        self.qwen_latent_pinned_mem_cb.setChecked(self.config.get('qwen_latent_pinned_mem', True))
        self.qwen_latent_precision_combo.setCurrentText(self.config.get('qwen_latent_precision', 'bf16'))
        
        fallback_bs = self.config.get('model_specific_batch_sizes', {}).get('qwen', 8)
        self.qwen_latent_bs_spinbox.setValue(self.config.get('qwen_latent_batch_size', fallback_bs))
        self.qwen_latent_prefetch.setValue(self.config.get('qwen_latent_prefetch', 2))

        self.sglang_legacy_support_checkbox.setChecked(self.config.get('sglang_legacy_support', False))
        self.sglang_use_native_generate_checkbox.setChecked(self.config.get("sglang_use_native_generate", True))
        self.sglang_url_input.setText(self.config.get('sglang_url', 'http://127.0.0.1:30000/generate'))
        self.sglang_system_prompt_input.setPlainText(self.config.get('sglang_system_context', ''))
        self.sglang_max_tokens_spinbox.setValue(self.config.get('sglang_max_tokens', 1024))
        self.sglang_max_res_spinbox.setValue(self.config.get('sglang_max_res', 256))
        self.sglang_concurrency_spinbox.setValue(self.config.get('sglang_concurrency', 40))
        self.sglang_auto_wsl_checkbox.setChecked(self.config.get('sglang_auto_wsl', False))
        self.sglang_shutdown_wsl_checkbox.setChecked(self.config.get("sglang_shutdown_wsl_on_unload", True))
        self.sglang_disable_reasoning_checkbox.setChecked(self.config.get("sglang_disable_reasoning", True))
        self.sglang_format_reasoning_checkbox.setChecked(self.config.get("sglang_format_reasoning", True))
        self.sglang_wsl_activate_input.setText((self.config.get('sglang_wsl_activate') or '').strip())
        self.sglang_wsl_model_path_input.setText((self.config.get('sglang_wsl_model_path') or '').strip())
        self.sglang_wsl_launch_input.setText((self.config.get('sglang_wsl_launch') or '').strip())
        self.sglang_wsl_extra_args_input.setText((self.config.get('sglang_wsl_extra_args') or '').strip())
        self.sglang_wsl_cmd_input.setText((self.config.get('sglang_wsl_cmd') or '').strip())

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

        self.config['qwen_legacy_support'] = self.qwen_legacy_support_checkbox.isChecked()
        self.config['qwen_precision'] = self.qwen_precision_combo.currentText()
        self.config['qwen_output_dim'] = self.qwen_output_dim_spinbox.value()
        self.config['qwen_use_json_cache'] = self.qwen_use_json_cache_checkbox.isChecked()
        self.config['qwen_tf32'] = self.qwen_tf32_checkbox.isChecked()
        self.config['qwen_quant'] = self.qwen_quant_checkbox.isChecked()
        self.config['qwen_prefetch'] = self.qwen_prefetch_spinbox.value()
        self.config['qwen_threshold'] = self.qwen_threshold_spinbox.value()
        self.config['qwen_max_tags'] = self.qwen_max_tags_spinbox.value()
        self.config['qwen_json_cache_path'] = self.qwen_json_cache_input.text()
        self.config['qwen_inductor_cache_dir'] = self.qwen_inductor_cache_dir_input.text()
        self.config['qwen_inductor_compile_threads'] = self.qwen_inductor_threads_spinbox.value()
        self.config['qwen_compile'] = self.qwen_compile_checkbox.isChecked()
        self.config['qwen_dynamic'] = self.qwen_dynamic_checkbox.isChecked()
        self.config['qwen_cuda_graphs'] = self.qwen_cuda_graphs_checkbox.isChecked()
        self.config['qwen_pinned_mem'] = self.qwen_pinned_mem_checkbox.isChecked()

        self.config['sglang_legacy_support'] = self.sglang_legacy_support_checkbox.isChecked()
        self.config["sglang_use_native_generate"] = self.sglang_use_native_generate_checkbox.isChecked()
        self.config['sglang_url'] = self.sglang_url_input.text()
        self.config['sglang_system_context'] = self.sglang_system_prompt_input.toPlainText()
        self.config['sglang_max_tokens'] = self.sglang_max_tokens_spinbox.value()
        self.config['sglang_max_res'] = self.sglang_max_res_spinbox.value()
        self.config['sglang_concurrency'] = self.sglang_concurrency_spinbox.value()
        self.config["sglang_shutdown_wsl_on_unload"] = self.sglang_shutdown_wsl_checkbox.isChecked()
        self.config['sglang_auto_wsl'] = self.sglang_auto_wsl_checkbox.isChecked()
        self.config["sglang_disable_reasoning"] = self.sglang_disable_reasoning_checkbox.isChecked()
        self.config["sglang_format_reasoning"] = self.sglang_format_reasoning_checkbox.isChecked()
        self.config['sglang_wsl_activate'] = self.sglang_wsl_activate_input.text()
        self.config['sglang_wsl_model_path'] = self.sglang_wsl_model_path_input.text()
        self.config['sglang_wsl_launch'] = self.sglang_wsl_launch_input.text()
        self.config['sglang_wsl_extra_args'] = self.sglang_wsl_extra_args_input.text()
        self.config['sglang_wsl_cmd'] = self.sglang_wsl_cmd_input.text()

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
        for key, json_key in ModelDownloader.JSON_KEY_MAP.items():
            if self.config['models_enabled'].get(json_key, False):
                if "CLIP" in key:
                    if self.config['clip_model_variant'] == 'heavy' and key != 'CLIP_HEAVY': continue
                    if self.config['clip_model_variant'] == 'light' and key != 'CLIP_LIGHT': continue
                details = ModelDownloader.MODEL_REGISTRY[key]
                resolved = resolve_model_dir(key, self.config, details)
                if not resolved:
                    all_present, missing = False, missing + [key]

        # Warning preflight: Qwen vocab cache tensor validation.
        # We no longer strictly lock all_present = False here because processing.py falls back gracefully.
        if self.config['models_enabled'].get("qwen", False) and self.config.get("qwen_use_json_cache", True):
            cache_json = (self.config.get("qwen_json_cache_path") or "").strip()
            if not cache_json:
                default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_embedding", "D+eng", "vocab_hybrid_meta.json")
                cache_json = default_cache if os.path.exists(default_cache) else ""
            tensor_ok = False
            if cache_json and os.path.exists(cache_json):
                try:
                    with open(cache_json, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(0, size - 65536))
                        tail = f.read()
                    m = re.search(br'"tensor_path"\s*:\s*"([^"]+)"', tail)
                    tensor_rel = m.group(1).decode("utf-8", errors="ignore").strip() if m else ""
                    tensor_path = os.path.join(os.path.dirname(cache_json), tensor_rel) if tensor_rel else ""
                    tensor_ok = bool(tensor_path and os.path.exists(tensor_path))
                except Exception:
                    tensor_ok = False

            if not tensor_ok:
                self.log_message.emit("WARNING: Qwen Vocab JSON Cache is missing or invalid. Qwen will use a localized fallback dictionary.")
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



    def _setup_qwen_standalone(self):
        self.qwen_tab = QWidget()
        self.tab_widget.addTab(self.qwen_tab, "Qwen Workstation (Standalone)")
        layout = QVBoxLayout(self.qwen_tab)

        qwen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_embedding")
        qwen_eng_dir = os.path.join(qwen_dir, "D+eng")
        qwen_default_out_dir = qwen_eng_dir if os.path.isdir(qwen_eng_dir) else qwen_dir
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # --- Vocab Cache Builder ---
        vocab_tab = QWidget()
        vocab_layout = QGridLayout(vocab_tab)
        tabs.addTab(vocab_tab, "1. Build Vocab Cache")
        
        vocab_layout.addWidget(QLabel("Danbooru JSON Source:"), 0, 0)
        self.qwen_danbooru_input = QLineEdit(os.path.join(qwen_dir, "data.json"))
        vocab_layout.addWidget(self.qwen_danbooru_input, 0, 1)
        browse_danbooru_btn = QPushButton("Browse")
        browse_danbooru_btn.clicked.connect(lambda: self.qwen_danbooru_input.setText(QFileDialog.getOpenFileName(self, "Select Danbooru JSON", qwen_dir, "JSON Files (*.json)")[0]))
        vocab_layout.addWidget(browse_danbooru_btn, 0, 2)
        
        vocab_layout.addWidget(QLabel("Output Matrix (.pt):"), 1, 0)
        self.qwen_pt_out_input = QLineEdit(os.path.join(qwen_default_out_dir, "vocab_hybrid_matrix.pt"))
        vocab_layout.addWidget(self.qwen_pt_out_input, 1, 1)
        browse_pt_btn = QPushButton("Browse")
        browse_pt_btn.clicked.connect(lambda: self.qwen_pt_out_input.setText(QFileDialog.getSaveFileName(self, "Save Matrix (.pt)", qwen_default_out_dir, "PyTorch Tensors (*.pt)")[0]))
        vocab_layout.addWidget(browse_pt_btn, 1, 2)
        
        vocab_layout.addWidget(QLabel("Output Meta (.json):"), 2, 0)
        self.qwen_json_out_input = QLineEdit(os.path.join(qwen_default_out_dir, "vocab_hybrid_meta.json"))
        vocab_layout.addWidget(self.qwen_json_out_input, 2, 1)
        browse_meta_btn = QPushButton("Browse")
        browse_meta_btn.clicked.connect(lambda: self.qwen_json_out_input.setText(QFileDialog.getSaveFileName(self, "Save Meta (.json)", qwen_default_out_dir, "JSON Files (*.json)")[0]))
        vocab_layout.addWidget(browse_meta_btn, 2, 2)
        
        vocab_layout.addWidget(QLabel("Min Occurrences:"), 3, 0)
        self.qwen_min_occ = QSpinBox(); self.qwen_min_occ.setRange(1, 100000); self.qwen_min_occ.setValue(200)
        vocab_layout.addWidget(self.qwen_min_occ, 3, 1)
        
        vocab_layout.addWidget(QLabel("Batch Size (GPU):"), 4, 0)
        self.qwen_vocab_bs = QSpinBox(); self.qwen_vocab_bs.setRange(1, 1048576); self.qwen_vocab_bs.setValue(64)
        vocab_layout.addWidget(self.qwen_vocab_bs, 4, 1)
        
        self.qwen_use_eng = QCheckBox("Merge 370k DWYL English Dictionary")
        self.qwen_use_eng.setChecked(True)
        vocab_layout.addWidget(self.qwen_use_eng, 5, 0, 1, 2)
        
        self.qwen_gen_vocab_btn = QPushButton("Generate Massive GPU Cache (.pt / .json)")
        self.qwen_gen_vocab_btn.clicked.connect(self._run_qwen_vocab)
        vocab_layout.addWidget(self.qwen_gen_vocab_btn, 6, 0, 1, 2)
        vocab_layout.setRowStretch(7, 1)
        
        # --- Latent Extractor ---
        latent_tab = QWidget()
        latent_layout = QVBoxLayout(latent_tab)
        tabs.addTab(latent_tab, "2. Extract Image Latents")
        
        latent_layout.addWidget(QLabel("Image Directory:"))
        self.qwen_latent_dir = QLineEdit()
        latent_layout.addWidget(self.qwen_latent_dir)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.qwen_latent_dir.setText(QFileDialog.getExistingDirectory(self, "Select Image Directory")))
        latent_layout.addWidget(btn)
        
        perf_group = QGroupBox("Preprocessing Performance Specs")
        perf_layout = QGridLayout(perf_group)
        
        perf_layout.addWidget(QLabel("Precision:"), 0, 0)
        self.qwen_latent_precision_combo = QComboBox()
        self.qwen_latent_precision_combo.addItems(["bf16", "fp32", "int8", "int4"])
        perf_layout.addWidget(self.qwen_latent_precision_combo, 0, 1)

        perf_layout.addWidget(QLabel("Latent Batch Size:"), 0, 2)
        self.qwen_latent_bs_spinbox = QSpinBox()
        self.qwen_latent_bs_spinbox.setRange(1, 65536)
        self.qwen_latent_bs_spinbox.setValue(8)
        perf_layout.addWidget(self.qwen_latent_bs_spinbox, 0, 3)

        perf_layout.addWidget(QLabel("Prefetch Threads:"), 1, 0)
        self.qwen_latent_prefetch = QSpinBox()
        self.qwen_latent_prefetch.setRange(1, 64)
        self.qwen_latent_prefetch.setValue(2)
        perf_layout.addWidget(self.qwen_latent_prefetch, 1, 1)

        self.qwen_latent_compile_cb = QCheckBox("torch.compile")
        self.qwen_latent_dynamic_cb = QCheckBox("Dynamic Shapes")
        self.qwen_latent_cuda_graphs_cb = QCheckBox("CUDA Graphs")
        self.qwen_latent_pinned_mem_cb = QCheckBox("Pinned Memory")

        self.qwen_latent_compile_cb.setChecked(True)
        self.qwen_latent_dynamic_cb.setChecked(True)
        self.qwen_latent_cuda_graphs_cb.setChecked(False)
        self.qwen_latent_pinned_mem_cb.setChecked(True)

        perf_layout.addWidget(self.qwen_latent_compile_cb, 2, 0)
        perf_layout.addWidget(self.qwen_latent_dynamic_cb, 2, 1)
        perf_layout.addWidget(self.qwen_latent_cuda_graphs_cb, 2, 2)
        perf_layout.addWidget(self.qwen_latent_pinned_mem_cb, 2, 3)

        latent_layout.addWidget(perf_group)
        
        self.qwen_latent_progress = QProgressBar()
        self.qwen_latent_progress.setTextVisible(True)
        self.qwen_latent_progress.setFormat("Idle")
        self.qwen_latent_progress.setValue(0)
        latent_layout.addWidget(self.qwen_latent_progress)
        
        btn_layout = QHBoxLayout()
        self.qwen_extract_btn = QPushButton("Extract Image Latents (Packed Cache: qwen_image_cache.pt)")
        self.qwen_extract_btn.clicked.connect(self._run_qwen_latents)
        btn_layout.addWidget(self.qwen_extract_btn)
        
        self.qwen_extract_stop_btn = QPushButton("Stop")
        self.qwen_extract_stop_btn.clicked.connect(self._stop_qwen_latents)
        self.qwen_extract_stop_btn.setDisabled(True)
        btn_layout.addWidget(self.qwen_extract_stop_btn)
        
        latent_layout.addLayout(btn_layout)
        
        # Global Qwen Log
        layout.addWidget(QLabel("Workers Log:"))
        self.qwen_log = QPlainTextEdit()
        self.qwen_log.setReadOnly(True)
        self.qwen_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.qwen_log)
    def _run_qwen_vocab(self):
        self.qwen_gen_vocab_btn.setDisabled(True)
        self.qwen_worker = QwenVocabCacheWorker(
            self.qwen_danbooru_input.text(), self.qwen_min_occ.value(),
            self.qwen_vocab_bs.value(), self.qwen_use_eng.isChecked(),
            self.qwen_pt_out_input.text(), self.qwen_json_out_input.text()
        )
        self.qwen_worker.log.connect(self.qwen_log.appendPlainText)
        self.qwen_worker.finished.connect(lambda: self.qwen_gen_vocab_btn.setEnabled(True))
        self.qwen_worker.start()

    def _run_qwen_latents(self):
        d = self.qwen_latent_dir.text()
        if not d: return
        self.qwen_extract_btn.setDisabled(True)
        self.qwen_extract_stop_btn.setEnabled(True)
        self._update_all_configs() # flush UI to config
        
        # Override global configs with the locally configured Latent tab settings
        latent_config = self.config.copy()
        latent_config.update({
            "qwen_precision": self.qwen_latent_precision_combo.currentText(),
            "qwen_tf32": True,
            "qwen_compile": self.qwen_latent_compile_cb.isChecked(),
            "qwen_dynamic": self.qwen_latent_dynamic_cb.isChecked(),
            "qwen_cuda_graphs": self.qwen_latent_cuda_graphs_cb.isChecked(),
            "qwen_pinned_mem": self.qwen_latent_pinned_mem_cb.isChecked(),
            "qwen_latent_batch_size": self.qwen_latent_bs_spinbox.value(),
            "qwen_prefetch": self.qwen_latent_prefetch.value(),
        })
        self.qwen_lat_worker = QwenLatentExtractWorker(self.qwen_latent_dir.text(), latent_config)
        self.qwen_lat_worker.log.connect(self.qwen_log.appendPlainText)
        self.qwen_lat_worker.progress.connect(self._update_qwen_latent_progress)
        self.qwen_lat_worker.finished.connect(self._on_qwen_latents_finished)
        self.qwen_lat_worker.start()

    def _stop_qwen_latents(self):
        if hasattr(self, 'qwen_lat_worker') and self.qwen_lat_worker.isRunning():
            self.qwen_extract_stop_btn.setDisabled(True)
            self.qwen_lat_worker.cancel()

    def _on_qwen_latents_finished(self):
        self.qwen_extract_btn.setEnabled(True)
        self.qwen_extract_stop_btn.setDisabled(True)

    def _update_qwen_latent_progress(self, current: int, total: int, msg: str):
        if total > 0:
            self.qwen_latent_progress.setMaximum(total)
            self.qwen_latent_progress.setValue(current)
            self.qwen_latent_progress.setFormat(f"{msg} (%p%)")
        else:
            self.qwen_latent_progress.setMaximum(0)
            self.qwen_latent_progress.setFormat(msg)
    def _setup_sglang_standalone(self):
        self.sglang_tab = QWidget()
        self.tab_widget.addTab(self.sglang_tab, "SGLang Workstation (Standalone)")
        layout = QVBoxLayout(self.sglang_tab)
        
        form_layout = QGridLayout()
        layout.addLayout(form_layout)
        
        form_layout.addWidget(QLabel("Input Images Directory:"), 0, 0)
        self.sglang_std_input = QLineEdit()
        form_layout.addWidget(self.sglang_std_input, 0, 1)
        btn1 = QPushButton("Browse")
        btn1.clicked.connect(lambda: self.sglang_std_input.setText(QFileDialog.getExistingDirectory(self, "Input")))
        form_layout.addWidget(btn1, 0, 2)
        
        form_layout.addWidget(QLabel("Target Image Extension:"), 1, 0)
        self.sglang_std_ext = QLineEdit("*.jpg")
        form_layout.addWidget(self.sglang_std_ext, 1, 1)
        
        form_layout.addWidget(QLabel("Output JSONL File:"), 2, 0)
        self.sglang_std_output = QLineEdit()
        form_layout.addWidget(self.sglang_std_output, 2, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.sglang_std_output.setText(QFileDialog.getSaveFileName(self, "Output", "", "JSONL (*.jsonl)")[0]))
        form_layout.addWidget(btn2, 2, 2)
        
        layout.addWidget(QLabel("URL, Concurrency, Context, Max Tokens, Max Res imported from Settings->SGLang."))
        layout.addWidget(QLabel("(SGLang Pipeline start button removed to enforce unified batch start sequence.)"))
        
        self.sglang_log = QPlainTextEdit()
        self.sglang_log.setReadOnly(True)
        self.sglang_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.sglang_log)

    def _update_all_configs(self):
        self.config['qwen_use_json_cache'] = self.qwen_use_json_cache_checkbox.isChecked()
        self.config['qwen_tf32'] = self.qwen_tf32_checkbox.isChecked()
        self.config['qwen_quant'] = self.qwen_quant_checkbox.isChecked()
        self.config['qwen_prefetch'] = self.qwen_prefetch_spinbox.value()
        self.config['qwen_json_cache_path'] = self.qwen_json_cache_input.text()
        self.config['qwen_inductor_cache_dir'] = self.qwen_inductor_cache_dir_input.text()
        self.config['qwen_inductor_compile_threads'] = self.qwen_inductor_threads_spinbox.value()
        self.config['qwen_compile'] = self.qwen_compile_checkbox.isChecked()
        self.config['qwen_dynamic'] = self.qwen_dynamic_checkbox.isChecked()
        self.config['qwen_cuda_graphs'] = self.qwen_cuda_graphs_checkbox.isChecked()
        self.config['qwen_pinned_mem'] = self.qwen_pinned_mem_checkbox.isChecked()

        self.config['sglang_legacy_support'] = self.sglang_legacy_support_checkbox.isChecked()
        self.config['sglang_url'] = self.sglang_url_input.text()
        self.config['sglang_system_context'] = self.sglang_system_prompt_input.toPlainText()
        self.config['sglang_max_tokens'] = self.sglang_max_tokens_spinbox.value()
        self.config['sglang_max_res'] = self.sglang_max_res_spinbox.value()
        self.config['sglang_concurrency'] = self.sglang_concurrency_spinbox.value()
        self.config['sglang_auto_wsl'] = self.sglang_auto_wsl_checkbox.isChecked()
        self.config['sglang_wsl_cmd'] = self.sglang_wsl_cmd_input.text()

        for key, cb in self.model_checkboxes.items(): self.config['models_enabled'][key] = cb.isChecked()
        for key, sb in self.model_batch_spinboxes.items(): self.config['model_specific_batch_sizes'][key] = sb.value()
        for key, sb in self.model_max_words_spinboxes.items(): self.config['model_specific_max_words'][key] = sb.value()

    def closeEvent(self, event):
        self.log_message.emit("Stopping active processes...")
        if self.processing_worker and self.processing_worker.isRunning():
            self.stop_processing(); self.processing_worker.wait(5000)
        if self.downloader and self.downloader.isRunning():
            self._stop_download(); self.downloader.wait(5000)
        self._get_current_config()
        self.log_message.emit("Configuration saved. Exiting.")
        event.accept()
