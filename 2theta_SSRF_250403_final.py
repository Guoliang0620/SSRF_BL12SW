import sys
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import simps
from scipy import sparse
from scipy.sparse.linalg import spsolve
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
                            QFileDialog, QMessageBox, QToolBar, QGroupBox, QScrollArea,
                            QSizePolicy, QHeaderView, QSplitter, QComboBox, QAction)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QDoubleValidator, QColor, QBrush, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import os

# Physical constants
HC_KEV_ANGSTROM = 12.39842  # h*c (keV·Å)

class Dataset:
    """Class to store dataset information"""
    def __init__(self, filename=None, raw_data=None):
        self.filename = filename
        self.raw_data = raw_data          # [channel, counts]
        self.adjusted_data = None         # [energy, counts]
        self.fit_results = []             # Peak fitting results
        self.selected_regions = []        # ROI storage
        self.x_axis_adjusted = False      # Flag if calibration was applied
        
        # Copy data as adjusted on init
        if raw_data is not None:
            self.adjusted_data = np.copy(raw_data)

    @property
    def name(self):
        """Get dataset display name"""
        if self.filename:
            return os.path.basename(self.filename)
        return "Unnamed Dataset"

class XRDDataAnalyzer(QMainWindow):
    # 样式常量
    BUTTON_STYLE_TPL = """
        QPushButton {{
            background-color: {};
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {};
        }}
    """
    
    GROUP_STYLE_TPL = "QGroupBox {{ border: 1px solid {}; font-weight: bold; padding-top: 15px; }}"
    
    # UI 颜色映射
    UI_COLORS = {
        'import': ('#1f77b4', '#4c9acd'),  # 主色, 悬停色
        'calibration': ('#2ca02c', '#5abb5a'),
        'export': ('#d62728', '#e85858'),
        'roi': ('#ff7f0e', '#e67e22'),
        'fit': ('#9467bd', '#b08be0'),
        'clear': ('#7f7f7f', '#a6a6a6'),
        'theta': ('#17becf', '#58d7e6'),
        'export_calib': ('#e377c2', '#f0a6d9'),
    }
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2theta fitting tool_SSRF BL12SW Ver.20250403")
        self.setGeometry(100, 100, 1280, 960)
        
        # Initialize settings
        self.settings = QSettings("SSRF", "2theta")
        
        # Data storage - modified for multiple datasets
        self.datasets = []  # List to store multiple datasets
        self.current_dataset_index = -1  # Current active dataset index
        self.calib_params = None      # Calibration parameters
        
        # Color configuration for plots/ROIs - using vibrant colors
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Blue, Orange, Green, Red, Purple
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'   # Brown, Pink, Gray, Olive, Cyan
        ]
        
        # UI interaction state
        self.region_selection_active = False
        self.current_roi = None
        
        # Initialize UI
        self.init_ui()
        self.connect_signals()
        
        # Restore saved values
        self.restore_saved_values()
        
    @property
    def current_dataset(self):
        """Get current active dataset or None"""
        if 0 <= self.current_dataset_index < len(self.datasets):
            return self.datasets[self.current_dataset_index]
        return None
        
    @property
    def raw_data(self):
        """Legacy accessor for raw_data from current dataset"""
        dataset = self.current_dataset
        return dataset.raw_data if dataset else None
        
    @property
    def adjusted_data(self):
        """Legacy accessor for adjusted_data from current dataset"""
        dataset = self.current_dataset
        return dataset.adjusted_data if dataset else None
        
    @property
    def selected_regions(self):
        """Legacy accessor for selected_regions from current dataset"""
        dataset = self.current_dataset
        return dataset.selected_regions if dataset else []
        
    @property
    def fit_results(self):
        """Legacy accessor for fit_results from current dataset"""
        dataset = self.current_dataset
        return dataset.fit_results if dataset else []

    def init_ui(self):
        """Initialize user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Contributors label in top right
        contributors_label = QLabel("Contributors: Niu Guoliang (HPSTAR) & Zhou Chunyin (SSRF)")
        contributors_label.setStyleSheet("color: #666666; font-weight: bold;")
        contributors_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        main_layout.addWidget(contributors_label)
        
        # Top control area (30% height)
        top_group = QGroupBox("Data Control")
        top_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 15px; }")
        top_layout = QHBoxLayout(top_group)
        top_layout.setSpacing(20)
        top_layout.setContentsMargins(10, 20, 10, 10)
        main_layout.setStretchFactor(top_group, 3)  # 30% height
        
        # Data import section
        data_group = QGroupBox("Data Import")
        data_group.setStyleSheet("QGroupBox { border: 1px solid #1f77b4; }")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(10)
        
        # Dataset selector
        dataset_layout = QHBoxLayout()
        dataset_layout.setSpacing(10)
        self.dataset_selector = QComboBox()
        self.dataset_selector.setStyleSheet("""
            QComboBox {
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                min-width: 150px;
            }
        """)
        self.dataset_selector.setToolTip("Select active dataset")
        dataset_layout.addWidget(QLabel("Dataset:"))
        dataset_layout.addWidget(self.dataset_selector, 1)
        data_layout.addLayout(dataset_layout)
        
        self.import_btn = QPushButton("Import Data")
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #1f77b4;
                color: white;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.import_btn.setToolTip("Click to import XRD data file")
        self.filename_label = QLabel("No data loaded")
        self.filename_label.setStyleSheet("QLabel { color: #666666; }")
        data_layout.addWidget(self.import_btn)
        data_layout.addWidget(self.filename_label)
        top_layout.addWidget(data_group, 40)  # Set width ratio to 40%
        
        # Energy calibration section
        adj_group = QGroupBox("Energy Calibration (a*Ch² + b*Ch + c)")
        adj_group.setStyleSheet("QGroupBox { border: 1px solid #2ca02c; }")
        adj_layout = QVBoxLayout(adj_group)
        adj_layout.setSpacing(10)
        param_layout = QHBoxLayout()
        param_layout.setSpacing(10)
        for name, val in [("a", "0.0"), ("b", "0.03758"), ("c", "-0.04962")]:
            label = QLabel(f"{name}:")
            label.setStyleSheet("QLabel { font-weight: bold; }")
            edit = QLineEdit(val)
            edit.setObjectName(f"{name}_edit")
            edit.setValidator(QDoubleValidator(-100, 100, 8))
            edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    padding: 3px;
                    border-radius: 3px;
                }
            """)
            param_layout.addWidget(label)
            param_layout.addWidget(edit)
        
        self.adjust_btn = QPushButton("Apply Calibration")
        self.adjust_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ca02c;
                color: white;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.adjust_btn.setToolTip("Apply energy calibration parameters")
        
        self.export_btn = QPushButton("Export Data")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #d62728;
                color: white;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.export_btn.setToolTip("Export calibrated data")
        
        adj_layout.addLayout(param_layout)
        adj_layout.addWidget(self.adjust_btn)
        adj_layout.addWidget(self.export_btn)
        top_layout.addWidget(adj_group, 60)  # Set width ratio to 60%
        
        main_layout.addWidget(top_group)

        # Main workspace
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("mainSplitter")  # Set name for saving state
        
        # Left control panel (40% width)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # Peak analysis controls
        peak_group = QGroupBox("Peak Analysis")
        peak_group.setStyleSheet("QGroupBox { border: 1px solid #ff7f0e; font-weight: bold; padding-top: 15px; }")
        peak_layout = QVBoxLayout(peak_group)
        peak_layout.setSpacing(10)
        
        self.select_btn = QPushButton("Select ROI", checkable=True)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff7f0e;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #d35400;
                border: 2px solid #ff7f0e;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.select_btn.setToolTip("Select region of interest (ROI)")
        
        self.fit_btn = QPushButton("Fit Peaks")
        self.fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #9467bd;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.fit_btn.setToolTip("Perform Gaussian fit on selected ROI")
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f7f7f;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.clear_btn.setToolTip("Clear all ROIs and fit results")
        
        peak_layout.addWidget(self.select_btn)
        peak_layout.addWidget(self.fit_btn)
        peak_layout.addWidget(self.clear_btn)
        left_layout.addWidget(peak_group)
        
        # 2θ calibration inputs
        calib_group = QGroupBox("2θ Calibration")
        calib_group.setStyleSheet("QGroupBox { border: 1px solid #17becf; font-weight: bold; padding-top: 15px; }")
        calib_layout = QVBoxLayout(calib_group)
        calib_layout.setSpacing(10)
        
        self.calib_table = QTableWidget(9, 3)
        self.calib_table.setHorizontalHeaderLabels(["d (Å)", "E (keV)", "Remarks"])
        # Set default MgO values
        self.calib_table.setItem(0, 0, QTableWidgetItem("2.1065"))
        self.calib_table.setItem(0, 2, QTableWidgetItem("MgO_200"))
        self.calib_table.setItem(1, 0, QTableWidgetItem("1.4895"))
        self.calib_table.setItem(1, 2, QTableWidgetItem("MgO_220"))
        
        # Restore column widths if available
        if self.settings.contains("calibTable/columnWidths"):
            widths = self.settings.value("calibTable/columnWidths")
            for col, width in enumerate(widths):
                self.calib_table.setColumnWidth(col, width)
        else:
            # Set default widths
            self.calib_table.setColumnWidth(0, 120)  # d (Å)
            self.calib_table.setColumnWidth(1, 100)  # E (keV)
            self.calib_table.setColumnWidth(2, 200)  # Remarks
        
        # Save column widths when changed
        self.calib_table.horizontalHeader().sectionResized.connect(self.save_calib_column_widths)
        
        self.calib_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
            QHeaderView::section {
                background-color: #17becf;
                color: white;
                padding: 4px;
                font-weight: bold;
            }
        """)
        
        self.calibrate_btn = QPushButton("Calibrate 2θ")
        self.calibrate_btn.setStyleSheet("""
            QPushButton {
                background-color: #17becf;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        
        self.export_calib_btn = QPushButton("Export Calibration")
        self.export_calib_btn.setStyleSheet("""
            QPushButton {
                background-color: #e377c2;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        
        self.calib_status = QLabel("Status: Waiting for calibration")
        self.calib_status.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background-color: #f5f5f5;
            }
        """)
        
        calib_layout.addWidget(self.calib_table)
        calib_layout.addWidget(self.calibrate_btn)
        calib_layout.addWidget(self.export_calib_btn)
        calib_layout.addWidget(self.calib_status)
        left_layout.addWidget(calib_group)
        
        splitter.addWidget(left_panel)

        # Right display area with vertical splitter
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setObjectName("rightSplitter")  # Set name for saving state
        
        # Use timer to restore splitter states after UI is fully initialized
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, lambda: self.restore_splitter_states(splitter, right_splitter))
        
        # Plot area
        # Plot area widget
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(10, 6))
        self.figure.patch.set_facecolor('#f9f9f9')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#f9f9f9')
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        right_splitter.addWidget(plot_widget)
        
        # Results area widget
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        
        self.result_table = QTableWidget(0, 5)  # Start with 0 rows
        self.result_table.setHorizontalHeaderLabels([
            "ROI", "Center (keV)", "FWHM (keV)", "Intensity", "R²"
        ])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        
        # Disable auto-stretch to allow manual column resize
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        
        # Restore column widths if available
        if self.settings.contains("resultTable/columnWidths"):
            widths = self.settings.value("resultTable/columnWidths")
            for col, width in enumerate(widths):
                if col < self.result_table.columnCount():
                    self.result_table.setColumnWidth(col, int(width))
        else:
            # Set default widths
            self.result_table.setColumnWidth(0, 200)  # ROI
            self.result_table.setColumnWidth(1, 120)  # Center
            self.result_table.setColumnWidth(2, 120)  # FWHM
            self.result_table.setColumnWidth(3, 120)  # Intensity
            self.result_table.setColumnWidth(4, 80)   # R²
        
        # Save column widths when changed
        self.result_table.horizontalHeader().sectionResized.connect(self.save_column_widths)
        
        # Add table to scroll area to ensure scrollability when needed
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.result_table)
        results_layout.addWidget(scroll_area)
        right_splitter.addWidget(results_widget)
        
        # Set initial sizes for the splitters to show full table content
        right_splitter.setSizes([400, 600])  # Plot area 40%, Results 60%
        
        splitter.addWidget(right_splitter)
        
        # Configure splitter behavior
        splitter.setHandleWidth(8)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #e0e0e0;
            }
            QSplitter::handle:hover {
                background: #1f77b4;
            }
        """)
        splitter.setStretchFactor(0, 1)  # Left panel
        splitter.setStretchFactor(1, 3)  # Right panel
        splitter.setSizes([400, 800])  # Initial sizes (33%/67%)
        splitter.splitterMoved.connect(lambda: self.settings.setValue("mainSplitter", splitter.saveState()))
        
        # Configure right splitter
        right_splitter.setHandleWidth(8)
        right_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #e0e0e0;
            }
            QSplitter::handle:hover {
                background: #1f77b4;
            }
        """)
        right_splitter.setStretchFactor(0, 2)  # Plot area
        right_splitter.setStretchFactor(1, 1)  # Results area
        right_splitter.setSizes([500, 400])  # Initial sizes (55%/45%)
        right_splitter.splitterMoved.connect(lambda: self.settings.setValue("rightSplitter", right_splitter.saveState()))
        
        main_layout.addWidget(splitter)
        
        # Set size policy for resizing
        self.result_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_table.verticalHeader().setDefaultSectionSize(24)  # Compact row height
        self.result_table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        
        # Set contributor info in status bar
        self.statusBar().showMessage("Contributors: Niu Guoliang (HPSTAR) & Zhou Chunyin (SSRF) | Tip: Drag splitters to resize panels")

    def connect_signals(self):
        """Connect signals to slots"""
        self.import_btn.clicked.connect(self.import_data)
        self.adjust_btn.clicked.connect(self.adjust_energy_axis)
        self.export_btn.clicked.connect(self.export_data)
        self.select_btn.toggled.connect(self.toggle_roi_selection)
        self.fit_btn.clicked.connect(self.fit_peaks)
        self.clear_btn.clicked.connect(self.clear_all)
        self.calibrate_btn.clicked.connect(self.calibrate_2theta)
        self.export_calib_btn.clicked.connect(self.export_calibration)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_changed)
        
        # Plot interaction events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def on_dataset_changed(self, index):
        """Handle dataset selection change"""
        if 0 <= index < len(self.datasets):
            self.current_dataset_index = index
            dataset = self.current_dataset
            self.filename_label.setText(dataset.name if dataset else "No data loaded")
            
            # Update result table with current dataset's fit results
            self.update_result_table()
            
            # Refresh plot
            self.plot_data()
        
    def update_dataset_selector(self):
        """Update dataset selector with current datasets"""
        # Block signals to prevent triggering change events during update
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        
        for idx, dataset in enumerate(self.datasets):
            self.dataset_selector.addItem(dataset.name if dataset else f"Dataset {idx+1}")
        
        # Select current dataset
        if 0 <= self.current_dataset_index < self.dataset_selector.count():
            self.dataset_selector.setCurrentIndex(self.current_dataset_index)
            
        self.dataset_selector.blockSignals(False)
        
        # Update controls based on whether we have data
        has_data = len(self.datasets) > 0
        self.select_btn.setEnabled(has_data)
        self.fit_btn.setEnabled(has_data)
        self.adjust_btn.setEnabled(has_data)
        self.export_btn.setEnabled(has_data)
        
    def update_result_table(self):
        """Update result table with current dataset's fit results"""
        self.result_table.setRowCount(0)
        
        dataset = self.current_dataset
        if not dataset or not dataset.fit_results:
            return
            
        regions = dataset.selected_regions
        results = dataset.fit_results
        
        self.result_table.setRowCount(len(results))
        for idx, (region, result) in enumerate(zip(regions, results)):
            for col, val in enumerate([
                f"ROI {idx+1} ({region['x_min']:.2f}-{region['x_max']:.2f} keV)",
                f"{result['params'][1]:.5f}",
                f"{result['fwhm']:.5f}",
                f"{result['integral']:.5f}",
                f"{result['r_squared']:.5f}"
            ]):
                item = QTableWidgetItem(val)
                # Set ROI row style - background color matches plot, text color depends on background
                item.setBackground(QColor(region['color']))
                # Calculate background brightness to determine text color
                bg_color = QColor(region['color'])
                brightness = (bg_color.red() * 299 + bg_color.green() * 587 + bg_color.blue() * 114) / 1000
                text_color = QColor(Qt.black) if brightness > 128 else QColor(Qt.white)
                item.setForeground(text_color)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                
                # For other columns, set text color based on background
                if col > 0:
                    item.setForeground(text_color)
                
                # Ensure text remains visible when selected
                item.setData(Qt.BackgroundRole, QColor(region['color']))
                item.setData(Qt.ForegroundRole, text_color)
                
                self.result_table.setItem(idx, col, item)

    def save_column_widths(self):
        """Save result table column widths to settings"""
        widths = [self.result_table.columnWidth(i) for i in range(self.result_table.columnCount())]
        self.settings.setValue("resultTable/columnWidths", widths)
        
    def save_calib_column_widths(self):
        """Save calibration table column widths to settings"""
        widths = [self.calib_table.columnWidth(i) for i in range(self.calib_table.columnCount())]
        self.settings.setValue("calibTable/columnWidths", widths)
        
    def restore_saved_values(self):
        """Restore saved values from settings"""
        # Restore energy calibration parameters (a, b, c)
        if self.settings.contains("energy_calib/a"):
            a = self.settings.value("energy_calib/a")
            self.findChild(QLineEdit, "a_edit").setText(a)
        
        if self.settings.contains("energy_calib/b"):
            b = self.settings.value("energy_calib/b")
            self.findChild(QLineEdit, "b_edit").setText(b)
            
        if self.settings.contains("energy_calib/c"):
            c = self.settings.value("energy_calib/c")
            self.findChild(QLineEdit, "c_edit").setText(c)
            
        # Restore calibration table data
        if self.settings.contains("calib_table/data"):
            # Clear default values first
            for row in range(self.calib_table.rowCount()):
                for col in range(self.calib_table.columnCount()):
                    self.calib_table.setItem(row, col, QTableWidgetItem(""))
                    
            # Fill with saved values
            calib_data = self.settings.value("calib_table/data")
            if calib_data:
                for i, row_data in enumerate(calib_data):
                    if i >= self.calib_table.rowCount():
                        break
                        
                    if 'd' in row_data:
                        self.calib_table.setItem(i, 0, QTableWidgetItem(row_data['d']))
                    if 'e' in row_data:
                        self.calib_table.setItem(i, 1, QTableWidgetItem(row_data['e']))
                    if 'remarks' in row_data:
                        self.calib_table.setItem(i, 2, QTableWidgetItem(row_data['remarks']))
    
    def gaussian_with_baseline(self, x, a, x0, sigma, b, c):
        """Gaussian function with linear baseline"""
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b * x + c
        
    def plot_data(self):
        """Update plot with compact layout and ROI-based legend"""
        self.ax.clear()
        
        # Get current dataset
        dataset = self.current_dataset
        if not dataset:
            self.canvas.draw()
            return
            
        # Determine data source
        data = dataset.adjusted_data if dataset.x_axis_adjusted else dataset.raw_data
        if data is None:
            return
        
        # Store current Y limits before plotting, if they exist
        prev_y_min, prev_y_max = self.ax.get_ylim() if hasattr(self, '_y_limits') else (None, None)
        
        # Plot raw data
        self.ax.plot(data[:,0], data[:,1], 'k-', lw=1, label='Raw data')
        
        # Get current Y limits if we don't have stored ones
        if not hasattr(self, '_y_limits'):
            y_min, y_max = self.ax.get_ylim()
            # Store these limits for future use
            self._y_limits = (y_min, y_max)
        else:
            # Use stored limits
            y_min, y_max = self._y_limits

        # Explicitly set Y limits to ensure consistency
        self.ax.set_ylim(y_min, y_max)
        
        # Plot ROI regions and fits based on their state
        legend_handles = []
        legend_labels = []
        
        # Add raw data to legend
        line, = self.ax.plot([], [], 'k-', lw=1)
        legend_handles.append(line)
        legend_labels.append('Raw data')
        
        for i, roi in enumerate(dataset.selected_regions):
            # Check if this ROI has been fitted
            is_fitted = i < len(dataset.fit_results)
            
            if not is_fitted:
                # Show region box and label for unfitted ROIs
                rect = Rectangle(
                    (roi['x_min'], y_min),
                    roi['width'], y_max - y_min,
                    facecolor=roi['color'], alpha=0.3,
                    edgecolor=roi['color'], lw=1
                )
                # Store rect reference for clear_all
                roi['rect'] = rect
                self.ax.add_patch(rect)
                self.ax.text(roi['x_min'] + roi['width']/2, 
                            y_max,
                            f"Region {i+1} ({roi['x_min']:.2f}-{roi['x_max']:.2f} keV)",
                            color=roi['color'], ha='center', va='top')
                
                # Add to legend
                patch = Rectangle((0,0), 1, 1, facecolor=roi['color'], alpha=0.3)
                legend_handles.append(patch)
                legend_labels.append(f'Region {i+1} ({roi["x_min"]:.2f}-{roi["x_max"]:.2f} keV)')
            else:
                # Show fit line for fitted ROIs
                result = dataset.fit_results[i]
                x_fit = np.linspace(roi['x_min'], roi['x_max'], 100)
                y_fit = self.gaussian_with_baseline(x_fit, *result['params'])
                
                line_color = roi['color']
                line, = self.ax.plot(x_fit, y_fit, '--', color=line_color, lw=1.5)
                
                # Add to legend
                legend_handles.append(line)
                legend_labels.append(f'Fit {i+1} ({roi["x_min"]:.2f}-{roi["x_max"]:.2f} keV)')
                
                # Annotate center and FWHM
                center = result['params'][1]
                fwhm = result['fwhm']
                self.ax.axvline(x=center, color=line_color, linestyle=':', alpha=0.7)
                self.ax.axvspan(center - fwhm/2, center + fwhm/2, 
                               color=line_color, alpha=0.1)
        
        # Set axis labels and legend
        self.ax.set_xlabel('Energy (keV)' if dataset.x_axis_adjusted else 'Channel')
        self.ax.set_ylabel('Intensity')
        
        if legend_handles:
            self.ax.legend(legend_handles, legend_labels, loc='upper right', 
                         framealpha=0.7, fancybox=True)
        
        # Refresh canvas
        self.canvas.draw()
        
    def toggle_roi_selection(self, active):
        """Enable/disable ROI selection mode"""
        self.region_selection_active = active
        if active:
            self.statusBar().showMessage("Click and drag to select a region of interest")
        else:
            self.statusBar().showMessage("ROI selection canceled", 3000)
            self.current_roi = None
    
    def on_press(self, event):
        """Handle mouse press event for ROI selection"""
        if not self.region_selection_active or event.inaxes != self.ax:
            return
            
        # Start a new ROI
        self.current_roi = {
            'x_min': event.xdata,
            'y_min': event.ydata,
            'rect': None  # Rectangle will be created on release
        }
        
    def on_motion(self, event):
        """Handle mouse move event for ROI selection"""
        if not self.region_selection_active or self.current_roi is None:
            return
            
        # Update current selection box if it exists
        if 'rect' in self.current_roi and self.current_roi['rect']:
            self.current_roi['rect'].remove()
            
        # Draw temporary selection box
        if event.inaxes == self.ax:
            self.current_roi['width'] = max(0.001, event.xdata - self.current_roi['x_min'])
            y_min, y_max = self.ax.get_ylim()
            height = y_max - y_min
            
            rect = Rectangle(
                (self.current_roi['x_min'], y_min),
                self.current_roi['width'], height,
                facecolor='gray', alpha=0.3, edgecolor='gray'
            )
            self.ax.add_patch(rect)
            self.current_roi['rect'] = rect
            self.canvas.draw_idle()
            
    def on_release(self, event):
        """Handle mouse release event for ROI selection"""
        if not self.region_selection_active or self.current_roi is None:
            return
            
        if event.inaxes == self.ax and abs(event.xdata - self.current_roi['x_min']) > 0.001:
            # Calculate min and max x values
            x_min = min(self.current_roi['x_min'], event.xdata)
            x_max = max(self.current_roi['x_min'], event.xdata)
            width = abs(x_max - x_min)
            
            # Remove temporary rectangle
            if 'rect' in self.current_roi and self.current_roi['rect']:
                self.current_roi['rect'].remove()
            
            # Get next color from color list
            dataset = self.current_dataset
            if dataset:
                color_idx = len(dataset.selected_regions) % len(self.colors)
                color = self.colors[color_idx]
                
                # Add to dataset's selected regions
                dataset.selected_regions.append({
                    'x_min': x_min,
                    'x_max': x_max,
                    'width': width,
                    'color': color
                })
                
                # Update plot with new ROI
                self.plot_data()
                
                # Disable selection mode automatically
                self.select_btn.setChecked(False)
                self.region_selection_active = False
                self.statusBar().showMessage(f"ROI selected: {x_min:.3f} - {x_max:.3f}", 3000)
            
        self.current_roi = None
        
    def fit_peaks(self):
        """Perform Gaussian fit on selected ROIs"""
        dataset = self.current_dataset
        if not dataset or not dataset.selected_regions:
            QMessageBox.warning(self, "Warning", "No regions selected for fitting")
            return
            
        # Get X and Y data
        data = dataset.adjusted_data if dataset.x_axis_adjusted else dataset.raw_data
        x_data = data[:, 0]
        y_data = data[:, 1]
        
        # Clear previous fit results
        dataset.fit_results = []
        
        for roi in dataset.selected_regions:
            # Get data within region
            mask = (x_data >= roi['x_min']) & (x_data <= roi['x_max'])
            if not np.any(mask):
                continue
                
            roi_x = x_data[mask]
            roi_y = y_data[mask]
            
            # Initial guess for parameters
            peak_idx = np.argmax(roi_y)
            a_guess = roi_y[peak_idx]  # Peak height
            x0_guess = roi_x[peak_idx]  # Peak center
            sigma_guess = (roi['x_max'] - roi['x_min']) / 6  # Width estimate
            
            # Linear baseline guess
            b_guess, c_guess = 0, np.min(roi_y)
            
            # Initial parameter guess
            p0 = [a_guess, x0_guess, sigma_guess, b_guess, c_guess]
            bounds = ([0, roi['x_min'], 0, -np.inf, -np.inf], 
                     [np.inf, roi['x_max'], (roi['x_max'] - roi['x_min']), np.inf, np.inf])
            
            try:
                # Perform fit
                popt, pcov = curve_fit(self.gaussian_with_baseline, roi_x, roi_y, 
                                     p0=p0, bounds=bounds, maxfev=5000)
                
                # Calculate goodness of fit (R²)
                y_fit = self.gaussian_with_baseline(roi_x, *popt)
                ss_res = np.sum((roi_y - y_fit) ** 2)
                ss_tot = np.sum((roi_y - np.mean(roi_y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Calculate FWHM: 2*sqrt(2*ln(2))*sigma
                fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                
                # Calculate peak area: integral of Gaussian only
                x_fine = np.linspace(roi['x_min'], roi['x_max'], 1000)
                baseline = popt[3] * x_fine + popt[4]
                gaussian = popt[0] * np.exp(-(x_fine - popt[1])**2 / (2 * popt[2]**2))
                integral = simps(gaussian, x_fine)
                
                # Store fit results
                dataset.fit_results.append({
                    'params': popt,
                    'cov': pcov,
                    'fwhm': fwhm,
                    'integral': integral,
                    'r_squared': r_squared
                })
                
            except Exception as e:
                print(f"Fit error: {str(e)}")
                # Add dummy results
                dataset.fit_results.append({
                    'params': p0,
                    'cov': np.zeros((5, 5)),
                    'fwhm': 0,
                    'integral': 0,
                    'r_squared': 0
                })
        
        # Update plot and result table
        self.update_result_table()
        self.plot_data()
        
    def clear_all(self):
        """Clear all ROIs and fit results"""
        dataset = self.current_dataset
        if not dataset:
            return
            
        # Reset Y-axis limits so new data can auto-scale
        if hasattr(self, '_y_limits'):
            delattr(self, '_y_limits')
            
        # Use try/except to handle case where 'rect' key might not exist
        for roi in dataset.selected_regions:
            try:
                if 'rect' in roi and roi['rect']:
                    roi['rect'].remove()
            except:
                pass  # Skip if rect can't be removed
        
        dataset.selected_regions = []
        dataset.fit_results = []
        
        # Update UI
        self.result_table.setRowCount(0)
        self.plot_data()
        self.statusBar().showMessage("All ROIs and fit results cleared", 3000)
    
    def adjust_energy_axis(self):
        """Apply energy calibration to current dataset"""
        dataset = self.current_dataset
        if not dataset or dataset.raw_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded for calibration")
            return
            
        # Get calibration parameters
        try:
            a = float(self.findChild(QLineEdit, "a_edit").text())
            b = float(self.findChild(QLineEdit, "b_edit").text())
            c = float(self.findChild(QLineEdit, "c_edit").text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid calibration parameters")
            return
            
        # Apply calibration: E = a*Ch² + b*Ch + c
        channels = dataset.raw_data[:, 0]
        counts = dataset.raw_data[:, 1]
        energies = a * channels**2 + b * channels + c
        
        # Store calibrated data
        dataset.adjusted_data = np.column_stack((energies, counts))
        dataset.x_axis_adjusted = True
        
        # Clear previous ROIs and fit results as they're now invalid
        self.clear_all()
        
        # Update plot
        self.plot_data()
        self.statusBar().showMessage(f"Calibration applied: E = {a:.6f}*Ch² + {b:.6f}*Ch + {c:.6f}", 5000)
        
    def calibrate_2theta(self):
        """Perform 2θ calibration using d-spacing and energy pairs based on Bragg's law"""
        # Collect valid d-E pairs from table
        pairs = []
        for row in range(self.calib_table.rowCount()):
            d_item = self.calib_table.item(row, 0)
            e_item = self.calib_table.item(row, 1)
            
            if d_item and e_item and d_item.text() and e_item.text():
                try:
                    d = float(d_item.text())
                    E = float(e_item.text())
                    pairs.append((d, E))
                except ValueError:
                    continue
        
        if len(pairs) == 0:
            QMessageBox.warning(self, "Warning", "Need at least 1 d-E pair for calibration")
            return
            
        try:
            # Prepare arrays for calculation and display
            twotheta_values = []
            
            for d, E in pairs:
                # Calculate 2θ directly using Bragg's law: 2θ = 2*arcsin(hc/(2dE))
                twotheta_deg = 2 * np.degrees(np.arcsin(HC_KEV_ANGSTROM / (2 * d * E)))
                twotheta_values.append(twotheta_deg)
            
            # Calculate average and standard deviation
            avg_2theta = np.mean(twotheta_values)
            std_2theta = np.std(twotheta_values)
            
            # Store calibration parameters
            self.calibrated = True
            self.calib_params = [avg_2theta, std_2theta, HC_KEV_ANGSTROM]
            
            # Prepare status text
            status_text = f"2θ = {avg_2theta:.5f}° ± {std_2theta:.5f}° (n={len(pairs)})"
            
            self.calib_status.setText(status_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration failed: {str(e)}")
            
    def export_calibration(self):
        """Export calibration data"""
        if not hasattr(self, 'calibrated') or not self.calibrated:
            QMessageBox.warning(self, "Warning", "No calibration data available")
            return
        
        # Default filename with date and time
        from datetime import datetime
        default_filename = f"2theta cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
        # Get last used directory from settings
        last_dir = self.settings.value("last_export_calib_dir", "")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", os.path.join(last_dir, default_filename), "Text files (*.txt)")
        # Save directory for next time
        if filename:
            self.settings.setValue("last_export_calib_dir", os.path.dirname(filename))
        if filename:
            try:
                # Get all valid d-E pairs
                d_e_pairs = []
                for row in range(self.calib_table.rowCount()):
                    d_item = self.calib_table.item(row, 0)
                    e_item = self.calib_table.item(row, 1)
                    remarks_item = self.calib_table.item(row, 2)
                    
                    if d_item and e_item and d_item.text() and e_item.text():
                        d = float(d_item.text())
                        E = float(e_item.text())
                        remarks = remarks_item.text() if remarks_item else ""
                        d_e_pairs.append((d, E, remarks))
                
                # Save with header and calibration status
                with open(filename, 'w') as f:
                    f.write("# 2θ Calibration Data\n")
                    f.write(f"# {self.calib_status.text().replace('\n', ' | ')}\n")
                    f.write("d (Å)\tE (keV)\tRemarks\n")
                    for d, E, remarks in d_e_pairs:
                        f.write(f"{d:.5f}\t{E:.5f}\t{remarks}\n")
                
                self.statusBar().showMessage(f"Calibration data exported to {filename}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
                
    def export_data(self):
        """Export calibrated data"""
        dataset = self.current_dataset
        if not dataset or dataset.adjusted_data is None:
            QMessageBox.warning(self, "Warning", "No data available for export")
            return
        
        # Get default filename based on original filename
        if dataset.filename:
            base_name = os.path.splitext(os.path.basename(dataset.filename))[0]
            default_filename = f"{base_name}_converted.txt"
        else:
            default_filename = "converted_data.txt"
        
        # Get last used directory from settings
        last_dir = self.settings.value("last_export_dir", "")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Data", os.path.join(last_dir, default_filename), "Text files (*.txt)")
        # Save directory for next time
        if filename:
            self.settings.setValue("last_export_dir", os.path.dirname(filename))
        if filename:
            try:
                np.savetxt(filename, dataset.adjusted_data, 
                          header="Energy(keV)\tCounts", fmt="%.5f")
                self.statusBar().showMessage(f"Data exported to: {filename}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
                
    def restore_splitter_states(self, main_splitter, right_splitter):
        """Restore splitter states after UI initialization"""
        if self.settings.contains("mainSplitter"):
            main_splitter.restoreState(self.settings.value("mainSplitter"))
        else:
            main_splitter.setSizes([300, 900])  # Left panel 30%, Right panel 70%
            
        if self.settings.contains("rightSplitter"):
            right_splitter.restoreState(self.settings.value("rightSplitter"))
        else:
            right_splitter.setSizes([600, 300])  # Plot area 60%, Results 40%

    def closeEvent(self, event):
        """Handle window close event - save all settings"""
        # Save splitter positions
        if hasattr(self, 'settings'):
            mainSplitter = self.findChild(QSplitter, "mainSplitter")
            if mainSplitter:
                self.settings.setValue("mainSplitter", mainSplitter.saveState())
                
            rightSplitter = self.findChild(QSplitter, "rightSplitter")
            if rightSplitter:
                self.settings.setValue("rightSplitter", rightSplitter.saveState())
                
        # Save column widths
        self.save_column_widths()
        self.save_calib_column_widths()
        
        # Save energy calibration parameters (a, b, c)
        try:
            a = self.findChild(QLineEdit, "a_edit").text()
            b = self.findChild(QLineEdit, "b_edit").text()
            c = self.findChild(QLineEdit, "c_edit").text()
            self.settings.setValue("energy_calib/a", a)
            self.settings.setValue("energy_calib/b", b)
            self.settings.setValue("energy_calib/c", c)
        except:
            pass
            
        # Save calibration table data (d values and remarks)
        calib_data = []
        for row in range(self.calib_table.rowCount()):
            row_data = {}
            d_item = self.calib_table.item(row, 0)
            e_item = self.calib_table.item(row, 1)
            remarks_item = self.calib_table.item(row, 2)
            
            if d_item and d_item.text():
                row_data['d'] = d_item.text()
            if e_item and e_item.text():
                row_data['e'] = e_item.text()
            if remarks_item and remarks_item.text():
                row_data['remarks'] = remarks_item.text()
                
            if row_data:  # Only add if we have some data
                calib_data.append(row_data)
                
        self.settings.setValue("calib_table/data", calib_data)
        
        # Proceed with standard close event
        super().closeEvent(event)
        
    def import_data(self):
        """Import raw data file"""
        # Get last used directory from settings
        last_dir = self.settings.value("last_import_dir", "")
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", last_dir, "Text files (*.txt);;All files (*)")
        # Save directory for next time
        if filename:
            self.settings.setValue("last_import_dir", os.path.dirname(filename))
        if filename:
            try:
                # Load the data
                raw_data = np.loadtxt(filename)
                if raw_data.ndim != 2 or raw_data.shape[1] < 2:
                    raise ValueError("Data requires at least 2 columns")
                
                # Remove the last row as required
                if len(raw_data) > 0:
                    raw_data = raw_data[:-1]
                    self.statusBar().showMessage(f"Loaded data and removed last row", 2000)
                
                # Create new dataset
                dataset = Dataset(filename=filename, raw_data=raw_data)
                self.datasets.append(dataset)
                self.current_dataset_index = len(self.datasets) - 1
                
                # Reset Y-axis limits for new data
                if hasattr(self, '_y_limits'):
                    delattr(self, '_y_limits')
                
                # Update UI
                self.filename_label.setText(dataset.name)
                self.update_dataset_selector()
                self.plot_data()
                
                self.statusBar().showMessage(f"Successfully loaded: {filename}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Data loading failed: {str(e)}")

# Main application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    analyzer = XRDDataAnalyzer()
    analyzer.show()
    sys.exit(app.exec_())
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
                            QFileDialog, QMessageBox, QToolBar, QGroupBox, QScrollArea,
                            QSizePolicy, QHeaderView, QSplitter, QComboBox, QAction)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QDoubleValidator, QColor, QBrush, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import os

# Physical constants
HC_KEV_ANGSTROM = 12.39842  # h*c (keV·Å)

class Dataset:
    """Class to store dataset information"""
    def __init__(self, filename=None, raw_data=None):
        self.filename = filename
        self.raw_data = raw_data          # [channel, counts]
        self.adjusted_data = None         # [energy, counts]
        self.fit_results = []             # Peak fitting results
        self.selected_regions = []        # ROI storage
        self.x_axis_adjusted = False      # Flag if calibration was applied
        
        # Copy data as adjusted on init
        if raw_data is not None:
            self.adjusted_data = np.copy(raw_data)

    @property
    def name(self):
        """Get dataset display name"""
        if self.filename:
            return os.path.basename(self.filename)
        return "Unnamed Dataset"

class XRDDataAnalyzer(QMainWindow):
    # 样式常量
    BUTTON_STYLE_TPL = """
        QPushButton {{
            background-color: {};
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {};
        }}
    """
    
    GROUP_STYLE_TPL = "QGroupBox {{ border: 1px solid {}; font-weight: bold; padding-top: 15px; }}"
    
    # UI 颜色映射
    UI_COLORS = {
        'import': ('#1f77b4', '#4c9acd'),  # 主色, 悬停色
        'calibration': ('#2ca02c', '#5abb5a'),
        'export': ('#d62728', '#e85858'),
        'roi': ('#ff7f0e', '#e67e22'),
        'fit': ('#9467bd', '#b08be0'),
        'clear': ('#7f7f7f', '#a6a6a6'),
        'theta': ('#17becf', '#58d7e6'),
        'export_calib': ('#e377c2', '#f0a6d9'),
    }
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2theta fitting tool_SSRF BL12SW")
        self.setGeometry(100, 100, 1280, 960)
        
        # Initialize settings
        self.settings = QSettings("SSRF", "2theta")
        
        # Data storage - modified for multiple datasets
        self.datasets = []  # List to store multiple datasets
        self.current_dataset_index = -1  # Current active dataset index
        self.calib_params = None      # Calibration parameters
        
        # Color configuration for plots/ROIs - using vibrant colors
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Blue, Orange, Green, Red, Purple
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'   # Brown, Pink, Gray, Olive, Cyan
        ]
        
        # UI interaction state
        self.region_selection_active = False
        self.current_roi = None
        
        # Initialize UI
        self.init_ui()
        self.connect_signals()
        
        # Restore saved values
        self.restore_saved_values()
        
    @property
    def current_dataset(self):
        """Get current active dataset or None"""
        if 0 <= self.current_dataset_index < len(self.datasets):
            return self.datasets[self.current_dataset_index]
        return None
        
    @property
    def raw_data(self):
        """Legacy accessor for raw_data from current dataset"""
        dataset = self.current_dataset
        return dataset.raw_data if dataset else None
        
    @property
    def adjusted_data(self):
        """Legacy accessor for adjusted_data from current dataset"""
        dataset = self.current_dataset
        return dataset.adjusted_data if dataset else None
        
    @property
    def selected_regions(self):
        """Legacy accessor for selected_regions from current dataset"""
        dataset = self.current_dataset
        return dataset.selected_regions if dataset else []
        
    @property
    def fit_results(self):
        """Legacy accessor for fit_results from current dataset"""
        dataset = self.current_dataset
        return dataset.fit_results if dataset else []

    def init_ui(self):
        """Initialize user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Contributors label in top right
        contributors_label = QLabel("Contributors: Niu Guoliang (HPSTAR) & Zhou Chunyin (SSRF)")
        contributors_label.setStyleSheet("color: #666666; font-weight: bold;")
        contributors_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        main_layout.addWidget(contributors_label)
        
        # Top control area (30% height)
        top_group = QGroupBox("Data Control")
        top_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 15px; }")
        top_layout = QHBoxLayout(top_group)
        top_layout.setSpacing(20)
        top_layout.setContentsMargins(10, 20, 10, 10)
        main_layout.setStretchFactor(top_group, 3)  # 30% height
        
        # Data import section
        data_group = QGroupBox("Data Import")
        data_group.setStyleSheet("QGroupBox { border: 1px solid #1f77b4; }")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(10)
        
        # Dataset selector
        dataset_layout = QHBoxLayout()
        dataset_layout.setSpacing(10)
        self.dataset_selector = QComboBox()
        self.dataset_selector.setStyleSheet("""
            QComboBox {
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                min-width: 150px;
            }
        """)
        self.dataset_selector.setToolTip("Select active dataset")
        dataset_layout.addWidget(QLabel("Dataset:"))
        dataset_layout.addWidget(self.dataset_selector, 1)
        data_layout.addLayout(dataset_layout)
        
        self.import_btn = QPushButton("Import Data")
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #1f77b4;
                color: white;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.import_btn.setToolTip("Click to import XRD data file")
        self.filename_label = QLabel("No data loaded")
        self.filename_label.setStyleSheet("QLabel { color: #666666; }")
        data_layout.addWidget(self.import_btn)
        data_layout.addWidget(self.filename_label)
        top_layout.addWidget(data_group, 40)  # Set width ratio to 40%
        
        # Energy calibration section
        adj_group = QGroupBox("Energy Calibration (a*Ch² + b*Ch + c)")
        adj_group.setStyleSheet("QGroupBox { border: 1px solid #2ca02c; }")
        adj_layout = QVBoxLayout(adj_group)
        adj_layout.setSpacing(10)
        param_layout = QHBoxLayout()
        param_layout.setSpacing(10)
        for name, val in [("a", "0.0"), ("b", "0.03758"), ("c", "-0.04962")]:
            label = QLabel(f"{name}:")
            label.setStyleSheet("QLabel { font-weight: bold; }")
            edit = QLineEdit(val)
            edit.setObjectName(f"{name}_edit")
            edit.setValidator(QDoubleValidator(-100, 100, 8))
            edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    padding: 3px;
                    border-radius: 3px;
                }
            """)
            param_layout.addWidget(label)
            param_layout.addWidget(edit)
        
        self.adjust_btn = QPushButton("Apply Calibration")
        self.adjust_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ca02c;
                color: white;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.adjust_btn.setToolTip("Apply energy calibration parameters")
        
        self.export_btn = QPushButton("Export Data")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #d62728;
                color: white;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.export_btn.setToolTip("Export calibrated data")
        
        adj_layout.addLayout(param_layout)
        adj_layout.addWidget(self.adjust_btn)
        adj_layout.addWidget(self.export_btn)
        top_layout.addWidget(adj_group, 60)  # Set width ratio to 60%
        
        main_layout.addWidget(top_group)

        # Main workspace
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("mainSplitter")  # Set name for saving state
        
        # Left control panel (40% width)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # Peak analysis controls
        peak_group = QGroupBox("Peak Analysis")
        peak_group.setStyleSheet("QGroupBox { border: 1px solid #ff7f0e; font-weight: bold; padding-top: 15px; }")
        peak_layout = QVBoxLayout(peak_group)
        peak_layout.setSpacing(10)
        
        self.select_btn = QPushButton("Select ROI", checkable=True)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff7f0e;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #d35400;
                border: 2px solid #ff7f0e;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.select_btn.setToolTip("Select region of interest (ROI)")
        
        self.fit_btn = QPushButton("Fit Peaks")
        self.fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #9467bd;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.fit_btn.setToolTip("Perform Gaussian fit on selected ROI")
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f7f7f;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.clear_btn.setToolTip("Clear all ROIs and fit results")
        
        peak_layout.addWidget(self.select_btn)
        peak_layout.addWidget(self.fit_btn)
        peak_layout.addWidget(self.clear_btn)
        left_layout.addWidget(peak_group)
        
        # 2θ calibration inputs
        calib_group = QGroupBox("2θ Calibration")
        calib_group.setStyleSheet("QGroupBox { border: 1px solid #17becf; font-weight: bold; padding-top: 15px; }")
        calib_layout = QVBoxLayout(calib_group)
        calib_layout.setSpacing(10)
        
        self.calib_table = QTableWidget(9, 3)
        self.calib_table.setHorizontalHeaderLabels(["d (Å)", "E (keV)", "Remarks"])
        # Set default MgO values
        self.calib_table.setItem(0, 0, QTableWidgetItem("2.1065"))
        self.calib_table.setItem(0, 2, QTableWidgetItem("MgO_200"))
        self.calib_table.setItem(1, 0, QTableWidgetItem("1.4895"))
        self.calib_table.setItem(1, 2, QTableWidgetItem("MgO_220"))
        
        # Restore column widths if available
        if self.settings.contains("calibTable/columnWidths"):
            widths = self.settings.value("calibTable/columnWidths")
            for col, width in enumerate(widths):
                self.calib_table.setColumnWidth(col, width)
        else:
            # Set default widths
            self.calib_table.setColumnWidth(0, 120)  # d (Å)
            self.calib_table.setColumnWidth(1, 100)  # E (keV)
            self.calib_table.setColumnWidth(2, 200)  # Remarks
        
        # Save column widths when changed
        self.calib_table.horizontalHeader().sectionResized.connect(self.save_calib_column_widths)
        
        self.calib_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
            QHeaderView::section {
                background-color: #17becf;
                color: white;
                padding: 4px;
                font-weight: bold;
            }
        """)
        
        self.calibrate_btn = QPushButton("Calibrate 2θ")
        self.calibrate_btn.setStyleSheet("""
            QPushButton {
                background-color: #17becf;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        
        self.export_calib_btn = QPushButton("Export Calibration")
        self.export_calib_btn.setStyleSheet("""
            QPushButton {
                background-color: #e377c2;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        
        self.calib_status = QLabel("Status: Waiting for calibration")
        self.calib_status.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background-color: #f5f5f5;
            }
        """)
        
        calib_layout.addWidget(self.calib_table)
        calib_layout.addWidget(self.calibrate_btn)
        calib_layout.addWidget(self.export_calib_btn)
        calib_layout.addWidget(self.calib_status)
        left_layout.addWidget(calib_group)
        
        splitter.addWidget(left_panel)

        # Right display area with vertical splitter
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setObjectName("rightSplitter")  # Set name for saving state
        
        # Restore splitter states if available - do this after adding to layout
        if self.settings.contains("mainSplitter"):
            splitter.restoreState(self.settings.value("mainSplitter"))
        else:
            # Ensure splitters are properly initialized if no saved state
            splitter.setSizes([300, 900])  # Left panel 30%, Right panel 70%
            
        if self.settings.contains("rightSplitter"):
            right_splitter.restoreState(self.settings.value("rightSplitter"))
        else:
            # Ensure splitters are properly initialized if no saved state
            right_splitter.setSizes([600, 300])  # Plot area 60%, Results 40%
        
        # Plot area
        # Plot area widget
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(10, 6))
        self.figure.patch.set_facecolor('#f9f9f9')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#f9f9f9')
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        right_splitter.addWidget(plot_widget)
        
        # Results area widget
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        
        self.result_table = QTableWidget(0, 5)  # Start with 0 rows
        self.result_table.setHorizontalHeaderLabels([
            "ROI", "Center (keV)", "FWHM (keV)", "Intensity", "R²"
        ])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        
        # Disable auto-stretch to allow manual column resize
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        
        # Restore column widths if available
        if self.settings.contains("resultTable/columnWidths"):
            widths = self.settings.value("resultTable/columnWidths")
            for col, width in enumerate(widths):
                if col < self.result_table.columnCount():
                    self.result_table.setColumnWidth(col, int(width))
        else:
            # Set default widths
            self.result_table.setColumnWidth(0, 200)  # ROI
            self.result_table.setColumnWidth(1, 120)  # Center
            self.result_table.setColumnWidth(2, 120)  # FWHM
            self.result_table.setColumnWidth(3, 120)  # Intensity
            self.result_table.setColumnWidth(4, 80)   # R²
        
        # Save column widths when changed
        self.result_table.horizontalHeader().sectionResized.connect(self.save_column_widths)
        
        # Add table to scroll area to ensure scrollability when needed
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.result_table)
        results_layout.addWidget(scroll_area)
        right_splitter.addWidget(results_widget)
        
        # Set initial sizes for the splitters to show full table content
        right_splitter.setSizes([400, 600])  # Plot area 40%, Results 60%
        
        splitter.addWidget(right_splitter)
        
        # Configure splitter behavior
        splitter.setHandleWidth(8)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #e0e0e0;
            }
            QSplitter::handle:hover {
                background: #1f77b4;
            }
        """)
        splitter.setStretchFactor(0, 1)  # Left panel
        splitter.setStretchFactor(1, 3)  # Right panel
        splitter.setSizes([400, 800])  # Initial sizes (33%/67%)
        splitter.splitterMoved.connect(lambda: self.settings.setValue("mainSplitter", splitter.saveState()))
        
        # Configure right splitter
        right_splitter.setHandleWidth(8)
        right_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #e0e0e0;
            }
            QSplitter::handle:hover {
                background: #1f77b4;
            }
        """)
        right_splitter.setStretchFactor(0, 2)  # Plot area
        right_splitter.setStretchFactor(1, 1)  # Results area
        right_splitter.setSizes([500, 400])  # Initial sizes (55%/45%)
        right_splitter.splitterMoved.connect(lambda: self.settings.setValue("rightSplitter", right_splitter.saveState()))
        
        main_layout.addWidget(splitter)
        
        # Set size policy for resizing
        self.result_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_table.verticalHeader().setDefaultSectionSize(24)  # Compact row height
        self.result_table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        
        # Set contributor info in status bar
        self.statusBar().showMessage("Contributors: Niu Guoliang (HPSTAR) & Zhou Chunyin (SSRF) | Tip: Drag splitters to resize panels")

    def connect_signals(self):
        """Connect signals to slots"""
        self.import_btn.clicked.connect(self.import_data)
        self.adjust_btn.clicked.connect(self.adjust_energy_axis)
        self.export_btn.clicked.connect(self.export_data)
        self.select_btn.toggled.connect(self.toggle_roi_selection)
        self.fit_btn.clicked.connect(self.fit_peaks)
        self.clear_btn.clicked.connect(self.clear_all)
        self.calibrate_btn.clicked.connect(self.calibrate_2theta)
        self.export_calib_btn.clicked.connect(self.export_calibration)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_changed)
        
        # Plot interaction events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def on_dataset_changed(self, index):
        """Handle dataset selection change"""
        if 0 <= index < len(self.datasets):
            self.current_dataset_index = index
            dataset = self.current_dataset
            self.filename_label.setText(dataset.name if dataset else "No data loaded")
            
            # Update result table with current dataset's fit results
            self.update_result_table()
            
            # Refresh plot
            self.plot_data()
        
    def update_dataset_selector(self):
        """Update dataset selector with current datasets"""
        # Block signals to prevent triggering change events during update
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        
        for idx, dataset in enumerate(self.datasets):
            self.dataset_selector.addItem(dataset.name if dataset else f"Dataset {idx+1}")
        
        # Select current dataset
        if 0 <= self.current_dataset_index < self.dataset_selector.count():
            self.dataset_selector.setCurrentIndex(self.current_dataset_index)
            
        self.dataset_selector.blockSignals(False)
        
        # Update controls based on whether we have data
        has_data = len(self.datasets) > 0
        self.select_btn.setEnabled(has_data)
        self.fit_btn.setEnabled(has_data)
        self.adjust_btn.setEnabled(has_data)
        self.export_btn.setEnabled(has_data)
        
    def update_result_table(self):
        """Update result table with current dataset's fit results"""
        self.result_table.setRowCount(0)
        
        dataset = self.current_dataset
        if not dataset or not dataset.fit_results:
            return
            
        regions = dataset.selected_regions
        results = dataset.fit_results
        
        self.result_table.setRowCount(len(results))
        for idx, (region, result) in enumerate(zip(regions, results)):
            for col, val in enumerate([
                f"ROI {idx+1} ({region['x_min']:.2f}-{region['x_max']:.2f} keV)",
                f"{result['params'][1]:.5f}",
                f"{result['fwhm']:.5f}",
                f"{result['integral']:.5f}",
                f"{result['r_squared']:.5f}"
            ]):
                item = QTableWidgetItem(val)
                # Set ROI row style - background color matches plot, text color depends on background
                item.setBackground(QColor(region['color']))
                # Calculate background brightness to determine text color
                bg_color = QColor(region['color'])
                brightness = (bg_color.red() * 299 + bg_color.green() * 587 + bg_color.blue() * 114) / 1000
                text_color = QColor(Qt.black) if brightness > 128 else QColor(Qt.white)
                item.setForeground(text_color)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                
                # For other columns, set text color based on background
                if col > 0:
                    item.setForeground(text_color)
                
                # Ensure text remains visible when selected
                item.setData(Qt.BackgroundRole, QColor(region['color']))
                item.setData(Qt.ForegroundRole, text_color)
                
                self.result_table.setItem(idx, col, item)

    def save_column_widths(self):
        """Save result table column widths to settings"""
        widths = [self.result_table.columnWidth(i) for i in range(self.result_table.columnCount())]
        self.settings.setValue("resultTable/columnWidths", widths)
        
    def save_calib_column_widths(self):
        """Save calibration table column widths to settings"""
        widths = [self.calib_table.columnWidth(i) for i in range(self.calib_table.columnCount())]
        self.settings.setValue("calibTable/columnWidths", widths)
        
    def restore_saved_values(self):
        """Restore saved values from settings"""
        # Restore energy calibration parameters (a, b, c)
        if self.settings.contains("energy_calib/a"):
            a = self.settings.value("energy_calib/a")
            self.findChild(QLineEdit, "a_edit").setText(a)
        
        if self.settings.contains("energy_calib/b"):
            b = self.settings.value("energy_calib/b")
            self.findChild(QLineEdit, "b_edit").setText(b)
            
        if self.settings.contains("energy_calib/c"):
            c = self.settings.value("energy_calib/c")
            self.findChild(QLineEdit, "c_edit").setText(c)
            
        # Restore calibration table data
        if self.settings.contains("calib_table/data"):
            # Clear default values first
            for row in range(self.calib_table.rowCount()):
                for col in range(self.calib_table.columnCount()):
                    self.calib_table.setItem(row, col, QTableWidgetItem(""))
                    
            # Fill with saved values
            calib_data = self.settings.value("calib_table/data")
            if calib_data:
                for i, row_data in enumerate(calib_data):
                    if i >= self.calib_table.rowCount():
                        break
                        
                    if 'd' in row_data:
                        self.calib_table.setItem(i, 0, QTableWidgetItem(row_data['d']))
                    if 'e' in row_data:
                        self.calib_table.setItem(i, 1, QTableWidgetItem(row_data['e']))
                    if 'remarks' in row_data:
                        self.calib_table.setItem(i, 2, QTableWidgetItem(row_data['remarks']))
    
    def gaussian_with_baseline(self, x, a, x0, sigma, b, c):
        """Gaussian function with linear baseline"""
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b * x + c
        
    def plot_data(self):
        """Update plot with compact layout and ROI-based legend"""
        self.ax.clear()
        
        # Get current dataset
        dataset = self.current_dataset
        if not dataset:
            self.canvas.draw()
            return
            
        # Determine data source
        data = dataset.adjusted_data if dataset.x_axis_adjusted else dataset.raw_data
        if data is None:
            return
        
        # Store current Y limits before plotting, if they exist
        prev_y_min, prev_y_max = self.ax.get_ylim() if hasattr(self, '_y_limits') else (None, None)
        
        # Plot raw data
        self.ax.plot(data[:,0], data[:,1], 'k-', lw=1, label='Raw data')
        
        # Get current Y limits if we don't have stored ones
        if not hasattr(self, '_y_limits'):
            y_min, y_max = self.ax.get_ylim()
            # Store these limits for future use
            self._y_limits = (y_min, y_max)
        else:
            # Use stored limits
            y_min, y_max = self._y_limits

        # Explicitly set Y limits to ensure consistency
        self.ax.set_ylim(y_min, y_max)
        
        # Plot ROI regions and fits based on their state
        legend_handles = []
        legend_labels = []
        
        # Add raw data to legend
        line, = self.ax.plot([], [], 'k-', lw=1)
        legend_handles.append(line)
        legend_labels.append('Raw data')
        
        for i, roi in enumerate(dataset.selected_regions):
            # Check if this ROI has been fitted
            is_fitted = i < len(dataset.fit_results)
            
            if not is_fitted:
                # Show region box and label for unfitted ROIs
                rect = Rectangle(
                    (roi['x_min'], y_min),
                    roi['width'], y_max - y_min,
                    facecolor=roi['color'], alpha=0.3,
                    edgecolor=roi['color'], lw=1
                )
                # Store rect reference for clear_all
                roi['rect'] = rect
                self.ax.add_patch(rect)
                self.ax.text(roi['x_min'] + roi['width']/2, 
                            y_max,
                            f"Region {i+1} ({roi['x_min']:.2f}-{roi['x_max']:.2f} keV)",
                            color=roi['color'], ha='center', va='top')
                
                # Add to legend
                patch = Rectangle((0,0), 1, 1, facecolor=roi['color'], alpha=0.3)
                legend_handles.append(patch)
                legend_labels.append(f'Region {i+1} ({roi["x_min"]:.2f}-{roi["x_max"]:.2f} keV)')
            else:
                # Show fit line for fitted ROIs
                result = dataset.fit_results[i]
                x_fit = np.linspace(roi['x_min'], roi['x_max'], 100)
                y_fit = self.gaussian_with_baseline(x_fit, *result['params'])
                
                line_color = roi['color']
                line, = self.ax.plot(x_fit, y_fit, '--', color=line_color, lw=1.5)
                
                # Add to legend
                legend_handles.append(line)
                legend_labels.append(f'Fit {i+1} ({roi["x_min"]:.2f}-{roi["x_max"]:.2f} keV)')
                
                # Annotate center and FWHM
                center = result['params'][1]
                fwhm = result['fwhm']
                self.ax.axvline(x=center, color=line_color, linestyle=':', alpha=0.7)
                self.ax.axvspan(center - fwhm/2, center + fwhm/2, 
                               color=line_color, alpha=0.1)
        
        # Set axis labels and legend
        self.ax.set_xlabel('Energy (keV)' if dataset.x_axis_adjusted else 'Channel')
        self.ax.set_ylabel('Intensity')
        
        if legend_handles:
            self.ax.legend(legend_handles, legend_labels, loc='upper right', 
                         framealpha=0.7, fancybox=True)
        
        # Refresh canvas
        self.canvas.draw()
