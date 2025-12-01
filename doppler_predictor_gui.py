"""
Doppler Predictor GUI for Satellite Communications
GUI application for real-time Starlink satellite tracking and Doppler visualization.
Uses PyQt5 for better macOS compatibility.

This is a standalone version with integrated predictor classes.
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Doppler Predictor Classes (integrated from doppler_predictor2.py)
# ============================================================================

class DopplerPredictor:
    """
    Predicts Doppler shift for satellites using TLE data.
    X-band reference frequency: 10.5 GHz (Starlink downlink)
    """
    
    # Constants
    EARTH_RADIUS_KM = 6371.0  # km
    STARLINK_TX_FREQ = 10.5e9  # Hz (10.5 GHz - Starlink transmit frequency)
    SPEED_OF_LIGHT = 299792458  # m/s
    RX_BAND_START = 10e9  # Hz (10 GHz - UE receive band start)
    RX_BAND_END = 11e9  # Hz (11 GHz - UE receive band end)
    
    def __init__(self, tle_line1: str, tle_line2: str, ue_location: dict):
        """
        Initialize the Doppler predictor.
        
        Args:
            tle_line1: First line of TLE data
            tle_line2: Second line of TLE data
            ue_location: Dict with keys 'latitude' (deg), 'longitude' (deg), 'altitude' (m)
        """
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        
        # Parse TLE to extract satellite name
        self.sat_name = tle_line1.strip()
        
        # Store UE location
        self.ue_lat = ue_location.get('latitude', 0)
        self.ue_lon = ue_location.get('longitude', 0)
        self.ue_alt = ue_location.get('altitude', 0) / 1000  # Convert to km
        
        # Import SGP4 for TLE propagation
        try:
            from skyfield.api import EarthSatellite, load, wgs84
            self.skyfield = True
            self.wgs84 = wgs84
            
            # Create satellite object
            self.satellite = EarthSatellite(self.tle_line1, self.tle_line2)
            self.ts = load.timescale()
            
        except ImportError:
            self.skyfield = False
            print("Warning: Skyfield not installed. Install with: pip install skyfield")
    
    def calculate_doppler_shift(self, obs_time: datetime = None) -> float:
        """
        Calculate Doppler shift in Hz for given observation time.
        
        Args:
            obs_time: Observation time (datetime). If None, uses current time.
            
        Returns:
            Doppler shift in Hz
        """
        if not self.skyfield:
            return 0.0
        
        if obs_time is None:
            obs_time = datetime.utcnow()
        
        try:
            from skyfield.api import utc
            
            # Convert to Skyfield time
            ts_time = self.ts.from_datetime(obs_time.replace(tzinfo=utc) if obs_time.tzinfo is None else obs_time)
            
            # Create observer location using wgs84.latlon
            observer_location = self.wgs84.latlon(
                self.ue_lat,
                self.ue_lon,
                elevation_m=self.ue_alt * 1000
            )
            
            # Get satellite position relative to observer at two times
            relative_now = (self.satellite - observer_location).at(ts_time)
            
            # Calculate at slightly later time for velocity
            dt_seconds = 1.0
            ts_time_plus = self.ts.from_datetime(
                datetime.utcfromtimestamp(obs_time.timestamp() + dt_seconds).replace(tzinfo=utc)
            )
            relative_later = (self.satellite - observer_location).at(ts_time_plus)
            
            # Get distance at both times
            dist_now_au = (relative_now.position.au[0]**2 + 
                          relative_now.position.au[1]**2 + 
                          relative_now.position.au[2]**2)**0.5
            
            dist_later_au = (relative_later.position.au[0]**2 + 
                            relative_later.position.au[1]**2 + 
                            relative_later.position.au[2]**2)**0.5
            
            # Range rate in AU/second
            range_rate_au_per_s = (dist_later_au - dist_now_au) / dt_seconds
            
            # Convert to m/s (1 AU = 150e9 m)
            range_rate_m_per_s = range_rate_au_per_s * 150e9
            
            # Calculate Doppler shift: f' = f * (c - v_r) / c
            # For recession (positive range_rate): frequency decreases (negative shift)
            doppler_shift = -self.STARLINK_TX_FREQ * range_rate_m_per_s / self.SPEED_OF_LIGHT
            
            return doppler_shift
            
        except Exception as e:
            return 0.0


class MultiSatellitePredictor:
    """
    Predicts combined Doppler spectrum from multiple Starlink satellites.
    """
    
    def __init__(self, tle_data: str, ue_location: dict, num_satellites: int = 20):
        """
        Initialize with multiple satellite TLEs.
        
        Args:
            tle_data: TLE data string from CelesTrak (multiple 3-line TLE sets)
            ue_location: Dict with keys 'latitude', 'longitude', 'altitude'
            num_satellites: Number of satellites to use from TLE data
        """
        self.ue_location = ue_location
        self.predictors = []
        
        # Parse TLE data into groups of 3 lines
        tle_lines = tle_data.strip().split('\n')
        
        # Process TLE data (groups of 3 lines: name, line1, line2)
        sat_count = 0
        i = 0
        while i < len(tle_lines) - 2 and sat_count < num_satellites:
            tle_name = tle_lines[i].strip()
            tle_line1 = tle_lines[i + 1].strip()
            tle_line2 = tle_lines[i + 2].strip()
            
            # Validate TLE format
            if tle_line1.startswith('1 ') and tle_line2.startswith('2 '):
                try:
                    predictor = DopplerPredictor(tle_line1, tle_line2, ue_location)
                    if predictor.skyfield:
                        self.predictors.append(predictor)
                        sat_count += 1
                except:
                    pass
            
            i += 3
        
        print(f"Loaded {len(self.predictors)} Starlink satellites")


# ============================================================================
# GUI Code
# ============================================================================

# Try different GUI backends
def try_pyqt5():
    """Try PyQt5-based GUI."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                                  QGroupBox, QFormLayout, QFileDialog, QMessageBox,
                                  QStatusBar, QSplitter, QFrame)
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QFont
    
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import threading
    
    class DopplerPredictorGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Starlink Doppler Predictor")
            self.setGeometry(100, 100, 1200, 800)
            
            # Initialize variables
            self.multi_predictor = None
            self.timer = None
            self.visibility_timer = None  # Slower timer for checking visibility of out-of-view satellites
            self.waterfall_timer = None   # Timer for waterfall animation
            self.tle_data = None
            self.is_running = False
            self.waterfall_running = False
            
            # Track which satellites are currently visible (for efficient updates)
            self.visible_predictors = []  # Satellites currently in view (updated frequently)
            self.hidden_predictors = []   # Satellites not in view (checked less often)
            
            # Waterfall data storage
            self.waterfall_data = None
            self.waterfall_times = None
            
            self.setup_ui()
            self.auto_load_tle()
            
        def setup_ui(self):
            """Set up the GUI layout."""
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            main_layout = QHBoxLayout(central_widget)
            
            # Left panel for controls
            control_panel = self.create_control_panel()
            main_layout.addWidget(control_panel)
            
            # Right panel for visualization
            viz_panel = self.create_visualization_panel()
            main_layout.addWidget(viz_panel, stretch=1)
            
            # Status bar
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage("Ready. Load TLE data to begin.")
            
        def create_control_panel(self):
            """Create the control panel."""
            panel = QFrame()
            panel.setFrameStyle(QFrame.StyledPanel)
            panel.setMaximumWidth(300)
            layout = QVBoxLayout(panel)
            
            # TLE Data Section
            tle_group = QGroupBox("TLE Data")
            tle_layout = QVBoxLayout(tle_group)
            
            self.load_btn = QPushButton("Load TLE File")
            self.load_btn.clicked.connect(self.load_tle_file)
            tle_layout.addWidget(self.load_btn)
            
            self.download_btn = QPushButton("Download Latest TLE")
            self.download_btn.clicked.connect(self.download_tle)
            tle_layout.addWidget(self.download_btn)
            
            layout.addWidget(tle_group)
            
            # Location Section
            loc_group = QGroupBox("Ground Station")
            loc_layout = QFormLayout(loc_group)
            
            self.lat_input = QLineEdit("47.6550")
            self.lon_input = QLineEdit("-122.3035")
            self.alt_input = QLineEdit("60")
            
            loc_layout.addRow("Latitude (°N):", self.lat_input)
            loc_layout.addRow("Longitude (°E):", self.lon_input)
            loc_layout.addRow("Altitude (m):", self.alt_input)
            
            layout.addWidget(loc_group)
            
            # Settings Section
            settings_group = QGroupBox("Settings")
            settings_layout = QFormLayout(settings_group)
            
            self.elev_input = QLineEdit("10.0")
            self.num_sats_input = QLineEdit("1000")
            self.interval_input = QLineEdit("100")
            self.duration_input = QLineEdit("5")
            self.wf_update_input = QLineEdit("5")
            
            settings_layout.addRow("Elevation Mask (°):", self.elev_input)
            settings_layout.addRow("Max Satellites:", self.num_sats_input)
            settings_layout.addRow("Sky Map Update (ms):", self.interval_input)
            settings_layout.addRow("Waterfall Duration (min):", self.duration_input)
            settings_layout.addRow("Waterfall Update (sec):", self.wf_update_input)
            
            # Update button for settings
            self.update_settings_btn = QPushButton("🔄 Update Settings")
            self.update_settings_btn.clicked.connect(self.apply_settings_update)
            self.update_settings_btn.setToolTip("Apply changes to location and settings")
            settings_layout.addRow(self.update_settings_btn)
            
            layout.addWidget(settings_group)
            
            # Action Buttons
            action_group = QGroupBox("Visualization")
            action_layout = QVBoxLayout(action_group)
            
            self.start_btn = QPushButton("▶ Start Sky Map")
            self.start_btn.clicked.connect(self.start_sky_map)
            action_layout.addWidget(self.start_btn)
            
            self.stop_btn = QPushButton("■ Stop")
            self.stop_btn.clicked.connect(self.stop_animation)
            self.stop_btn.setEnabled(False)
            action_layout.addWidget(self.stop_btn)
            
            self.waterfall_btn = QPushButton("▶ Start Live Waterfall")
            self.waterfall_btn.clicked.connect(self.start_live_waterfall)
            action_layout.addWidget(self.waterfall_btn)
            
            self.stop_waterfall_btn = QPushButton("■ Stop Waterfall")
            self.stop_waterfall_btn.clicked.connect(self.stop_waterfall)
            self.stop_waterfall_btn.setEnabled(False)
            action_layout.addWidget(self.stop_waterfall_btn)
            
            self.save_btn = QPushButton("Save Current View")
            self.save_btn.clicked.connect(self.save_figure)
            action_layout.addWidget(self.save_btn)
            
            layout.addWidget(action_group)
            
            # Statistics
            stats_group = QGroupBox("Statistics")
            stats_layout = QVBoxLayout(stats_group)
            
            self.tle_count_label = QLabel("TLE in file: 0")
            self.sats_label = QLabel("Loaded: 0")
            self.visible_label = QLabel("Visible: 0")
            
            stats_layout.addWidget(self.tle_count_label)
            stats_layout.addWidget(self.sats_label)
            stats_layout.addWidget(self.visible_label)
            
            layout.addWidget(stats_group)
            
            layout.addStretch()
            
            return panel
            
        def create_visualization_panel(self):
            """Create the visualization panel."""
            panel = QFrame()
            panel.setFrameStyle(QFrame.StyledPanel)
            layout = QVBoxLayout(panel)
            
            # Create matplotlib figure with space for inset map
            self.fig = Figure(figsize=(8, 7), dpi=100)
            # Position polar plot to the right to make room for inset map
            # [left, bottom, width, height] in figure coordinates (0-1)
            self.ax = self.fig.add_axes([0.35, 0.1, 0.6, 0.8], projection='polar')
            self.setup_polar_plot()
            
            # Add inset map showing UE location
            self.create_location_inset()
            
            # Canvas
            self.canvas = FigureCanvas(self.fig)
            layout.addWidget(self.canvas)
            
            # Toolbar
            self.toolbar = NavigationToolbar(self.canvas, panel)
            layout.addWidget(self.toolbar)
            
            return panel
        
        def create_location_inset(self):
            """Create a small inset map showing the ground station location."""
            # Create inset axes in the left side (larger map now that polar plot is shifted right)
            self.inset_ax = self.fig.add_axes([0.07, 0.05, 0.30, 0.35])  # [left, bottom, width, height]
            self.update_location_inset()
        
        def update_location_inset(self):
            """Update the inset map with current UE location."""
            if not hasattr(self, 'inset_ax'):
                return
                
            self.inset_ax.clear()
            
            try:
                lat = float(self.lat_input.text())
                lon = float(self.lon_input.text())
            except:
                lat, lon = 47.655, -122.3035  # Default
            
            import numpy as np
            
            # Try to use cartopy for proper map, fallback to simple version
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                
                # Remove old inset and create new one with cartopy projection
                self.inset_ax.remove()
                self.inset_ax = self.fig.add_axes([0.05, 0.05, 0.25, 0.35], 
                                                   projection=ccrs.PlateCarree())
                
                # Set regional extent centered on UE (±10° lat/lon for closer zoom)
                lon_range = 3
                lat_range = 3
                self.inset_ax.set_extent([lon - lon_range, lon + lon_range, 
                                          max(-90, lat - lat_range), min(90, lat + lat_range)],
                                         crs=ccrs.PlateCarree())
                
                self.inset_ax.add_feature(cfeature.OCEAN, color='#1a1a2e')
                self.inset_ax.add_feature(cfeature.LAND, color='#2d4a3e')
                self.inset_ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='white')
                self.inset_ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
                self.inset_ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray', alpha=0.3)
                
                # Plot UE location (red cross to match sky map)
                self.inset_ax.plot(lon, lat, 'r+', markersize=12, markeredgewidth=2,
                                  transform=ccrs.PlateCarree(), zorder=10)
                
                # Visibility circle (~2500 km radius ≈ 22°)
                circle_angles = np.linspace(0, 2*np.pi, 100)
                vis_radius = 22  # degrees, approximate visibility range
                circle_lats = lat + vis_radius * np.cos(circle_angles)
                circle_lons = lon + vis_radius / np.cos(np.radians(lat)) * np.sin(circle_angles)
                self.inset_ax.plot(circle_lons, circle_lats, 'c-', linewidth=1.5, alpha=0.7,
                                  transform=ccrs.PlateCarree(), label='Visibility')
                self.inset_ax.fill(circle_lons, circle_lats, color='cyan', alpha=0.1,
                                  transform=ccrs.PlateCarree())
                
                # Add gridlines
                gl = self.inset_ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, 
                                             color='gray', linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 6}
                gl.ylabel_style = {'size': 6}
                
                self.inset_ax.set_title(f'UE: {lat:.2f}°, {lon:.2f}°', fontsize=8, pad=2)
                
            except ImportError:
                # Fallback: simple plot without cartopy
                lon_range = 10
                lat_range = 8
                self.inset_ax.set_xlim(lon - lon_range, lon + lon_range)
                self.inset_ax.set_ylim(max(-90, lat - lat_range), min(90, lat + lat_range))
                self.inset_ax.set_facecolor('#1a1a2e')
                
                # Plot UE location (red cross to match sky map)
                self.inset_ax.plot(lon, lat, 'r+', markersize=12, markeredgewidth=2, zorder=10)
                
                # Visibility circle
                circle_angles = np.linspace(0, 2*np.pi, 100)
                vis_radius = 22
                circle_lats = lat + vis_radius * np.cos(circle_angles)
                circle_lons = lon + vis_radius / np.cos(np.radians(lat)) * np.sin(circle_angles)
                self.inset_ax.plot(circle_lons, circle_lats, 'c-', linewidth=1.5, alpha=0.7)
                self.inset_ax.fill(circle_lons, circle_lats, color='cyan', alpha=0.1)
                
                self.inset_ax.tick_params(labelsize=6)
                self.inset_ax.set_title(f'UE: {lat:.2f}°, {lon:.2f}°', fontsize=8, pad=2)
                self.inset_ax.grid(True, alpha=0.3, linewidth=0.3)
                self.inset_ax.text(lon, lat - lat_range + 3, 'Install cartopy for map', fontsize=5, 
                                  ha='center', color='yellow', alpha=0.7)
            
        def setup_polar_plot(self):
            """Initialize the polar plot."""
            self.ax.clear()
            self.ax.set_theta_zero_location('N')
            self.ax.set_theta_direction(-1)
            
            # Plot zenith marker at center (directly overhead the ground station) - red cross
            self.ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=3, label='Zenith (UE location)', zorder=10)
            self.ax.scatter([0], [0], c='red', s=100, marker='o', alpha=0.3, zorder=9)
            
            # Initialize empty scatter plots
            self.satellite_glow = self.ax.scatter([], [], c='yellow', s=150, marker='o', alpha=0.3, zorder=4)
            self.satellite_scatter = self.ax.scatter([], [], c='white', s=50, marker='o', zorder=5,
                                                      edgecolors='red', linewidths=0.5, label='Starlink Satellites')
            
            # Set up axes - center is 90° elevation (zenith), edge is 0° elevation (horizon)
            # r = 90 - elevation, so r=0 means 90° elev (center), r=90 means 0° elev (edge)
            self.ax.set_ylim(0, 90)
            self.ax.set_rscale('linear')  # Must be set BEFORE yticklabels!
            self.ax.grid(True, alpha=0.4, linestyle='--')
            
            # Now set ticks and labels (after rscale and grid)
            self.ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
            self.ax.set_yticklabels(['90°', '80°', '70°', '60°', '50°', '40°', '30°', '20°', '10°', '0°'], fontsize=9)
            self.ax.set_rlabel_position(22.5)  # Position the radial labels
            
            # Set azimuth ticks: cardinal (N, E, S, W) and intercardinal (NE, SE, SW, NW)
            az_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # 0° = North
            az_rad = np.radians(az_deg)
            self.ax.set_xticks(az_rad)
            self.ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=12, fontweight='bold')
            
            self.ax.set_title('Sky Map - Load TLE data to begin', fontsize=12, pad=10)
            self.ax.legend(loc='upper right', bbox_to_anchor=(-0.1, 1.05), fontsize=8)


            
        def auto_load_tle(self):
            """Auto-load starlink.txt if it exists."""
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'starlink.txt')
            if os.path.exists(default_path):
                try:
                    with open(default_path, 'r') as f:
                        self.tle_data = f.read()
                    tle_lines = self.tle_data.strip().split('\n')
                    num_sats = len([line for line in tle_lines if line.startswith('1 ')])
                    self.status_bar.showMessage(f"Auto-loaded {num_sats} TLEs from starlink.txt")
                    self.tle_count_label.setText(f"TLE in file: {num_sats}")
                    self.initialize_predictor()
                except:
                    pass
                    
        def load_tle_file(self):
            """Load TLE data from file."""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Select TLE File", 
                os.path.dirname(os.path.abspath(__file__)),
                "Text files (*.txt);;TLE files (*.tle);;All files (*.*)"
            )
            
            if filepath:
                try:
                    with open(filepath, 'r') as f:
                        self.tle_data = f.read()
                    
                    tle_lines = self.tle_data.strip().split('\n')
                    num_sats = len([line for line in tle_lines if line.startswith('1 ')])
                    
                    self.status_bar.showMessage(f"Loaded {num_sats} TLEs from {os.path.basename(filepath)}")
                    self.tle_count_label.setText(f"TLE in file: {num_sats}")
                    self.initialize_predictor()
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load TLE file: {e}")
                    
        def download_tle(self):
            """Download latest TLE data."""
            self.status_bar.showMessage("Downloading TLE data...")
            
            def download_thread():
                try:
                    import requests
                    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    self.tle_data = response.text
                    
                    tle_lines = self.tle_data.strip().split('\n')
                    num_sats = len([line for line in tle_lines if line.startswith('1 ')])
                    
                    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'starlink_downloaded.txt')
                    with open(save_path, 'w') as f:
                        f.write(self.tle_data)
                    
                    # Update UI from main thread
                    QTimer.singleShot(0, lambda: self.status_bar.showMessage(f"Downloaded {num_sats} TLEs"))
                    QTimer.singleShot(0, lambda: self.tle_count_label.setText(f"TLE in file: {num_sats}"))
                    QTimer.singleShot(0, self.initialize_predictor)
                    
                except Exception as e:
                    QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", f"Download failed: {e}"))
                    
            threading.Thread(target=download_thread, daemon=True).start()
            
        def initialize_predictor(self):
            """Initialize the predictor."""
            if not self.tle_data:
                return
                
            try:
                ue_location = {
                    'latitude': float(self.lat_input.text()),
                    'longitude': float(self.lon_input.text()),
                    'altitude': float(self.alt_input.text())
                }
                
                num_sats = int(self.num_sats_input.text())
                
                self.status_bar.showMessage("Initializing predictor...")
                
                self.multi_predictor = MultiSatellitePredictor(self.tle_data, ue_location, num_satellites=num_sats)
                
                self.status_bar.showMessage(f"Ready. {len(self.multi_predictor.predictors)} satellites loaded.")
                self.sats_label.setText(f"Loaded: {len(self.multi_predictor.predictors)}")
                
                # Update the location inset map
                self.update_location_inset()
                self.canvas.draw_idle()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize: {e}")
        
        def apply_settings_update(self):
            """Apply updated settings from the input fields."""
            if not self.tle_data:
                QMessageBox.warning(self, "Warning", "Please load TLE data first.")
                return
            
            # Remember if animation was running
            was_running = self.is_running
            was_waterfall_running = self.waterfall_running
            
            # Stop any running animations
            if self.is_running:
                self.stop_animation()
            if self.waterfall_running:
                self.stop_waterfall()
            
            # Reinitialize predictor with new settings
            self.initialize_predictor()
            
            # Update timer interval if sky map was running
            if was_running and self.multi_predictor:
                self.start_sky_map()
            
            # Restart waterfall if it was running
            if was_waterfall_running and self.multi_predictor:
                self.start_live_waterfall()
            
            self.status_bar.showMessage("Settings updated successfully.")
                
        def start_sky_map(self):
            """Start the real-time animation."""
            if not self.multi_predictor:
                QMessageBox.warning(self, "Warning", "Please load TLE data first.")
                return
                
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Initialize all satellites as hidden (will be checked on first visibility update)
            self.visible_predictors = []
            self.hidden_predictors = list(self.multi_predictor.predictors)
                
            self.setup_polar_plot()
            
            # Create fast timer for updating visible satellites
            interval = int(self.interval_input.text())
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_visible_satellites)
            self.timer.start(interval)
            
            # Create slow timer for checking visibility of hidden satellites
            self.visibility_timer = QTimer()
            self.visibility_timer.timeout.connect(self.check_hidden_satellites)
            self.visibility_timer.start(10000)  # Check every 10 seconds
            
            # Do initial full scan
            self.check_hidden_satellites()
            
            self.status_bar.showMessage("Sky map running...")
            
        def update_sky_map(self):
            """Update the sky map - DEPRECATED, kept for compatibility."""
            self.update_visible_satellites()
        
        def calculate_satellite_position(self, predictor, current_time_utc, elevation_mask):
            """Calculate position for a single satellite. Returns (is_visible, theta, r) or (False, None, None)."""
            try:
                ts_time = predictor.ts.from_datetime(current_time_utc)
                
                observer_location = predictor.wgs84.latlon(
                    predictor.ue_lat,
                    predictor.ue_lon,
                    elevation_m=predictor.ue_alt * 1000
                )
                
                relative = (predictor.satellite - observer_location).at(ts_time)
                xyz = relative.position.au
                magnitude = (xyz[0]**2 + xyz[1]**2 + xyz[2]**2)**0.5
                
                horizontal_dist = np.sqrt(xyz[0]**2 + xyz[1]**2)
                
                if magnitude > 0:
                    elevation_rad = np.arctan2(xyz[2], horizontal_dist)
                    elevation_deg = np.degrees(elevation_rad)
                else:
                    elevation_deg = -90
                
                azimuth_rad = np.arctan2(xyz[0], xyz[1])
                azimuth_deg = np.degrees(azimuth_rad)
                if azimuth_deg < 0:
                    azimuth_deg += 360
                
                if elevation_deg >= elevation_mask:
                    theta_rad = np.radians(azimuth_deg)
                    r = 90 - elevation_deg
                    return (True, theta_rad, r)
                else:
                    return (False, None, None)
            except:
                return (False, None, None)
        
        def check_hidden_satellites(self):
            """Check hidden satellites for visibility changes (runs every 10 seconds)."""
            if not self.is_running:
                return
            
            from skyfield.api import utc
            
            elevation_mask = float(self.elev_input.text())
            current_time_utc = datetime.utcnow().replace(tzinfo=utc)
            
            # Check all hidden satellites
            newly_visible = []
            still_hidden = []
            
            for predictor in self.hidden_predictors:
                is_visible, theta, r = self.calculate_satellite_position(predictor, current_time_utc, elevation_mask)
                if is_visible:
                    newly_visible.append(predictor)
                else:
                    still_hidden.append(predictor)
            
            # Also check currently visible satellites to see if any went out of view
            still_visible = []
            newly_hidden = []
            
            for predictor in self.visible_predictors:
                is_visible, theta, r = self.calculate_satellite_position(predictor, current_time_utc, elevation_mask)
                if is_visible:
                    still_visible.append(predictor)
                else:
                    newly_hidden.append(predictor)
            
            # Update the lists
            self.visible_predictors = still_visible + newly_visible
            self.hidden_predictors = still_hidden + newly_hidden
            
            # Update display immediately after visibility check
            self.update_visible_satellites()
        
        def update_visible_satellites(self):
            """Update positions of only visible satellites (runs frequently)."""
            if not self.is_running:
                return
                
            from skyfield.api import utc
            import numpy as np
            
            elevation_mask = float(self.elev_input.text())
            current_time_utc = datetime.utcnow().replace(tzinfo=utc)
            
            satellites_visible = []
            
            # Only update satellites that are already known to be visible
            for predictor in self.visible_predictors:
                is_visible, theta, r = self.calculate_satellite_position(predictor, current_time_utc, elevation_mask)
                if is_visible:
                    satellites_visible.append({'theta': theta, 'r': r})
            
            # Update scatter plots
            if satellites_visible:
                thetas = [s['theta'] for s in satellites_visible]
                rs = [s['r'] for s in satellites_visible]
                offsets = np.c_[thetas, rs]
                self.satellite_scatter.set_offsets(offsets)
                self.satellite_glow.set_offsets(offsets)
            else:
                self.satellite_scatter.set_offsets(np.c_[[], []])
                self.satellite_glow.set_offsets(np.c_[[], []])
            
            # Update title
            current_time_aware = current_time_utc.astimezone()
            current_time_str = current_time_aware.strftime('%Y-%m-%d %H:%M:%S %Z')
            num_visible = len(self.visible_predictors)
            num_hidden = len(self.hidden_predictors)
            self.ax.set_title(f'Sky Map - Starlink Satellites (LIVE)\n{current_time_str}\n({num_visible} visible, {num_hidden} hidden, Elevation > {elevation_mask}°)',
                             fontsize=11, pad=-20)
            
            self.visible_label.setText(f"Visible: {num_visible}")
            self.canvas.draw_idle()
            
        def stop_animation(self):
            """Stop the animation."""
            self.is_running = False
            if self.timer:
                self.timer.stop()
                self.timer = None
            if self.visibility_timer:
                self.visibility_timer.stop()
                self.visibility_timer = None
                
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("Animation stopped.")
        
        def start_live_waterfall(self):
            """Start live updating waterfall display."""
            if not self.multi_predictor:
                QMessageBox.warning(self, "Warning", "Please load TLE data first.")
                return
            
            from skyfield.api import utc
            
            self.waterfall_running = True
            self.waterfall_btn.setEnabled(False)
            self.stop_waterfall_btn.setEnabled(True)
            
            # Get settings
            self.wf_elevation_mask = float(self.elev_input.text())
            self.wf_duration_minutes = float(self.duration_input.text())
            
            # Get frequency info
            predictor = self.multi_predictor.predictors[0]
            self.wf_tx_hz = predictor.STARLINK_TX_FREQ
            self.wf_tx_ghz = self.wf_tx_hz / 1e9
            self.wf_doppler_range_hz = 500e3  # ±500 kHz
            self.wf_freq_start_hz = self.wf_tx_hz - self.wf_doppler_range_hz
            self.wf_freq_end_hz = self.wf_tx_hz + self.wf_doppler_range_hz
            
            # Waterfall dimensions
            self.wf_n_freq_bins = 300
            self.wf_n_time_samples = max(int(self.wf_duration_minutes * 6), 30)  # ~10 sec per row
            self.wf_sigma = 5e3  # 5 kHz
            
            # Create frequency grid
            self.wf_freq_grid_hz = np.linspace(self.wf_freq_start_hz, self.wf_freq_end_hz, self.wf_n_freq_bins)
            
            # Initialize waterfall with high path loss values (no signal = high loss)
            self.waterfall_data = np.full((self.wf_n_time_samples, self.wf_n_freq_bins), 250.0)
            self.waterfall_times = []
            
            # Create waterfall window
            self.waterfall_window = QMainWindow(self)
            self.waterfall_window.setWindowTitle("Live Doppler Waterfall")
            self.waterfall_window.setGeometry(150, 150, 1000, 700)
            
            central = QWidget()
            self.waterfall_window.setCentralWidget(central)
            layout = QVBoxLayout(central)
            
            # Create figure
            self.wf_fig = Figure(figsize=(12, 8), dpi=100)
            self.wf_ax = self.wf_fig.add_subplot(111)
            
            # Initial plot - use absolute frequency in GHz
            freq_start_ghz = self.wf_freq_start_hz / 1e9
            freq_end_ghz = self.wf_freq_end_hz / 1e9
            self.wf_extent = [freq_start_ghz, freq_end_ghz, self.wf_duration_minutes, 0]
            
            self.wf_im = self.wf_ax.imshow(self.waterfall_data, aspect='auto', extent=self.wf_extent,
                                           cmap='jet_r', vmin=170, vmax=185,
                                           interpolation='bilinear')
            
            self.wf_cbar = self.wf_fig.colorbar(self.wf_im, ax=self.wf_ax, pad=0.02)
            self.wf_cbar.set_label('Free Space Path Loss (dB)', fontsize=11)
            
            # Mark TX frequency (no Doppler)
            self.wf_ax.axvline(x=self.wf_tx_ghz, color='white', linestyle='--', linewidth=1.5, alpha=0.8,
                              label=f'TX: {self.wf_tx_ghz:.3f} GHz')
            self.wf_ax.set_xlabel('Frequency (GHz)', fontsize=12)
            self.wf_ax.set_ylabel('Time ago (minutes)', fontsize=12)
            self.wf_title = self.wf_ax.set_title(f'Live Doppler Waterfall - Path Loss\nInitializing...', fontsize=12)
            self.wf_ax.legend(loc='upper right', fontsize=9)
            
            # Format x-axis to show full GHz values without offset notation
            from matplotlib.ticker import FuncFormatter
            self.wf_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.4f}'))
            
            self.wf_fig.tight_layout()
            
            self.wf_canvas = FigureCanvas(self.wf_fig)
            layout.addWidget(self.wf_canvas)
            
            toolbar = NavigationToolbar(self.wf_canvas, central)
            layout.addWidget(toolbar)
            
            self.waterfall_window.show()
            
            # Start update timer based on user setting
            wf_update_sec = float(self.wf_update_input.text())
            self.waterfall_timer = QTimer()
            self.waterfall_timer.timeout.connect(self.update_waterfall)
            self.waterfall_timer.start(int(wf_update_sec * 1000))
            
            # Do initial update
            self.update_waterfall()
            
            self.status_bar.showMessage("Live waterfall running...")
        
        def calculate_fspl_db(self, distance_m, frequency_hz):
            """Calculate Free Space Path Loss in dB.
            
            FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
                     = 20*log10(d) + 20*log10(f) - 147.55
            
            Args:
                distance_m: Slant distance in meters
                frequency_hz: Frequency in Hz
                
            Returns:
                FSPL in dB (positive value representing loss)
            """
            c = 299792458  # Speed of light in m/s
            if distance_m <= 0:
                return 200  # Return very high loss for invalid distance
            
            fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / c)
            return fspl_db
        
        def update_waterfall(self):
            """Update the live waterfall display with FSPL-based signal power."""
            if not self.waterfall_running:
                return
            
            from skyfield.api import utc
            from datetime import timezone
            
            current_time_utc = datetime.utcnow().replace(tzinfo=utc)
            
            # Calculate current Doppler spectrum (single time slice)
            # Store FSPL in dB directly for each frequency bin (lower = stronger signal)
            current_spectrum_db = np.full(self.wf_n_freq_bins, 250.0)  # Initialize with very high path loss
            visible_count = 0
            
            for pred in self.multi_predictor.predictors:
                try:
                    ts_time = pred.ts.from_datetime(current_time_utc)
                    observer_location = pred.wgs84.latlon(
                        pred.ue_lat,
                        pred.ue_lon,
                        elevation_m=pred.ue_alt * 1000
                    )
                    
                    relative = (pred.satellite - observer_location).at(ts_time)
                    xyz = relative.position.au
                    
                    # Calculate slant distance in meters (1 AU = 149597870700 m)
                    AU_TO_METERS = 149597870700
                    slant_distance_m = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2) * AU_TO_METERS
                    
                    horizontal_dist = np.sqrt(xyz[0]**2 + xyz[1]**2)
                    magnitude = (xyz[0]**2 + xyz[1]**2 + xyz[2]**2)**0.5
                    
                    if magnitude > 0:
                        elevation_rad = np.arctan2(xyz[2], horizontal_dist)
                        elevation_deg = np.degrees(elevation_rad)
                    else:
                        elevation_deg = -90
                    
                    if elevation_deg >= self.wf_elevation_mask:
                        shift = pred.calculate_doppler_shift(current_time_utc)
                        received_freq = pred.STARLINK_TX_FREQ + shift
                        
                        if self.wf_freq_start_hz <= received_freq <= self.wf_freq_end_hz:
                            # Calculate absolute FSPL in dB
                            fspl_db = self.calculate_fspl_db(slant_distance_m, received_freq)
                            
                            # Add signal with Gaussian shape
                            # Create Gaussian envelope centered at received frequency
                            gaussian = np.exp(-((self.wf_freq_grid_hz - received_freq) ** 2) / (2 * self.wf_sigma ** 2))
                            
                            # Convert gaussian to dB attenuation (peak = 0 dB, tails = positive addition to loss)
                            gaussian_db = -10 * np.log10(gaussian + 1e-10)
                            
                            # Total path loss: FSPL + gaussian shape (minimum = best signal)
                            signal_loss_db = fspl_db + gaussian_db
                            
                            # Take minimum path loss (strongest signal)
                            current_spectrum_db = np.minimum(current_spectrum_db, signal_loss_db)
                            visible_count += 1
                except:
                    pass
            
            # Scroll waterfall down and add new data at top (data is FSPL in dB)
            self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
            self.waterfall_data[0, :] = current_spectrum_db
            
            # Data is path loss in dB
            waterfall_db = self.waterfall_data.copy()
            
            # Update image data - find appropriate display range
            # For Starlink at ~550-2000 km, FSPL at 12 GHz is roughly 170-185 dB
            valid_data = waterfall_db[waterfall_db < 200]
            if len(valid_data) > 0:
                vmin = np.min(valid_data)  # Minimum path loss (strongest)
                vmax = vmin + 15  # 15 dB dynamic range
            else:
                vmin, vmax = 170, 185
            
            self.wf_im.set_data(waterfall_db)
            self.wf_im.set_clim(vmin=vmin, vmax=vmax)
            
            # Update title with current time
            current_time_aware = current_time_utc.astimezone()
            current_time_str = current_time_aware.strftime('%H:%M:%S %Z')
            self.wf_title.set_text(f'Live Doppler Waterfall - Path Loss (TX: {self.wf_tx_ghz:.3f} GHz)\n{current_time_str} ({visible_count} satellites visible)')
            
            self.wf_canvas.draw_idle()
        
        def stop_waterfall(self):
            """Stop the live waterfall."""
            self.waterfall_running = False
            if self.waterfall_timer:
                self.waterfall_timer.stop()
                self.waterfall_timer = None
            
            self.waterfall_btn.setEnabled(True)
            self.stop_waterfall_btn.setEnabled(False)
            self.status_bar.showMessage("Live waterfall stopped.")
                
        def save_figure(self):
            """Save figure to file."""
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Figure", "", "PNG files (*.png);;PDF files (*.pdf)"
            )
            
            if filepath:
                try:
                    self.fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                    self.status_bar.showMessage(f"Saved to {os.path.basename(filepath)}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save: {e}")
                    
        def closeEvent(self, event):
            """Handle window close."""
            self.stop_animation()
            self.stop_waterfall()
            event.accept()
    
    # Run the application
    app = QApplication(sys.argv)
    window = DopplerPredictorGUI()
    window.show()
    sys.exit(app.exec_())


def try_terminal_ui():
    """Fallback to a simple terminal-based menu."""
    import matplotlib
    matplotlib.use('macosx')  # Try native macOS backend
    
    from doppler_predictor2 import MultiSatellitePredictor
    import os
    
    print("\n" + "="*60)
    print("  Starlink Doppler Predictor - Terminal Interface")
    print("="*60)
    
    # Default location
    ue_location = {
        'latitude': 47.6550,
        'longitude': -122.3035,
        'altitude': 60
    }
    
    # Try to load TLE
    tle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'starlink.txt')
    
    if not os.path.exists(tle_path):
        print("\nNo starlink.txt found. Downloading...")
        try:
            import requests
            url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(tle_path, 'w') as f:
                f.write(response.text)
            print("Downloaded successfully!")
        except Exception as e:
            print(f"Download failed: {e}")
            return
    
    with open(tle_path, 'r') as f:
        tle_data = f.read()
    
    print("\nLoading satellites...")
    predictor = MultiSatellitePredictor(tle_data, ue_location, num_satellites=1000)
    
    while True:
        print("\n" + "-"*40)
        print("Options:")
        print("  1. Show Sky Map (real-time)")
        print("  2. Show Spectrum")
        print("  3. Change Location")
        print("  4. Quit")
        print("-"*40)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            print("\nStarting sky map... (close window to return)")
            predictor.plot_sky_map_realtime(elevation_mask=10.0, update_interval=1000)
        elif choice == '2':
            print("\nGenerating spectrum...")
            predictor.plot_combined_spectrum(duration_minutes=5, elevation_mask=10.0)
        elif choice == '3':
            try:
                lat = float(input("Enter latitude (°N): "))
                lon = float(input("Enter longitude (°E): "))
                alt = float(input("Enter altitude (m): "))
                ue_location = {'latitude': lat, 'longitude': lon, 'altitude': alt}
                predictor = MultiSatellitePredictor(tle_data, ue_location, num_satellites=1000)
                print("Location updated!")
            except ValueError:
                print("Invalid input!")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")


def main():
    """Main entry point - try different GUI backends."""
    print("Starting Doppler Predictor GUI...")
    
    # Try PyQt5 first
    try:
        try_pyqt5()
        return
    except ImportError as e:
        print(f"PyQt5 not available: {e}")
    except Exception as e:
        print(f"PyQt5 failed: {e}")
    
    # Fallback to terminal UI
    print("\nFalling back to terminal interface...")
    try_terminal_ui()


if __name__ == "__main__":
    main()
