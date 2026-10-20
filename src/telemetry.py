import os
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, List
from src.tasks.risk_assessment import TelemetryData
from src.tasks.config.utils import CONFIG

logger = logging.getLogger(__name__)

class TelemetryProcessor:
    """
    Processes GoPro GPMF telemetry files, synchronizes them to a 30 FPS grid,
    and computes orientation (Roll, Pitch, Yaw) and Speed with GPS fallback.

    This module is designed to align physical sensor data with the vision pipeline's
    epoch-based processing (30 frames per epoch).
    """
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.fps = 30.0
        self.dt = 1.0 / self.fps

        # Configuration Toggle: Allows skipping telemetry processing for performance or missing data
        self.use_telemetry = CONFIG["processing"].get("use_imu_telemetry", True)

        if not self.use_telemetry:
            logger.info(f"Telemetry processing disabled via config for {folder_path}")
            self.data = None
            return

        logger.debug(f"Initializing telemetry processor for folder: {folder_path}")
        self._load_and_sync()

    def _load_and_sync(self):
        """
        Loads standardized GPMF CSV files and resamples them onto a master 30 FPS time grid.
        """
        # Standardized filenames as per GoPro GPMF export specifications
        files = {
            "accl": "1_ACCL.csv", # Linear Acceleration [x, y, z]
            "gyro": "1_GYRO.csv", # Angular Velocity [x, y, z]
            "gps": "1_GPS5.csv",  # GPS Metrics (Speed, HDOP, etc.)
            "cori": "1_CORI.csv", # Camera Orientation Quaternions [w, x, y, z]
            "grav": "1_GRAV.csv", # Gravity direction vector [x, y, z]
            "iori": "1_IORI.csv"  # Image Orientation Quaternions
        }

        streams = {}
        for key, filename in files.items():
            path = os.path.join(self.folder_path, filename)
            if os.path.exists(path):
                try:
                    # Use utf-8-sig to handle the Byte Order Mark (BOM) often found in GoPro CSVs
                    df = pd.read_csv(path, encoding='utf-8-sig')

                    # Time synchronization: Ensure we have a consistent time reference (t_s in seconds)
                    if 't_s' not in df.columns:
                        if 'cts' in df.columns:
                            df['t_s'] = df['cts'] / 1000.0
                        else:
                            logger.warning(f"Telemetry file {filename} missing both 't_s' and 'cts' columns. Skipping.")
                            continue
                    streams[key] = df
                    logger.debug(f"Successfully loaded {filename} ({len(df)} rows)")
                except Exception as e:
                    logger.error(f"Error reading telemetry file {filename}: {e}")
            else:
                logger.warning(f"Telemetry file {filename} not found in {self.folder_path}")

        if not streams:
            logger.error(f"No valid telemetry streams found in {self.folder_path}. Telemetry will be disabled.")
            self.data = None
            return

        # 1. Construct Master Frame Grid
        t_max = min([df['t_s'].max() for df in streams.values()])
        t_grid = np.arange(0, t_max, self.dt)
        logger.debug(f"Constructed master time grid: 0.0s to {t_max:.2f}s ({len(t_grid)} frames)")

        # 2. Resample/Interpolate streams onto the 30 FPS grid
        synced_data = {"t_s": t_grid}

        # Orientation (CORI) - Map indices to quaternions [w, x, y, z]
        if "cori" in streams:
            df = streams["cori"]
            if len(df.columns) >= 8:
                synced_data["q_w"] = np.interp(t_grid, df['t_s'], df.iloc[:, 4])
                synced_data["q_x"] = np.interp(t_grid, df['t_s'], df.iloc[:, 5])
                synced_data["q_y"] = np.interp(t_grid, df['t_s'], df.iloc[:, 6])
                synced_data["q_z"] = np.interp(t_grid, df['t_s'], df.iloc[:, 7])
            else:
                logger.warning(f"CORI file has unexpected column count ({len(df.columns)}). Skipping orientation.")

        # Gravity (GRAV) - Map indices to gravity vector [x, y, z]
        if "grav" in streams:
            df = streams["grav"]
            if len(df.columns) >= 6:
                synced_data["g_x"] = np.interp(t_grid, df['t_s'], df.iloc[:, 3])
                synced_data["g_y"] = np.interp(t_grid, df['t_s'], df.iloc[:, 4])
                synced_data["g_z"] = np.interp(t_grid, df['t_s'], df.iloc[:, 5])
            else:
                logger.warning(f"GRAV file has unexpected column count ({len(df.columns)}). Skipping gravity.")

        # Acceleration (ACCL) - Map indices to [z, x, y]
        if "accl" in streams:
            df = streams["accl"]
            if len(df.columns) >= 7:
                synced_data["a_x"] = np.interp(t_grid, df['t_s'], df.iloc[:, 5])
                synced_data["a_y"] = np.interp(t_grid, df['t_s'], df.iloc[:, 6])
                synced_data["a_z"] = np.interp(t_grid, df['t_s'], df.iloc[:, 4])
            else:
                logger.warning(f"ACCL file has unexpected column count ({len(df.columns)}). Skipping accel.")

        # Gyro (GYRO) - Map indices to [z, x, y]
        if "gyro" in streams:
            df = streams["gyro"]
            if len(df.columns) >= 7:
                synced_data["gy_x"] = np.interp(t_grid, df['t_s'], df.iloc[:, 5])
                synced_data["gy_y"] = np.interp(t_grid, df['t_s'], df.iloc[:, 6])
                synced_data["gy_z"] = np.interp(t_grid, df['t_s'], df.iloc[:, 4])
            else:
                logger.warning(f"GYRO file has unexpected column count ({len(df.columns)}). Skipping gyro.")

        # GPS (GPS5) - Use indices for robustness: precision=col 4, 2D speed=col 9
        if "gps" in streams:
            df = streams["gps"]
            if len(df.columns) >= 10:
                gps_df = df[['t_s', df.columns[9], df.columns[4]]].sort_values('t_s')
                grid_df = pd.DataFrame({'t_s': t_grid})
                merged = pd.merge_asof(grid_df, gps_df, on='t_s', direction='backward')
                synced_data["gps_speed"] = merged.iloc[:, 1].values
                synced_data["gps_hdop"] = merged.iloc[:, 2].values
            else:
                logger.warning(f"GPS file has unexpected column count ({len(df.columns)}). Skipping GPS.")
        else:
            synced_data["gps_speed"] = np.full(len(t_grid), np.nan)
            synced_data["gps_hdop"] = np.full(len(t_grid), np.nan)

        self.data = pd.DataFrame(synced_data)
        self._compute_metrics()
        logger.info(f"Telemetry synchronization complete for {self.folder_path}")

    def _compute_metrics(self):
        """
        Calculates derived physical metrics: Euler angles from quaternions and
        speed with an IMU dead-reckoning fallback.
        """
        if self.data is None: return

        # 1. Orientation: Convert Quaternions [w, x, y, z] to Euler Angles (Roll, Pitch, Yaw)
        if "q_w" in self.data.columns:
            qw = self.data['q_w'].values
            qx = self.data['q_x'].values
            qy = self.data['q_y'].values
            qz = self.data['q_z'].values

            # Roll (phi): Rotation around X-axis
            self.data['roll'] = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
            # Pitch (theta): Rotation around Y-axis
            self.data['pitch'] = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0)))
            # Yaw (psi): Rotation around Z-axis
            self.data['yaw'] = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))
        else:
            logger.warning("Orientation data (CORI) missing. Setting Roll/Pitch/Yaw to 0.0")
            self.data['roll'] = 0.0
            self.data['pitch'] = 0.0
            self.data['yaw'] = 0.0

        # 2. Speed Estimation with GPS Fallback
        # Priority: GPS -> IMU Dead Reckoning -> Last Known Velocity
        speeds = []
        sources = []
        v_curr = 0.0

        for i in range(len(self.data)):
            # Primary Source: GPS
            gps_v = self.data['gps_speed'].iloc[i]
            hdop = self.data['gps_hdop'].iloc[i]

            # Validate GPS quality: exclude unrealistic speeds (>35m/s) and poor precision (HDOP > 2.0)
            if not np.isnan(gps_v) and gps_v < 35.0 and hdop < 2.0:
                v_curr = gps_v
                speeds.append(v_curr)
                sources.append("GPS")
            else:
                # Fallback Source: Dead Reckoning from IMU
                if "a_x" in self.data.columns and "q_w" in self.data.columns:
                    # 1. Get raw acceleration in camera frame
                    ax = self.data['a_x'].iloc[i]
                    ay = self.data['a_y'].iloc[i]
                    az = self.data['a_z'].iloc[i]

                    # 2. Get current orientation quaternions
                    qw = self.data['q_w'].iloc[i]
                    qx = self.data['q_x'].iloc[i]
                    qy = self.data['q_y'].iloc[i]
                    qz = self.data['q_z'].iloc[i]

                    # 3. Rotate acceleration from Camera Frame to Earth Frame (X-forward)
                    # We only calculate the Forward (X) component of the rotation matrix R_cam_to_earth
                    r00 = 1 - 2*(qy**2 + qz**2)
                    r10 = 2*(qx*qy + qw*qz)
                    r20 = 2*(qx*qz - qw*qy)
                    a_fwd = r00*ax + r10*ay + r20*az

                    # 4. ZUPT (Zero Velocity Update)
                    # If the bike is physically stationary (low gyro/accel), force velocity to 0 to kill drift
                    is_stationary = False
                    if "gy_x" in self.data.columns:
                        gyro_mag = np.sqrt(self.data['gy_x'].iloc[i]**2 +
                                           self.data['gy_y'].iloc[i]**2 +
                                           self.data['gy_z'].iloc[i]**2)
                        if gyro_mag < 0.1:
                            is_stationary = True

                    if is_stationary:
                        v_curr = 0.0
                    else:
                        # Integrate acceleration: v_t = v_{t-1} + a * dt
                        v_curr += a_fwd * self.dt

                    speeds.append(max(0.0, v_curr))
                    sources.append("IMU_DEAD_RECKONING")
                else:
                    # Total fallback: keep last known velocity
                    speeds.append(v_curr)
                    sources.append("UNKNOWN")

        self.data['speed_ms'] = speeds
        self.data['speed_kmh'] = np.array(speeds) * 3.6
        self.data['speed_source'] = sources
        logger.debug("Speed estimation and orientation metrics computed.")

    def get_epoch_telemetry(self, epoch_idx: int) -> Optional[TelemetryData]:
        """
        Extracts a summary of telemetry for a specific 30-frame epoch.
        Used by the G-VEval risk assessment prompt.
        """
        if self.data is None:
            return None

        start_idx = epoch_idx * 30
        end_idx = start_idx + 30

        if start_idx >= len(self.data):
            logger.debug(f"Epoch index {epoch_idx} out of bounds. Returning None.")
            return None

        epoch_df = self.data.iloc[start_idx:end_idx]

        # We use the mean for rates/angles and the last value for velocity to represent the epoch's state
        return TelemetryData(
            ttc=0.0, # Requires per-object distance (computed in risk assessment)
            pet=0.0,
            d_lat=0.0,
            v_rel=0.0,
            speed_ms=float(epoch_df['speed_ms'].iloc[-1]) if 'speed_ms' in epoch_df.columns else 0.0,
            speed_kmh=float(epoch_df['speed_kmh'].iloc[-1]) if 'speed_kmh' in epoch_df.columns else 0.0,
            a_ego=float(epoch_df['a_x'].mean()) if 'a_x' in epoch_df.columns else 0.0,
            roll=float(epoch_df['roll'].mean()),
            yaw_rate=float(epoch_df['gy_z'].mean()) if 'gy_z' in epoch_df.columns else 0.0
        )
