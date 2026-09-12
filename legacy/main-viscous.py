"""
main-viscous.py: Automated Viscous Damping Characterization

PURPOSE:
    This script performs a 'coast-down' test on a single RobStride RS03
    actuator to identify its viscous damping characteristics. It spins the 
    motor to a steady velocity, instantly cuts torque to zero, and logs 
    the velocity decay over time.

HARDWARE PREREQUISITES:
    - RobStride RS03 Actuator (connected via CAN)
    - Raspberry Pi or Linux machine with SocketCAN setup
    - The actuator must be in a 'clear' physical state (free spinning, NO load)
"""

import sys
import time
import os
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Internal Project Imports
from robot.leg import Leg
from robstride.protocol import ParameterType
from utils.exceptions import HardwareIOError, HardwareError
from config import RS03_LIMITS, CAN_CHANNEL, HOST_ID
from robstride.protocol import ParameterType, CommunicationType

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# 1. Hardware Selection
TEST_MOTOR_ID = 1          # The CAN ID of the motor you wish to test

# 2. Experiment Volume
N_TRIALS = 5               # Total number of test repetitions to perform

# 3. Kinematic Parameters
TARGET_VELOCITY = 15.0     # Target speed to reach before coasting (rad/s)
SETTLE_TIME_SEC = 2.0      # Time to hold target velocity to ensure steady state
COAST_STOP_VELOCITY = 0.5  # Velocity threshold (rad/s) to consider the motor "stopped"

# 4. Timing
LOOP_RATE_HZ = 100.0       # Polling rate during coast down (higher = better curve fit)
DT = 1.0 / LOOP_RATE_HZ

# 5. Known Friction Configuration (Outputs from main-coulomb.py)
# Used to calculate Reflected Inertia and Raw Viscous Damping. 
# Set to None if you do not want to calculate derived properties.
KNOWN_COULOMB_NM_MEAN = 0.5  # REPLACE with your main-coulomb.py mean (Nm)
KNOWN_COULOMB_NM_STD = 0.1   # REPLACE with your main-coulomb.py std (Nm)

# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_coast_down_trial(leg, motor_id, trial_dir, trial_idx):
    """
    Executes one spin-up and coast-down sequence.
    """
    recorded_times = []
    recorded_velocities = []

    print(f"\n[TRIAL {trial_idx:02d}] Initializing...")
    leg.robstride.flush_CAN_bus() 
    
    # 1. Spin up phase (MIT Mode)
    print(f"  -> Spinning up to {TARGET_VELOCITY} rad/s in MIT mode...")
    leg.robstride.enable_and_verify_all(limits=RS03_LIMITS, control_mode='MIT')
    
    # Gentle ramp up (Using Kd as a velocity P-gain, Kp=0)
    for v in np.linspace(0, TARGET_VELOCITY, 20):
        leg.robstride.send_target_state_vector(motor_id, pos=0.0, vel=v, kp=0.0, kd=0.5, torque=0.0, limits=RS03_LIMITS)
        time.sleep(0.05)
        
    print(f"  -> Holding steady state for {SETTLE_TIME_SEC} seconds...")
    hold_start = time.perf_counter()
    while (time.perf_counter() - hold_start) < SETTLE_TIME_SEC:
        # Constantly send the target to keep the hardware watchdog happy
        leg.robstride.send_target_state_vector(motor_id, pos=0.0, vel=TARGET_VELOCITY, kp=0.0, kd=0.5, torque=0.0, limits=RS03_LIMITS)
        time.sleep(0.01)

    # 2. Coast-down phase (Freewheel via Zero-Gain)
    print("  -> CUTTING GAINS to 0.0 for true freewheel coasting...")

    # 3. Data logging
    start_time = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        
        # Keep the motor enabled but with 0 resistance (freewheel)
        leg.robstride.send_target_state_vector(motor_id, pos=0.0, vel=0.0, kp=0.0, kd=0.0, torque=0.0, limits=RS03_LIMITS)
        
        # Read parameters over CAN (it will broadcast correctly now)
        pos, vel = leg.get_position_and_velocity(motor_id)
        
        t = loop_start - start_time
        recorded_times.append(t)
        recorded_velocities.append(vel)
        
        # Print live velocity occasionally so we can watch it decay
        if len(recorded_times) % 5 == 0:
            print(f"     [t={t:.2f}s] Velocity: {vel:.3f} rad/s")
            
        # Break condition: Velocity drops below the stop threshold
        if abs(vel) < COAST_STOP_VELOCITY and t > 0.5:
            print(f"  -> Motor stopped. Coast duration: {t:.3f} seconds.")
            break
            
        # Maintain loop timing
        elapsed = time.perf_counter() - loop_start
        if elapsed < DT:
            time.sleep(DT - elapsed)

    # Convert to numpy arrays for math
    t_arr = np.array(recorded_times)
    v_arr = np.array(recorded_velocities)

    # 4. Calculate Decay Constants using Non-Linear Curve Fit (Mixed Friction Model)
    # ODE: dV/dt = -a*V - b  ---> Solution: V(t) = (V0 + b/a) * exp(-a*t) - b/a
    # where a = viscous term (c/I), b = coulomb term (C_dry/I)
    
    def coast_model(t, v0, a, b):
        # Clamp 'a' to a tiny positive number to prevent division by zero during solver iteration
        a_safe = max(a, 1e-6) 
        return (v0 + b/a_safe) * np.exp(-a_safe * t) - b/a_safe

    # Filter for positive velocities
    valid_idx = v_arr > 0.05 
    t_fit_data = t_arr[valid_idx]
    v_fit_data = v_arr[valid_idx]
    
    v0_fit, viscous_rate, coulomb_rate = None, None, None
    
    if np.sum(valid_idx) > 5:
        try:
            # Initial guesses: V0 ~ starting vel, a ~ 0.5 (low viscous), b ~ 2.0 (moderate dry friction)
            p0 = [v_fit_data[0], 0.5, 2.0]
            # Bounds: V0 > 0, a >= 0 (viscous), b >= 0 (coulomb)
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            
            popt, _ = curve_fit(coast_model, t_fit_data, v_fit_data, p0=p0, bounds=bounds)
            v0_fit, viscous_rate, coulomb_rate = popt
            
            print(f"  -> Viscous Damping Rate (c/I): {viscous_rate:.4f} s^-1")
            print(f"  -> Coulomb Friction Rate (C_dry/I): {coulomb_rate:.4f} rad/s^2")
        except RuntimeError:
            print("  -> [WARN] Curve fit failed to converge.")
    else:
        print("  -> [WARN] Not enough clean data points for curve fit.")

    # Generate a plot for this specific trial
    plot_trial_data(trial_dir, trial_idx, t_arr, v_arr, v0_fit, viscous_rate, coulomb_rate)

    return {
        'time': t_arr,
        'vel': v_arr,
        'viscous_rate': viscous_rate,
        'coulomb_rate': coulomb_rate
    }

def plot_trial_data(path, idx, t, v, v0_fit, viscous_rate, coulomb_rate):
    """Saves a diagnostic plot for a single coast-down trial."""
    plt.figure(figsize=(8, 5))
    
    # Raw Data
    plt.plot(t, v, 'b.', alpha=0.5, label='Measured Velocity')
    
    # Curve Fit Overlay
    if None not in (v0_fit, viscous_rate, coulomb_rate):
        t_fit = np.linspace(0, max(t), 100)
        
        # Reconstruct the combined model
        a_safe = max(viscous_rate, 1e-6)
        v_fit = (v0_fit + coulomb_rate/a_safe) * np.exp(-a_safe * t_fit) - coulomb_rate/a_safe
        
        # Stop plotting the fit once it crosses zero (since dry friction reverses direction)
        v_fit = np.maximum(v_fit, 0)
        
        plt.plot(t_fit, v_fit, 'r-', linewidth=2, 
                 label=f'Mixed Fit\nViscous: {viscous_rate:.2f} $s^{{-1}}$\nCoulomb: {coulomb_rate:.2f} $rad/s^2$')

    plt.title(f'Trial {idx}: Coast-Down Profile')
    plt.xlabel('Time since Torque Cut (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'trial_{idx}_results.png'))
    plt.close()

def main():
    print(f"--- Automated {N_TRIALS}-Trial Viscous Damping Test ---")
    print(f"Target Motor ID: {TEST_MOTOR_ID}")
    print("WARNING: Motor will spin rapidly to 15 rad/s.")
    
    confirm = input("Confirm hardware is clear of obstacles and type 'YES': ")
    if confirm != "YES": 
        print("Aborting.")
        sys.exit(0)

    # Create directory structure for data logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"viscous_test_{timestamp}_id{TEST_MOTOR_ID}"
    os.makedirs(session_dir, exist_ok=True)

    # Instantiate the Leg interface (Start in passive MIT mode)
    leg = Leg(limits=RS03_LIMITS, channel=CAN_CHANNEL, host_id=HOST_ID, 
              motor_ids=[TEST_MOTOR_ID], control_mode='MIT')
    
    all_trial_data = []

    try:
        leg.robstride.init_CAN_bus()
        leg.robstride.enable_hardware_watchdog(timeout_ms=1000) 
        
        for i in range(1, N_TRIALS + 1):
            trial_dir = os.path.join(session_dir, f"trial_{i:02d}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Execute coast down
            data = run_coast_down_trial(leg, TEST_MOTOR_ID, trial_dir, i)
            all_trial_data.append(data)
            
            # Let motor rest
            time.sleep(1.0)

    except Exception as e:
        print(f"\n[FATAL ERROR] Experiment halted: {e}")
    finally:
        leg.shutdown()

    # =========================================================================
    # POST-PROCESSING & SUMMARY
    # =========================================================================
    if not all_trial_data:
        return

    print(f"\n[INFO] Generating summary reports...")
    
    plt.figure(figsize=(10, 6))
    valid_viscous = []
    valid_coulomb = []
    
    for i, d in enumerate(all_trial_data):
        plt.plot(d['time'], d['vel'], alpha=0.6, label=f'Trial {i+1}')
        
        # Collect valid rates for the statistical summary
        if d['viscous_rate'] is not None and d['coulomb_rate'] is not None:
            valid_viscous.append(d['viscous_rate'])
            valid_coulomb.append(d['coulomb_rate'])
            
    plt.title(f"Actuator {TEST_MOTOR_ID}: Coast-Down Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(session_dir, "summary_overlaid_trials.png"))
    
    if valid_viscous and valid_coulomb:
        mean_visc_rate = np.mean(valid_viscous)
        std_visc_rate = np.std(valid_viscous)
        mean_coul_rate = np.mean(valid_coulomb)
        std_coul_rate = np.std(valid_coulomb)

        print(f"\n[STATISTICAL SUMMARY]")
        print("Note: Rates are scaled by Rotor Inertia (1/I).")
        print(f"  Mean Viscous Rate (c/I):     {mean_visc_rate:.4f} ± {std_visc_rate:.4f} s^-1")
        print(f"  Mean Coulomb Rate (C_dry/I): {mean_coul_rate:.4f} ± {std_coul_rate:.4f} rad/s^2")

        # --- DERIVED PHYSICAL PROPERTIES ---
        if KNOWN_COULOMB_NM_MEAN is not None:
            # 1. Reflected Inertia (I) = C_dry_raw / Coulomb_Rate
            mean_inertia = KNOWN_COULOMB_NM_MEAN / mean_coul_rate
            
            # Error propagation for I (division: combining relative errors)
            rel_std_coulomb_raw = (KNOWN_COULOMB_NM_STD / KNOWN_COULOMB_NM_MEAN) if KNOWN_COULOMB_NM_STD else 0.0
            rel_std_coul_rate = std_coul_rate / mean_coul_rate
            std_inertia = mean_inertia * np.sqrt(rel_std_coulomb_raw**2 + rel_std_coul_rate**2)
            
            # 2. Raw Viscous Damping (c) = Viscous_Rate * I
            mean_viscous_raw = mean_visc_rate * mean_inertia
            
            # Error propagation for c (multiplication: combining relative errors)
            rel_std_visc_rate = std_visc_rate / mean_visc_rate
            rel_std_inertia = std_inertia / mean_inertia
            std_viscous_raw = mean_viscous_raw * np.sqrt(rel_std_visc_rate**2 + rel_std_inertia**2)
            
            print(f"\n[DERIVED PHYSICAL PROPERTIES]")
            print(f"Using known static friction: {KNOWN_COULOMB_NM_MEAN:.4f} ± {(KNOWN_COULOMB_NM_STD or 0):.4f} Nm")
            print(f"  Reflected Inertia (I):       {mean_inertia:.6f} ± {std_inertia:.6f} kg*m^2")
            print(f"  Raw Viscous Damping (c):     {mean_viscous_raw:.6f} ± {std_viscous_raw:.6f} N*m*s/rad")

    print(f"\n[SUCCESS] Session data saved to: {session_dir}")

if __name__ == "__main__":
    main()