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

    # 4. Calculate Decay Constant using Log-Linear Regression
    # V(t) = V0 * e^(-decay_rate * t)  =>  ln(V) = ln(V0) - decay_rate * t
    valid_idx = v_arr > 0.1 # Filter out negatives/zeros before log
    decay_rate = None
    v0_fit = None
    
    if np.sum(valid_idx) > 5:
        p = np.polyfit(t_arr[valid_idx], np.log(v_arr[valid_idx]), 1)
        decay_rate = -p[0]       # Represents c/I (viscous damping / inertia)
        v0_fit = np.exp(p[1])    # Theoretical V0
        print(f"  -> Estimated Decay Rate (c/I): {decay_rate:.4f} s^-1")
    else:
        print("  -> [WARN] Not enough clean data points for curve fit.")

    # Generate a plot for this specific trial
    plot_trial_data(trial_dir, trial_idx, t_arr, v_arr, v0_fit, decay_rate)

    return {
        'time': t_arr,
        'vel': v_arr,
        'decay_rate': decay_rate
    }

def plot_trial_data(path, idx, t, v, v0_fit, decay_rate):
    """Saves a diagnostic plot for a single coast-down trial."""
    plt.figure(figsize=(8, 5))
    
    # Raw Data
    plt.plot(t, v, 'b.', alpha=0.5, label='Measured Velocity')
    
    # Curve Fit Overlay
    if decay_rate is not None and v0_fit is not None:
        t_fit = np.linspace(0, max(t), 100)
        v_fit = v0_fit * np.exp(-decay_rate * t_fit)
        plt.plot(t_fit, v_fit, 'r-', linewidth=2, 
                 label=f'Exp Fit (Decay: {decay_rate:.2f} $s^{{-1}}$)')

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
    valid_decays = []
    
    for i, d in enumerate(all_trial_data):
        plt.plot(d['time'], d['vel'], alpha=0.6, label=f'Trial {i+1}')
        if d['decay_rate'] is not None:
            valid_decays.append(d['decay_rate'])
            
    plt.title(f"Actuator {TEST_MOTOR_ID}: Coast-Down Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(session_dir, "summary_overlaid_trials.png"))
    
    if valid_decays:
        mean_decay = np.mean(valid_decays)
        std_decay = np.std(valid_decays)
        print(f"\n[STATISTICAL SUMMARY]")
        print("Note: Decay rate represents 'c/I' (Viscous Damping coeff / Rotor Inertia).")
        print("Multiply by estimated Rotor Inertia (kg*m^2) to find pure 'c'.")
        print(f"  Mean Decay Rate: {mean_decay:.4f} s^-1")
        print(f"  Standard Deviation: {std_decay:.4f} s^-1")
    
    print(f"\n[SUCCESS] Session data saved to: {session_dir}")

if __name__ == "__main__":
    main()