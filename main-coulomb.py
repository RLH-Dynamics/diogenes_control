"""
main-coulomb.py: Automated Static Friction Characterization

PURPOSE:
    This script performs a series of 'ramp-up' tests on a single RobStride RS03
    actuator to identify its breakaway torque (static friction). It captures 
    the exact moment the internal friction is overcome by the commanded torque.

HARDWARE PREREQUISITES:
    - RobStride RS03 Actuator (connected via CAN)
    - Raspberry Pi or Linux machine with SocketCAN setup
    - The actuator must be in a 'clear' physical state (no mechanical obstructions)
"""

import sys
import time
import os
import datetime
import numpy as np
import matplotlib
# Use the 'Agg' backend to allow saving plots to files without needing a GUI display
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Internal Project Imports
from robot.leg import Leg
from utils.exceptions import HardwareIOError, HardwareError
from config import RS03_LIMITS, CAN_CHANNEL, HOST_ID

# =============================================================================
# USER CONFIGURATION: Modify these variables to change the test behavior
# =============================================================================

# 1. Hardware Selection
TEST_MOTOR_ID = 1          # The CAN ID of the motor you wish to test

# 2. Experiment Volume
N_TRIALS = 100             # Total number of test repetitions to perform

# 3. Torque Resolution & Limits
TORQUE_STEP = 0.005        # Amount of torque (N.m) to add in each loop iteration
MAX_TORQUE = 5.0           # Safety limit: Stop test if this torque is reached
SETTLE_TIME_SEC = 1.5      # Time to wait between trials for the motor to cool/settle

# 4. Sensitivity
# Radians of change required to trigger a 'Movement Detected' event (~1.1 degrees)
MOVEMENT_THRESHOLD = 0.02  

# 5. Timing
LOOP_RATE_HZ = 30.0        # Rate at which we poll the motor. Keep low to avoid CAN congestion.
DT = 1.0 / LOOP_RATE_HZ

# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_single_trial(leg, motor_id, trial_dir, trial_idx):
    """
    Executes one ramp-up sequence from 0.0 N.m until breakaway.
    """
    recorded_torques = []
    recorded_positions = []
    recorded_velocities = []
    static_friction = None

    print(f"\n[TRIAL {trial_idx:02d}] Initializing...")
    
    # Clear out any status messages left in the CAN buffer from previous runs
    leg.robstride.flush_CAN_bus() 
    
    # Establish the 'Zero' position for this specific trial
    # This helps account for any small drifts or gear backlash between runs
    initial_pos, _ = leg.get_position_and_velocity(motor_id)
    
    current_torque = 0.0
    while current_torque <= MAX_TORQUE:
        loop_start = time.perf_counter()

        # Command the new torque value
        leg.set_output_torques({motor_id: current_torque})
        
        # Read the resulting physical state
        pos, vel = leg.get_position_and_velocity(motor_id)
        
        recorded_torques.append(current_torque)
        recorded_positions.append(pos)
        recorded_velocities.append(vel)
        
        # Check if the displacement from the start exceeds our threshold
        if abs(pos - initial_pos) >= MOVEMENT_THRESHOLD:
            static_friction = current_torque
            print(f"\n  -> BREAKAWAY DETECTED: {static_friction:.3f} N.m")
            break
            
        current_torque += TORQUE_STEP
        
        # Maintain loop timing
        elapsed = time.perf_counter() - loop_start
        if elapsed < DT:
            time.sleep(DT - elapsed)

    # Safety: Gradually return torque to zero before concluding the trial
    print(f"  -> Ramping torque down...")
    temp_torque = current_torque
    while temp_torque > 0.0:
        temp_torque -= (TORQUE_STEP * 10)
        leg.set_output_torques({motor_id: max(0.0, temp_torque)})
        time.sleep(0.005)

    # Generate a plot for this specific trial
    plot_trial_data(trial_dir, trial_idx, recorded_torques, recorded_positions, 
                    recorded_velocities, static_friction, initial_pos)

    return {
        'torque': np.array(recorded_torques),
        'pos': np.array(recorded_positions),
        'vel': np.array(recorded_velocities),
        'breakaway': static_friction,
        'initial_pos': initial_pos
    }

def plot_trial_data(path, idx, t, p, v, friction, init_p):
    """Saves a diagnostic plot for a single trial."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    ax1.plot(t, p, 'b-', label='Measured Position')
    ax1.axhline(init_p, color='k', linestyle='--', alpha=0.3, label='Start Position')
    if friction: 
        ax1.axvline(friction, color='r', linestyle=':', label='Breakaway Torque')
    ax1.set_ylabel('Position (rad)')
    ax1.set_title(f'Trial {idx} Raw Data')
    ax1.legend(fontsize='x-small')
    
    ax2.plot(t, v, 'm-', label='Measured Velocity')
    if friction: 
        ax2.axvline(friction, color='r', linestyle=':')
    ax2.set_xlabel('Commanded Torque (N.m)')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.legend(fontsize='x-small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'trial_{idx}_results.png'))
    plt.close()

def main():
    print(f"--- Automated {N_TRIALS}-Trial Coulomb Friction Test ---")
    print(f"Target Motor ID: {TEST_MOTOR_ID}")
    
    confirm = input("Confirm hardware is clear and type 'YES' to proceed: ")
    if confirm != "YES": 
        print("Aborting.")
        sys.exit(0)

    # 1. Create directory structure for data logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"coulomb_test_{timestamp}_id{TEST_MOTOR_ID}"
    os.makedirs(session_dir, exist_ok=True)

    # 2. Instantiate the Leg interface
    # We use TORQUE mode to allow direct current (Iq) control
    leg = Leg(limits=RS03_LIMITS, channel=CAN_CHANNEL, host_id=HOST_ID, 
              motor_ids=[TEST_MOTOR_ID], control_mode='TORQUE')
    
    all_trial_data = []

    try:
        # Initialize hardware and set the safety watchdog
        # The 1000ms watchdog prevents the motor from shutting down during the inter-trial sleep
        leg.robstride.init_CAN_bus()
        leg.robstride.enable_hardware_watchdog(timeout_ms=1000) 
        
        for i in range(1, N_TRIALS + 1):
            # Re-enable the motor at the start of every trial. 
            # This clears any safety faults (like a previous watchdog trip)
            leg.robstride.enable_and_verify_all(limits=RS03_LIMITS, control_mode='TORQUE')
            
            trial_dir = os.path.join(session_dir, f"trial_{i:02d}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Execute the ramp-up
            data = run_single_trial(leg, TEST_MOTOR_ID, trial_dir, i)
            all_trial_data.append(data)
            
            # Allow mechanical vibrations to damp out before the next run
            time.sleep(SETTLE_TIME_SEC)

    except Exception as e:
        print(f"\n[FATAL ERROR] Experiment halted: {e}")
    finally:
        # Always shut down the CAN bus and disable motors on exit
        leg.shutdown()

    # =========================================================================
    # POST-PROCESSING & SUMMARY VISUALIZATION
    # =========================================================================
    if not all_trial_data:
        print("[WARN] No data collected. Exiting.")
        return

    print(f"\n[INFO] Generating summary reports for {N_TRIALS} trials...")
    
    # 1. Generate Overlay Plot (Relative Position)
    plt.figure(figsize=(10, 6))
    breakaway_values = []
    for i, d in enumerate(all_trial_data):
        # We plot position relative to the start of the trial
        plt.plot(d['torque'], d['pos'] - d['initial_pos'], alpha=0.6, label=f'Trial {i+1}')
        if d['breakaway']: 
            breakaway_values.append(d['breakaway'])
    
    plt.title(f"Actuator {TEST_MOTOR_ID}: Multi-Trial Comparison")
    plt.xlabel("Torque (N.m)")
    plt.ylabel("Displacement (rad)")
    plt.legend(ncol=3, fontsize='xx-small')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(session_dir, "summary_overlaid_trials.png"))
    
    # 2. Generate Statistical Average Plot
    if breakaway_values:
        mean_val = np.mean(breakaway_values)
        std_val = np.std(breakaway_values)
        
        # Interpolate results to a common torque grid to calculate a mathematical average
        common_grid = np.linspace(0, max(breakaway_values), 200)
        interp_positions = []
        for d in all_trial_data:
            interp_p = np.interp(common_grid, d['torque'], d['pos'] - d['initial_pos'])
            interp_positions.append(interp_p)
            
        avg_trajectory = np.mean(interp_positions, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(common_grid, avg_trajectory, 'k-', linewidth=2, label='Mean Displacement')
        plt.axvline(mean_val, color='r', label=f'Mean Friction: {mean_val:.3f} N.m')
        plt.axvspan(mean_val - std_val, mean_val + std_val, color='r', alpha=0.2, label='Std Deviation')
        
        plt.title(f"Average Friction Characteristic (N={N_TRIALS})")
        plt.xlabel("Torque (N.m)")
        plt.ylabel("Mean Displacement (rad)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(session_dir, "summary_average_friction.png"))
        
        print(f"\n[STATISTICAL SUMMARY]")
        print(f"  Mean Static Friction: {mean_val:.4f} N.m")
        print(f"  Standard Deviation:   {std_val:.4f} N.m")
    
    print(f"\n[SUCCESS] Session data saved to: {session_dir}")

if __name__ == "__main__":
    main()