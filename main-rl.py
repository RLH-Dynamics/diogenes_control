import sys
import time
import csv
from datetime import datetime
import numpy as np
from robot.leg import Leg
from control.policy import Policy
from utils.safety import SafetyMonitor
from utils.exceptions import HardwareIOError, ActuatorFault, HardwareError, SafetyLimitError
from config import (
    JOINT_CONFIG, RS03_LIMITS, CAN_CHANNEL, HOST_ID, 
    KP_GAIN, KD_GAIN, DT, MODEL_PATH, NUM_JOINTS, 
    HISTORY_LEN, DEFAULT_POS, ACTION_SCALE,
    LOOP_RATE_HZ, DT, POLICY_UPDATE_INTERVAL, LPF_ALPHA
)

def format_targets(target_array):
    """
    Helper function to convert the policy's flat array output into the 
    dictionary structure expected by the robstride leg commands.
    """
    # Sort motor configs by ID to match the sorting in policy.py
    sorted_configs = sorted(JOINT_CONFIG.values(), key=lambda x: x['id'])
    return {
        config['id']: {'pos': float(target_array[i]), 'vel': 0.0, 'torque': 0.0} 
        for i, config in enumerate(sorted_configs)
    }

def main():
    print(f"[INFO] Setting up Leg for dual-rate RL policy control loop...")
    print(f"[INFO] Control Loop: {LOOP_RATE_HZ}Hz")

    # Extract motor IDs from the joint configuration
    motor_ids = [config['id'] for config in JOINT_CONFIG.values()]
    
    # Instantiate the Leg class
    leg = Leg(
        limits=RS03_LIMITS,
        channel=CAN_CHANNEL,
        host_id=HOST_ID,
        motor_ids=motor_ids
    )
    
    # Instantiate the Safety Monitor
    safety_monitor = SafetyMonitor(joint_limits=JOINT_CONFIG)

    # Instantiate the RL Policy
    direction_vector = [config['direction'] for config in sorted(JOINT_CONFIG.values(), key=lambda x: x['id'])]
    policy = Policy(
        model_path=MODEL_PATH, 
        num_joints=NUM_JOINTS, 
        history_len=HISTORY_LEN, 
        period=5.0,
        default_pos=DEFAULT_POS, 
        direction_vector=direction_vector, 
        action_scale=ACTION_SCALE
    )

    try:
        # Initialize the hardware
        leg.init_leg()
        print("[INFO] Initialization complete.")

        # Read the initial zero state to verify the robot is safely communicative before starting
        print("[INFO] Checking initial hardware state...")
        zero_targets = {mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} for mid in motor_ids}
        initial_state_vector = leg.get_latest_state_vector(
            target_states=zero_targets, 
            kp=0.0, 
            kd=0.0
        )
        
        # Verify measured state is within safe operating bounds defined in JOINT_CONFIG
        safety_monitor.verify_measured_state(initial_state_vector)

        # --- NEW: INITIALIZE FILTER TRACKERS ---
        filtered_pos = {mid: initial_state_vector[mid]['pos'] for mid in motor_ids}
        filtered_vel = {mid: initial_state_vector[mid]['vel'] for mid in motor_ids}

        # Pre-compute the starting position using the initial physical state
        initial_physical_targets = policy.compute_action(initial_state_vector)
        safety_monitor.validate_commanded_targets(initial_physical_targets)
        current_targets = format_targets(initial_physical_targets)

        # --- SETUP LOGGING ---
        log_data = []
        log_headers = ['time']
        for mid in motor_ids:
            # We'll log raw and filtered values for visibility
            log_headers.extend([
                f'meas_pos_{mid}', f'meas_vel_{mid}', 
                f'filt_pos_{mid}', f'filt_vel_{mid}',
                f'cmd_pos_{mid}'
            ])
        # ---------------------

        print(f"[INFO] Entering {LOOP_RATE_HZ}Hz Control loop (Press Ctrl+C to stop)...")
        
        start_time = time.perf_counter()
        loop_counter = 0

        while True:
            loop_start = time.perf_counter()

            # 1. Pipeline out the previous targets and read the latest physical state (Runs at 200Hz)
            state_vector = leg.get_latest_state_vector(
                target_states=current_targets, 
                kp=KP_GAIN, 
                kd=KD_GAIN
            )

            # 2. Hard fault immediately if the RAW physical state has strayed outside physical bounds
            safety_monitor.verify_measured_state(state_vector)

            # --- NEW: 3. Apply Low-Pass Filter (LPF) to positions and velocities ---
            filtered_state_vector = {}
            for mid in motor_ids:
                raw_pos = state_vector[mid]['pos']
                raw_vel = state_vector[mid]['vel']
                
                filtered_pos[mid] = (LPF_ALPHA * raw_pos) + ((1.0 - LPF_ALPHA) * filtered_pos[mid])
                filtered_vel[mid] = (LPF_ALPHA * raw_vel) + ((1.0 - LPF_ALPHA) * filtered_vel[mid])
                
                # Construct the filtered state vector specifically for the policy
                filtered_state_vector[mid] = {
                    'pos': filtered_pos[mid],
                    'vel': filtered_vel[mid]
                }
            # ------------------------------------------------------------------------

            # --- NEW: 4. Compute actions only on policy ticks (50Hz) ---
            if loop_counter % POLICY_UPDATE_INTERVAL == 0:
                # Provide the policy with the CLEANED, FILTERED observations
                physical_targets = policy.compute_action(filtered_state_vector)

                # Validate the commanded targets before applying them
                safety_monitor.validate_commanded_targets(physical_targets)

                # Format targets for the leg API
                current_targets = format_targets(physical_targets)
            # -----------------------------------------------------------

            # --- LOG CURRENT STEP DATA ---
            timestamp = loop_start - start_time
            log_row = {'time': timestamp}
            for mid in motor_ids:
                log_row[f'meas_pos_{mid}'] = state_vector[mid]['pos']
                log_row[f'meas_vel_{mid}'] = state_vector[mid]['vel']
                log_row[f'filt_pos_{mid}'] = filtered_pos[mid]
                log_row[f'filt_vel_{mid}'] = filtered_vel[mid]
                log_row[f'cmd_pos_{mid}']  = current_targets[mid]['pos']
            log_data.append(log_row)
            # -----------------------------

            # 5. Send the validated output targets (Sends the held target at 200Hz)
            leg.set_output_state_vector(
                physical_targets=current_targets, 
                kp=KP_GAIN, 
                kd=KD_GAIN
            )

            # 6. Loop Timing Control (200Hz enforcement)
            elapsed = time.perf_counter() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)
            else:
                # Optional: Uncomment to track real-time overruns
                # pass 
                print(f"[WARN] Loop overrun by {(elapsed - DT)*1000:.2f} ms")
                
            loop_counter += 1

    except SafetyLimitError as e:
        print(f"\n[EMERGENCY STOP] Safety Interlock Tripped: {e}")
    except (HardwareIOError, ActuatorFault, HardwareError) as e:
        print(f"\n[CRITICAL] Hardware Failure: {e}")
        print("Initiating emergency shutdown...")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Stopping RL policy...")
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")
    finally:
        # --- SAVE LOG DATA ---
        if log_data:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rl_log_{timestamp_str}.csv"
            print(f"\n[INFO] Saving {len(log_data)} data points to {filename}...")
            try:
                with open(filename, mode='w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=log_headers)
                    writer.writeheader()
                    writer.writerows(log_data)
                print("[INFO] Log saved successfully.")
            except IOError as e:
                print(f"[ERROR] Failed to save log data: {e}")
        # ---------------------

        leg.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()