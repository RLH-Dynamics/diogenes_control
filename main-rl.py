import sys
import time
import numpy as np
from robot.leg import Leg
from control.policy import Policy
from utils.safety import SafetyMonitor
from utils.exceptions import HardwareIOError, ActuatorFault, HardwareError, SafetyLimitError
from config import (
    JOINT_CONFIG, RS03_LIMITS, CAN_CHANNEL, HOST_ID, 
    KP_GAIN, KD_GAIN, DT, MODEL_PATH, NUM_JOINTS, 
    HISTORY_LEN, DEFAULT_POS, ACTION_SCALE
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
    print("[INFO] Setting up Leg for RL policy control loop...")

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
        period=DT, 
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

        # Pre-compute the starting position using the initial physical state
        initial_physical_targets = policy.compute_action(initial_state_vector)
        safety_monitor.validate_commanded_targets(initial_physical_targets)
        current_targets = format_targets(initial_physical_targets)

        # --- High-Frequency Control Setup ---
        policy_hz = int(1.0 / DT)
        control_hz = 200.0
        control_dt = 1.0 / control_hz
        policy_update_interval = int(control_hz / policy_hz)

        print(f"[INFO] Entering {control_hz}Hz Control / {policy_hz}Hz Policy loop (Press Ctrl+C to stop)...")
        
        loop_counter = 0
        previous_physical_targets = np.array(initial_physical_targets, dtype=np.float32)
        new_physical_targets = np.array(initial_physical_targets, dtype=np.float32)
        
        start_time = time.perf_counter()

        while True:
            loop_start = time.perf_counter()

            # 1. Pipeline out the previous targets and read the latest physical state
            state_vector = leg.get_latest_state_vector(
                target_states=current_targets, 
                kp=KP_GAIN, 
                kd=KD_GAIN
            )

            # 2. Hard fault immediately if the robot has strayed outside physical bounds
            safety_monitor.verify_measured_state(state_vector)

            # 3. Compute the new actions from the ONNX policy at the slower policy frequency
            if loop_counter % policy_update_interval == 0:
                previous_physical_targets = np.copy(new_physical_targets)
                new_physical_targets = np.array(policy.compute_action(state_vector), dtype=np.float32)

            # 4. Interpolate actions to send smoothly at the faster control frequency
            # Calculate interpolation factor (alpha ranges from 0.0 to 1.0 between policy ticks)
            denominator = float(max(1, policy_update_interval - 1))
            interpolation_alpha = (loop_counter % policy_update_interval) / denominator
            
            interpolated_targets = (1.0 - interpolation_alpha) * previous_physical_targets + interpolation_alpha * new_physical_targets

            # 5. Validate the commanded targets before applying them
            safety_monitor.validate_commanded_targets(interpolated_targets.tolist())

            # 6. Format targets for the leg API
            current_targets = format_targets(interpolated_targets.tolist())

            # 7. Send the validated output targets
            leg.set_output_state_vector(
                physical_targets=current_targets, 
                kp=KP_GAIN, 
                kd=KD_GAIN
            )

            # 8. Loop Timing Control
            elapsed = time.perf_counter() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
            else:
                # Optional: Uncomment to track real-time overruns
                pass 
                # print(f"[WARN] Loop overrun by {(elapsed - control_dt)*1000:.2f} ms")

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
        leg.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()