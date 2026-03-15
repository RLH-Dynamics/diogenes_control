import sys
import time
import math
from robot.leg import Leg
from utils.safety import SafetyMonitor
from utils.exceptions import HardwareIOError, ActuatorFault, HardwareError, SafetyLimitError
from config import JOINT_CONFIG, RS03_LIMITS, CAN_CHANNEL, HOST_ID, KP_GAIN, KD_GAIN, DT

def calculate_leg_ik(x: float, y: float, z: float, is_left_stance: bool = True, knee_forward: bool = True) -> tuple[float, float, float]:
    """
    Calculates the inverse kinematics for a 3-DOF robot leg.
    
    Args:
        x, y, z: Target center of the ball foot relative to Joint 1.
        is_left_stance: Toggles the yaw calculation sign (elbow left vs elbow right).
        knee_forward: Toggles the knee bend direction.
        
    Returns:
        A tuple of (q1, q2, q3) representing the joint angles in radians.
    """
    # --- 1. Joint 1 (Yaw) ---
    xz_dist_sq = x**2 + z**2
    
    # Safety clamp for the yaw domain to prevent division by zero or math domain errors
    # 2.5 is the absolute lateral offset, so 2.5^2 = 6.25
    if xz_dist_sq < 6.25: 
        xz_dist_sq = 6.25
        
    yaw_offset_angle = math.acos(-2.5 / math.sqrt(xz_dist_sq))
    
    # Apply the correct sign for the desired leg configuration
    if is_left_stance:
        q1 = math.atan2(z, x) + yaw_offset_angle
    else:
        q1 = math.atan2(z, x) - yaw_offset_angle
        
    # --- 2. Translating to the 2D Leg Plane ---
    y_prime = y - 88.5
    z_prime = z * math.cos(q1) - x * math.sin(q1)
    
    L_diag_sq = y_prime**2 + z_prime**2
    
    # --- 3. Joint 3 (Knee) ---
    # 102500.0 is L_thigh^2 + L_calf^2 (200^2 + 250^2)
    # 100000.0 is 2 * L_thigh * L_calf (2 * 200 * 250)
    D = (L_diag_sq - 102500.0) / 100000.0
    
    # Clamp D to [-1.0, 1.0] to prevent acos domain errors from unreachable targets
    D = max(-1.0, min(1.0, D))
    
    knee_inner_angle = math.acos(D)
    
    # Apply the correct sign for the desired knee bend, minus the calf's resting offset
    calf_offset = 3.0 * math.pi / 4.0
    if knee_forward:
        q3 = knee_inner_angle - calf_offset
    else:
        q3 = -knee_inner_angle - calf_offset
        
    # --- 4. Joint 2 (Hip Pitch) ---
    # Calculate the absolute knee angle contribution
    phi = -q3 - calf_offset
    
    k1 = 200.0 + 250.0 * math.cos(phi)
    k2 = 250.0 * math.sin(phi)
    
    q2 = math.atan2(z_prime, y_prime) - math.atan2(k2, k1)
    
    return q1, q2, q3

def main():
    print("[INFO] Setting up Leg for linear IK tracking test...")

    # Extract configuration
    motor_ids = [config['id'] for config in JOINT_CONFIG.values()]
    leg = Leg(
        limits=RS03_LIMITS,
        channel=CAN_CHANNEL,
        host_id=HOST_ID,
        motor_ids=motor_ids
    )
    safety_monitor = SafetyMonitor(joint_limits=JOINT_CONFIG)

    # Trajectory Parameters (using stable center defaults from ik-visualizer.py)
    CENTER_X = -2.5
    CENTER_Y = 112.0
    CENTER_Z = -177.0 - 150.0
    STROKE_LENGTH_Y = 30.0  # Total back-to-front stroke length (modulate as needed)
    CYCLE_PERIOD = 4.0      # Seconds to complete one full back-and-forth stroke

    try:
        # Initialize Hardware
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

        # Pre-compute the starting position
        q1, q2, q3 = calculate_leg_ik(CENTER_X, CENTER_Y, CENTER_Z)
        current_targets = {
            1: {'pos': q1, 'vel': 0.0, 'torque': 0.0},
            2: {'pos': q2, 'vel': 0.0, 'torque': 0.0},
            3: {'pos': q3, 'vel': 0.0, 'torque': 0.0}
        }

        print(f"[INFO] Entering 50Hz linear IK loop (Press Ctrl+C to stop)...")
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

            # 3. Compute the new trajectory target using a smooth sine wave generator
            t = time.perf_counter() - start_time
            y_offset = (STROKE_LENGTH_Y / 2.0) * math.sin(2.0 * math.pi * t / CYCLE_PERIOD)
            target_y = CENTER_Y + y_offset

            # 4. Compute Inverse Kinematics for the new target
            q1, q2, q3 = calculate_leg_ik(CENTER_X, target_y, CENTER_Z, is_left_stance=True, knee_forward=True)
            
            # The SafetyMonitor expects a simple array ordered explicitly by ID: [hip (1), thigh (2), knee (3)]
            commanded_array = [q1, q2, q3]

            # 5. Validate the commanded targets before applying them
            safety_monitor.validate_commanded_targets(commanded_array)

            # 6. Format targets for the leg API
            current_targets = {
                1: {'pos': q1, 'vel': 0.0, 'torque': 0.0},
                2: {'pos': q2, 'vel': 0.0, 'torque': 0.0},
                3: {'pos': q3, 'vel': 0.0, 'torque': 0.0}
            }

            # 7. Send the validated output targets
            leg.set_output_state_vector(
                physical_targets=current_targets, 
                kp=KP_GAIN, 
                kd=KD_GAIN
            )

            # 8. Loop Timing Control (50Hz defined by DT)
            elapsed = time.perf_counter() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)
            else:
                print(f"[WARN] Loop overrun by {(elapsed - DT)*1000:.2f} ms")

    except SafetyLimitError as e:
        print(f"\n[EMERGENCY STOP] Safety Interlock Tripped: {e}")
    except (HardwareIOError, ActuatorFault, HardwareError) as e:
        print(f"\n[CRITICAL] Hardware Failure: {e}")
        print("Initiating emergency shutdown...")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Stopping test...")
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")
    finally:
        leg.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()