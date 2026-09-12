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
    
    if xz_dist_sq < 6.25: 
        xz_dist_sq = 6.25
        
    yaw_offset_angle = math.acos(-2.5 / math.sqrt(xz_dist_sq))
    
    if is_left_stance:
        q1 = math.atan2(z, x) + yaw_offset_angle
    else:
        q1 = math.atan2(z, x) - yaw_offset_angle
        
    # --- 2. Translating to the 2D Leg Plane ---
    y_prime = y - 88.5
    z_prime = z * math.cos(q1) - x * math.sin(q1)
    
    L_diag_sq = y_prime**2 + z_prime**2
    
    # --- 3. Joint 3 (Knee) ---
    D = (L_diag_sq - 102500.0) / 100000.0
    D = max(-1.0, min(1.0, D))
    
    knee_inner_angle = math.acos(D)
    
    calf_offset = 3.0 * math.pi / 4.0
    if knee_forward:
        q3 = knee_inner_angle - calf_offset
    else:
        q3 = -knee_inner_angle - calf_offset
        
    # --- 4. Joint 2 (Hip Pitch) ---
    phi = -q3 - calf_offset
    
    k1 = 200.0 + 250.0 * math.cos(phi)
    k2 = 250.0 * math.sin(phi)
    
    q2 = math.atan2(z_prime, y_prime) - math.atan2(k2, k1)
    
    return q1, q2, q3


def main():
    print("[INFO] Setting up Leg for 3D IK stepping trajectory test...")

    # Extract configuration
    motor_ids = [config['id'] for config in JOINT_CONFIG.values()]
    leg = Leg(
        limits=RS03_LIMITS,
        channel=CAN_CHANNEL,
        host_id=HOST_ID,
        motor_ids=motor_ids
    )
    safety_monitor = SafetyMonitor(joint_limits=JOINT_CONFIG)

    # Trajectory Parameters
    CENTER_X = -2.5
    CENTER_Y = 112.0
    CENTER_Z = -177.0 - 80.0
    
    # Step definition
    STRIDE_X = 0.0      # Lateral stride distance (0 for straight forward)
    STRIDE_Y = 60.0     # Forward stride distance 
    CLEARANCE_Z = 30.0  # Apex height of the foot during the swing phase
    
    CYCLE_PERIOD = 1.5  # Seconds to complete one full step cycle
    STANCE_RATIO = 0.5  # Fraction of the cycle spent on the ground (0.0 to 1.0)

    try:
        # Initialize Hardware
        leg.init_leg()
        print("[INFO] Initialization complete.")

        print("[INFO] Checking initial hardware state...")
        zero_targets = {mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} for mid in motor_ids}
        initial_state_vector = leg.get_latest_state_vector(
            target_states=zero_targets, 
            kp=0.0, 
            kd=0.0
        )
        
        safety_monitor.verify_measured_state(initial_state_vector)

        # Pre-compute the starting position
        q1, q2, q3 = calculate_leg_ik(CENTER_X, CENTER_Y, CENTER_Z)
        current_targets = {
            1: {'pos': q1, 'vel': 0.0, 'torque': 0.0},
            2: {'pos': q2, 'vel': 0.0, 'torque': 0.0},
            3: {'pos': q3, 'vel': 0.0, 'torque': 0.0}
        }

        print(f"[INFO] Entering 50Hz stepping IK loop (Press Ctrl+C to stop)...")
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

            # 3. Compute the new trajectory targets using a synchronized phase
            t = time.perf_counter() - start_time
            cycle_time = t % CYCLE_PERIOD
            phase = cycle_time / CYCLE_PERIOD  # Normalized from 0.0 to 1.0

            if phase < STANCE_RATIO:
                # --- STANCE PHASE ---
                # The foot moves backward relative to the body to propel the robot forward.
                # It goes from +STRIDE/2 to -STRIDE/2 along the ground plane.
                p = phase / STANCE_RATIO  # Normalized stance progress (0.0 to 1.0)
                
                x_offset = (STRIDE_X / 2.0) - (p * STRIDE_X)
                y_offset = (STRIDE_Y / 2.0) - (p * STRIDE_Y)
                z_offset = 0.0  # Foot is flat on the ground
            else:
                # --- SWING PHASE ---
                # The foot lifts and moves forward from -STRIDE/2 to +STRIDE/2.
                p = (phase - STANCE_RATIO) / (1.0 - STANCE_RATIO)  # Normalized swing progress (0.0 to 1.0)
                
                x_offset = -(STRIDE_X / 2.0) + (p * STRIDE_X)
                y_offset = -(STRIDE_Y / 2.0) + (p * STRIDE_Y)
                
                # Sinusoidal profile to create a smooth lifting arc
                z_offset = CLEARANCE_Z * math.sin(p * math.pi)
            
            target_x = CENTER_X + x_offset
            target_y = CENTER_Y + y_offset
            target_z = CENTER_Z + z_offset

            # 4. Compute Inverse Kinematics for the new 3D target
            q1, q2, q3 = calculate_leg_ik(target_x, target_y, target_z, is_left_stance=True, knee_forward=True)
            
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

            # 8. Loop Timing Control
            elapsed = time.perf_counter() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)
            else:
                print(f"[WARN] Loop overrun by {(elapsed - DT)*1000:.2f} ms")

            actual_loop_time = time.perf_counter() - loop_start
            print(f"[INFO] Elapsed (Active): {elapsed * 1000:.2f} ms | Actual Loop (Total): {actual_loop_time * 1000:.2f} ms")

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