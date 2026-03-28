import sys
import time
from robot.leg import Leg
from utils.exceptions import HardwareIOError, HardwareError
from config import RS03_LIMITS, CAN_CHANNEL, HOST_ID

def main():
    TEST_MOTOR_ID = 1
    
    print(f"--- Coulomb Static Friction Test ---")
    print(f"Testing Actuator CAN ID: {TEST_MOTOR_ID}")
    print("WARNING: Ensure the actuator is free to move slightly and is not locked.")
    
    confirm = input("Type 'YES' to proceed: ")
    if confirm != "YES":
        print("Aborting.")
        sys.exit(0)

    # Instantiate the Leg class strictly for the single test motor in TORQUE mode
    leg = Leg(
        limits=RS03_LIMITS,
        channel=CAN_CHANNEL,
        host_id=HOST_ID,
        motor_ids=[TEST_MOTOR_ID],
        control_mode='TORQUE' 
    )

    # Test Parameters
    TORQUE_STEP = 0.01        # N.m to increase per loop iteration
    LOOP_RATE_HZ = 50.0       # 50 Hz loop (20ms)
    DT = 1.0 / LOOP_RATE_HZ
    MAX_TORQUE = 5.0          # N.m absolute safety limit for the test
    MOVEMENT_THRESHOLD = 0.02 # Radians (~1.1 degrees) of change to consider as "movement"

    try:
        # Initialize the hardware
        leg.init_leg()
        print("\n[INFO] Initialization complete. Motor is in TORQUE mode.")
        
        # 1. Read the initial resting position using our new exposed method
        print("[INFO] Polling initial mechanical position...")
        initial_pos, _ = leg.get_position_and_velocity(motor_id=TEST_MOTOR_ID)
        print(f"  -> Initial Position: {initial_pos:.4f} rad")

        print("\n[INFO] Beginning torque ramp-up...")
        current_torque = 0.0
        static_friction = None
        
        # 2. Ramp up the torque until movement is detected
        while current_torque <= MAX_TORQUE:
            loop_start = time.perf_counter()

            # Command the new torque
            leg.set_output_torques({TEST_MOTOR_ID: current_torque})
            
            # Poll the current position and velocity
            current_pos, current_vel = leg.get_position_and_velocity(motor_id=TEST_MOTOR_ID)
            
            # Print status to the console (overwriting the same line)
            print(f"\r[TESTING] Torque: {current_torque:.3f} N.m | Pos: {current_pos:.4f} rad | Vel: {current_vel:.4f} rad/s", end="")
            
            # Check if the position has deviated past our movement threshold
            if abs(current_pos - initial_pos) >= MOVEMENT_THRESHOLD:
                static_friction = current_torque
                print(f"\n\n[SUCCESS] Movement detected!")
                print(f"  -> Breakaway Position: {current_pos:.4f} rad")
                print(f"  -> Breakaway Velocity: {current_vel:.4f} rad/s")
                print(f"  -> Estimated Static Friction: {static_friction:.3f} N.m")
                break
                
            current_torque += TORQUE_STEP

            # Maintain loop timing
            elapsed = time.perf_counter() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)

        if static_friction is None:
            print(f"\n\n[WARN] Reached maximum test torque ({MAX_TORQUE} N.m) without detecting movement.")

        # 3. Safely ramp the torque back down to 0.0 to prevent a sudden snap back
        print("\n[INFO] Ramping torque back down to zero...")
        while current_torque > 0.0:
            current_torque -= (TORQUE_STEP * 5) # Faster ramp down
            if current_torque < 0.0: 
                current_torque = 0.0
            leg.set_output_torques({TEST_MOTOR_ID: current_torque})
            time.sleep(0.01)

    except (HardwareIOError, HardwareError) as e:
        print(f"\n[CRITICAL] Hardware Failure: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Test aborted by user.")
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")
    finally:
        leg.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()