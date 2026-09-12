import sys
import time
from robot.leg import Leg
from config import JOINT_CONFIG, RS03_LIMITS, CAN_CHANNEL, HOST_ID

def main():
    print("[INFO] Setting up Leg for IK state reading test...")
    
    # 1. Extract motor IDs from the joint configuration
    motor_ids = [config['id'] for config in JOINT_CONFIG.values()]
    
    # 2. Instantiate the Leg class
    leg = Leg(
        limits=RS03_LIMITS, 
        channel=CAN_CHANNEL, 
        host_id=HOST_ID, 
        motor_ids=motor_ids
    )

    # 3. Define a zeroed target state vector (required by get_latest_state_vector)
    zero_targets = {
        mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} 
        for mid in motor_ids
    }

    try:
        # Initialize the hardware (starts CAN, enables motors into a passive state)
        leg.init_leg()
        print("[INFO] Initialization complete. Entering read loop (Press Ctrl+C to stop)...")
        time.sleep(1) # Brief pause before spamming terminal
        
        while True:
            # Query the latest state. 
            # kp=0.0 and kd=0.0 guarantees the leg remains limp.
            state_vector = leg.get_latest_state_vector(
                target_states=zero_targets, 
                kp=0.0, 
                kd=0.0
            )
            
            # Print the state vector legibly
            # Using carriage return \r and end="" to overwrite the same line or clear terminal
            print("\033[H\033[J", end="") # Clears the terminal screen for a clean read
            print("--- Latest Leg State Vector ---")
            
            # Sort by motor ID so the printing is consistent
            for motor_id in sorted(state_vector.keys()):
                state = state_vector[motor_id]
                print(f"Motor ID: {motor_id}")
                print(f"  Pos:    {state.get('pos', 0.0):>8.4f} rad")
                print(f"  Vel:    {state.get('vel', 0.0):>8.4f} rad/s")
                print(f"  Torque: {state.get('torque', 0.0):>8.4f} N.m")
                print(f"  Temp:   {state.get('temp', 0.0):>8.1f} °C\n")
            
            # Run at roughly 10 Hz for readability
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Stopping test...")
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")
    finally:
        # 4. Gracefully shut down the leg and close the CAN bus
        leg.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()