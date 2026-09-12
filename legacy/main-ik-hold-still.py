import sys
import time
from robot.leg import Leg
from config import JOINT_CONFIG, RS03_LIMITS, CAN_CHANNEL, HOST_ID, KP_GAIN, KD_GAIN, DT

def main():
    print("[INFO] Setting up Leg for set_output_state_vector position-hold test...")

    # Extract motor IDs from the joint configuration
    motor_ids = [config['id'] for config in JOINT_CONFIG.values()]

    # Instantiate the Leg class
    leg = Leg(
        limits=RS03_LIMITS,
        channel=CAN_CHANNEL,
        host_id=HOST_ID,
        motor_ids=motor_ids
    )

    try:
        # Initialize the hardware (starts CAN, enables motors into a passive state)
        leg.init_leg()
        print("[INFO] Initialization complete.")

        # Step 1: Obtain the current positions of all joints
        print("[INFO] Reading initial joint positions...")
        zero_targets = {
            mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} 
            for mid in motor_ids
        }
        
        # Using kp=0.0 and kd=0.0 guarantees the leg remains limp while we poll the state
        initial_state_vector = leg.get_latest_state_vector(
            target_states=zero_targets, 
            kp=0.0, 
            kd=0.0
        )

        # Build the target dictionary to hold the starting positions
        target_dict = {}
        for motor_id, state in initial_state_vector.items():
            current_pos = state.get('pos', 0.0)
            print(f"  -> Motor ID {motor_id} initial position: {current_pos:>8.4f} rad")
            target_dict[motor_id] = {
                'pos': current_pos,
                'vel': 0.0,      # We want it to stay completely still
                'torque': 0.0    # No feed-forward torque
            }

        print("[INFO] Entering 50Hz position-hold loop (Press Ctrl+C to stop)...")
        
        # Step 2 & 3: Command the leg to stay at the initial position at 50Hz
        while True:
            start_time = time.perf_counter()
            
            # --- CRITICAL FIX ---
            # Flush the unread feedback replies from the previous loop iteration.
            # This prevents the OS SocketCAN buffer from overflowing and tripping the watchdogs.
            leg.robstride.flush_CAN_bus() 
            
            # Command the leg to hold the target positions without polling for a return reply
            leg.set_output_state_vector(
                physical_targets=target_dict, 
                kp=KP_GAIN, 
                kd=KD_GAIN
            )
            
            # Maintain 50Hz loop rate based on the DT from config.py
            elapsed = time.perf_counter() - start_time
            if elapsed < DT:
                time.sleep(DT - elapsed)
            else:
                print(f"[WARN] Loop overrun by {(elapsed - DT)*1000:.2f} ms")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Stopping test...")
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")
    finally:
        # Step 4: Gracefully shut down the leg and close the CAN bus
        leg.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()