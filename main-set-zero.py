import sys
import time
from robot.leg import Leg
from robstride.protocol import CommunicationType
from config import JOINT_CONFIG, RS03_LIMITS, CAN_CHANNEL, HOST_ID

def main():
    print("--- RobStride RS03 Zero Point Setter ---")
    print("WARNING: This will overwrite the internal mechanical zero position of the motors.")
    print("Ensure the robot leg is physically held perfectly still at the desired zero coordinates.")
    
    confirm = input("Type 'YES' to proceed: ")
    if confirm != "YES":
        print("Aborting.")
        sys.exit(0)

    # 1. Extract motor IDs from the joint configuration
    motor_ids = [config['id'] for config in JOINT_CONFIG.values()]
    
    # 2. Instantiate the Leg class
    leg = Leg(
        limits=RS03_LIMITS, 
        channel=CAN_CHANNEL, 
        host_id=HOST_ID, 
        motor_ids=motor_ids
    )

    try:
        # Initialize the hardware (starts CAN, enables motors into a passive state)
        leg.init_leg()
        print("\n[INFO] Initialization complete. Motors are enabled and passive.")
        
        # 3. Iterate through joints and set zero
        print("\n[INFO] Transmitting SET_ZERO_POSITION commands...")
        for motor_id in sorted(motor_ids):
            print(f"  -> Sending zero command to Motor ID {motor_id}...")
            
            # Send Communication Type 6 with Byte[0] = 1 as the trigger payload
            leg.robstride.transmit(
                comm_type=CommunicationType.SET_ZERO_POSITION,
                extra_data=HOST_ID,
                destination_id=motor_id,
                data=b'\x01\x00\x00\x00\x00\x00\x00\x00'
            )
            
            # Short sleep to allow the motor's MCU to process the command and flash memory
            time.sleep(0.5)

        # 4. Verification Step
        print("\n[INFO] Verifying new mechanical zero positions...")
        
        # Define a zeroed target state vector to safely query the leg without movement
        zero_targets = {mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} for mid in motor_ids}
        
        # Flush the CAN bus and get the latest state vector
        state_vector = leg.get_latest_state_vector(
            target_states=zero_targets, 
            kp=0.0, 
            kd=0.0
        )
        
        # Check that the reported position is ~0.0 rad
        all_zeroed = True
        TOLERANCE = 0.05 # radians

        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        for motor_id in sorted(motor_ids):
            state = state_vector.get(motor_id, {})
            pos = state.get('pos', 999.0) # Default to a clearly wrong number if missing
            
            if abs(pos) <= TOLERANCE:
                print(f"  {GREEN}[SUCCESS]{RESET} Motor ID {motor_id}: {pos:>8.4f} rad")
            else:
                print(f"  {RED}[FAIL]{RESET}    Motor ID {motor_id}: {pos:>8.4f} rad (Exceeds {TOLERANCE} rad tolerance!)")
                all_zeroed = False
                
        if all_zeroed:
            print("\n{GREEN}[DONE] All motors successfully zeroed and verified.{RESET}")
            print("You may need to power cycle the motors to ensure the calibration is permanently saved.")
        else:
            print("\n{RED}[CRITICAL ERROR] One or more motors failed to verify their zero position.{RESET}")

    except Exception as e:
        print(f"\n{RED}[FATAL] An unexpected error occurred: {e}{RESET}")
    finally:
        # 5. Gracefully shut down the leg and close the CAN bus
        leg.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()