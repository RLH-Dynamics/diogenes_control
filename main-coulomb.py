import sys
import time
import matplotlib
matplotlib.use('Qt5Agg') # Match the backend used in your IK visualizer
import matplotlib.pyplot as plt
from robot.leg import Leg
from config import JOINT_CONFIG, RS03_LIMITS, CAN_CHANNEL, HOST_ID, DT

def main():
    print("--- RS03 Breakaway Torque (Static Friction) Test ---")
    print("NOTE: The RS03 actuators use the CAN bus, not I2C.")
    
    # We will test the first motor defined in the JOINT_CONFIG
    motor_ids = [config['id'] for config in JOINT_CONFIG.values()]
    test_motor_id = motor_ids[0] 
    
    print(f"[INFO] Initializing Leg on {CAN_CHANNEL}...")
    leg = Leg(
        limits=RS03_LIMITS,
        channel=CAN_CHANNEL,
        host_id=HOST_ID,
        motor_ids=motor_ids
    )

    # Test Parameters
    MAX_TEST_TORQUE = 5.0      # N.m - Safety limit to prevent runaway rotation
    TORQUE_RAMP_RATE = 0.5     # N.m per second - Slow increase for high resolution
    MOVEMENT_THRESHOLD = 0.03  # Radians - Threshold to qualify as "breakaway"
    
    # Data recording lists for the plot
    log_time = []
    log_cmd_torque = []
    log_meas_torque = []
    log_position = []

    try:
        # 1. Initialize the CAN bus and enable the motors
        leg.init_leg()
        print(f"[INFO] Motors enabled. Testing Motor ID: {test_motor_id}")
        
        # 2. Read the initial starting position
        zero_targets = {mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} for mid in motor_ids}
        initial_state = leg.get_latest_state_vector(target_states=zero_targets, kp=0.0, kd=0.0)
        
        start_pos = initial_state[test_motor_id].get('pos', 0.0)
        print(f"[INFO] Initial position: {start_pos:.4f} rad")
        
        print("[INFO] Beginning torque ramp. Do not physically touch the actuator...")
        time.sleep(1.0)
        
        start_time = time.perf_counter()
        current_cmd_torque = 0.0
        breakaway_torque = None

        # 3. Main Testing Loop
        while current_cmd_torque <= MAX_TEST_TORQUE:
            loop_start = time.perf_counter()
            elapsed_time = loop_start - start_time
            
            # Increment the commanded torque based on the ramp rate
            current_cmd_torque = TORQUE_RAMP_RATE * elapsed_time
            
            # Prepare MIT targets: pure torque on the test motor, 0.0 Nm on the rest
            targets = {mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} for mid in motor_ids}
            targets[test_motor_id]['torque'] = current_cmd_torque
            
            # Send the torque command and read the immediate hardware feedback
            state = leg.get_latest_state_vector(target_states=targets, kp=0.0, kd=0.0)
            
            curr_pos = state[test_motor_id].get('pos', 0.0)
            meas_torque = state[test_motor_id].get('torque', 0.0)
            disp = curr_pos - start_pos
            
            # Log data for the graph
            log_time.append(elapsed_time)
            log_cmd_torque.append(current_cmd_torque)
            log_meas_torque.append(meas_torque)
            log_position.append(disp) 
            
            print(f"\r[TEST] Cmd: {current_cmd_torque:>5.3f} Nm | Meas: {meas_torque:>5.3f} Nm | Disp: {abs(disp):>6.4f} rad", end="")
            
            # 4. Check for static friction breakaway
            if abs(disp) >= MOVEMENT_THRESHOLD:
                breakaway_torque = current_cmd_torque
                print(f"\n\n[SUCCESS] Breakaway detected! Static friction overcome at {breakaway_torque:.3f} N.m")
                break
                
            # Hardware cycle timing
            elapsed = time.perf_counter() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)

        if breakaway_torque is None:
            print(f"\n[WARN] Reached maximum test torque ({MAX_TEST_TORQUE} N.m) without detecting movement. Is the joint physically locked?")

        # 5. Ramp down safely to prevent sudden jerking
        print("[INFO] Safely ramping torque back to 0 N.m...")
        while current_cmd_torque > 0:
            loop_start = time.perf_counter()
            
            # Ramp down twice as fast as the ramp up
            current_cmd_torque -= (TORQUE_RAMP_RATE * DT * 2) 
            if current_cmd_torque < 0: 
                current_cmd_torque = 0.0
            
            targets = {mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0} for mid in motor_ids}
            targets[test_motor_id]['torque'] = current_cmd_torque
            
            # Pipeline the state down without waiting for feedback
            leg.set_output_state_vector(physical_targets=targets, kp=0.0, kd=0.0)
            
            elapsed = time.perf_counter() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user.")
    except Exception as e:
        print(f"\n[FATAL] A hardware error occurred: {e}")
    finally:
        # Shutdown the CAN bus to secure the actuators
        leg.shutdown()
        
        # 6. Plot the verification graph
        if len(log_time) > 0:
            print("[INFO] Generating validation graph...")
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Left Y-Axis: Torque
            color = 'tab:red'
            ax1.set_xlabel('Time (Seconds)')
            ax1.set_ylabel('Torque (N.m)', color=color)
            ax1.plot(log_time, log_cmd_torque, label='Commanded Torque', color='lightcoral', linestyle='--')
            ax1.plot(log_time, log_meas_torque, label='Measured Feedback Torque', color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')

            # Right Y-Axis: Displacement
            ax2 = ax1.twinx()  
            color = 'tab:blue'
            ax2.set_ylabel('Displacement (Radians)', color=color)  
            ax2.plot(log_time, log_position, label='Joint Displacement', color=color, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(y=MOVEMENT_THRESHOLD, color='blue', linestyle=':', label='Breakaway Threshold')
            ax2.axhline(y=-MOVEMENT_THRESHOLD, color='blue', linestyle=':')
            ax2.legend(loc='lower right')

            if breakaway_torque:
                plt.title(f'RS03 Static Friction Test - Motor ID {test_motor_id}\nBreakaway Torque: {breakaway_torque:.3f} N.m')
            else:
                plt.title(f'RS03 Static Friction Test - Motor ID {test_motor_id}')
                
            fig.tight_layout()  
            plt.savefig('coulomb_breakaway_graph.png')
            print("[INFO] Graph saved to workspace as 'coulomb_breakaway_graph.png'.")
            plt.show()

if __name__ == "__main__":
    main()