import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use the anti-grain geometry backend (no GUI required)
import matplotlib.pyplot as plt

def main():
    # 1. Automatically find the most recent CSV log file
    log_files = glob.glob("rl_log_*.csv")
    if not log_files:
        print("Error: Could not find any log files matching 'rl_log_*.csv'.")
        print("Make sure you have run main-rl.py first.")
        return
    
    # Sort files by modification time, get the latest
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Loading data from: {latest_log}")
    
    # Load the CSV data using pandas
    df = pd.read_csv(latest_log)
    time_steps = df['time']

    # Extract motor IDs dynamically from the CSV headers
    motor_ids = []
    for col in df.columns:
        if col.startswith('meas_pos_'):
            motor_ids.append(col.split('_')[-1])

    # 2. Setup the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Plot Joint Positions ---
    for mid in motor_ids:
        # Plot Measured Position
        axs[0].plot(time_steps, df[f'meas_pos_{mid}'], label=f'Motor {mid} (Meas)')
        # Plot Commanded Position with dashed lines
        axs[0].plot(time_steps, df[f'cmd_pos_{mid}'], label=f'Motor {mid} (Cmd)', linestyle='--')
    
    axs[0].set_title('Robot Joint Positions Tracking')
    axs[0].set_ylabel('Position (rad)')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # --- Plot Joint Velocities ---
    for mid in motor_ids:
        axs[1].plot(time_steps, df[f'meas_vel_{mid}'], label=f'Motor {mid} (Meas)')
        
    axs[1].set_title('Robot Joint Velocities over Time')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (rad/s)')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust layout to prevent clipping of the external legend
    plt.tight_layout()
    
    # 3. Save the plot to a file matching the log file's name
    output_filename = latest_log.replace('.csv', '.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to '{output_filename}'")

if __name__ == "__main__":
    main()