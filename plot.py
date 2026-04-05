"""
How to use this script from the terminal:

1. Automatic detection (loads the most recent 'rl_log_*.csv' in the current directory):
   python plot.py

2. Specify an absolute (or relative) path to a specific CSV file:
   python plot.py -f /absolute/path/to/your/rl_log_custom.csv
   # or
   python plot.py --file ../relative/path/to/rl_log_custom.csv

3. View the help menu:
   python plot.py -h
"""

import os
import glob
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg') # Use the anti-grain geometry backend (no GUI required)
import matplotlib.pyplot as plt

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plot robot joint data from a CSV log.")
    parser.add_argument('-f', '--file', type=str, 
                        help='Absolute (or relative) path to a specific CSV log file.')
    args = parser.parse_args()

    # 1. Determine which file to load
    if args.file:
        # User provided a file path via the terminal
        if not os.path.isfile(args.file):
            print(f"Error: The specified file does not exist: {args.file}")
            return
        target_log = args.file
        print(f"Loading data from specified file: {target_log}")
    else:
        # Automatically find the most recent CSV log file
        log_files = glob.glob("rl_log_*.csv")
        if not log_files:
            print("Error: Could not find any log files matching 'rl_log_*.csv'.")
            print("Make sure you have run main-rl.py first, or specify a file path using --file.")
            return
        
        # Sort files by modification time, get the latest
        target_log = max(log_files, key=os.path.getctime)
        print(f"Loading data from most recent file: {target_log}")
    
    # Load the CSV data using pandas
    df = pd.read_csv(target_log)
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
    # If an absolute path is provided, the PNG will be saved in that same absolute directory.
    output_filename = target_log.replace('.csv', '.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to '{output_filename}'")

if __name__ == "__main__":
    main()