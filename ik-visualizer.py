import math
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # Using the working Qt5 backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- 1. Inverse Kinematics ---
def calculate_leg_ik(x, y, z, is_left_stance=True, knee_forward=True):
    """Calculates IK angles (q1, q2, q3) given foot target (x,y,z)."""
    xz_dist_sq = x**2 + z**2
    if xz_dist_sq < 6.25: 
        xz_dist_sq = 6.25
        
    yaw_offset_angle = math.acos(-2.5 / math.sqrt(xz_dist_sq))
    
    if is_left_stance:
        q1 = math.atan2(z, x) + yaw_offset_angle
    else:
        q1 = math.atan2(z, x) - yaw_offset_angle
        
    y_prime = y - 88.5
    z_prime = z * math.cos(q1) - x * math.sin(q1)
    
    L_diag_sq = y_prime**2 + z_prime**2
    D = (L_diag_sq - 102500.0) / 100000.0
    D = max(-1.0, min(1.0, D))
    
    knee_inner_angle = math.acos(D)
    calf_offset = 3.0 * math.pi / 4.0
    
    if knee_forward:
        q3 = knee_inner_angle - calf_offset
    else:
        q3 = -knee_inner_angle - calf_offset
        
    phi = -q3 - calf_offset
    k1 = 200.0 + 250.0 * math.cos(phi)
    k2 = 250.0 * math.sin(phi)
    
    q2 = math.atan2(z_prime, y_prime) - math.atan2(k2, k1)
    
    return q1, q2, q3

# --- 2. Forward Kinematics (For Plotting) ---
def Rx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def Ry(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def get_joint_positions(q1, q2, q3):
    """Calculates 3D coordinates of all joints for drawing the stick figure."""
    v1 = np.array([25.0, 88.5, 0.0])
    v2 = np.array([-41.5, 200.0, 0.0])
    v3 = np.array([14.0, -176.78, -176.78])
    
    R1 = Ry(-q1)
    R2 = Rx(q2)
    R3 = Rx(-q3)
    
    j1 = np.array([0.0, 0.0, 0.0])
    j2 = j1 + R1 @ v1
    j3 = j2 + R1 @ R2 @ v2
    foot = j3 + R1 @ R2 @ R3 @ v3
    
    return np.vstack([j1, j2, j3, foot])

# --- 3. Interactive GUI Setup ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.3) 

init_x = -2.5
init_y = 112.0
init_z = -177.0

q1, q2, q3 = calculate_leg_ik(init_x, init_y, init_z)
joints = get_joint_positions(q1, q2, q3)

line, = ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'o-', color='orange', linewidth=4, markersize=8)
target_scatter = ax.scatter([init_x], [init_y], [init_z], color='red', s=50, label='Target')

ax.set_xlim([-300, 300])
ax.set_ylim([-100, 400])
ax.set_zlim([-400, 100])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3-DOF Leg IK Visualizer')
ax.legend()

# --- NEW: Angle Text Display ---
# We use text2D to pin the text to the window coordinates, so it doesn't move when you rotate the 3D plot
angle_display = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def update_text(q1, q2, q3):
    text_str = (
        f"Joint Angles:\n"
        f"q1 (Yaw):  {math.degrees(q1):>6.1f}°  |  {q1:>5.2f} rad\n"
        f"q2 (Hip):  {math.degrees(q2):>6.1f}°  |  {q2:>5.2f} rad\n"
        f"q3 (Knee): {math.degrees(q3):>6.1f}°  |  {q3:>5.2f} rad"
    )
    angle_display.set_text(text_str)

# Initialize text box
update_text(q1, q2, q3)

# --- 4. Sliders ---
axcolor = 'lightgoldenrodyellow'
ax_x = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_y = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_z = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)

s_x = Slider(ax_x, 'X Target', -200.0, 200.0, valinit=init_x)
s_y = Slider(ax_y, 'Y Target', -50.0, 350.0, valinit=init_y)
s_z = Slider(ax_z, 'Z Target', -450.0, 0.0, valinit=init_z)

def update(val):
    x = s_x.val
    y = s_y.val
    z = s_z.val
    
    q1, q2, q3 = calculate_leg_ik(x, y, z)
    new_joints = get_joint_positions(q1, q2, q3)
    
    line.set_data(new_joints[:, 0], new_joints[:, 1])
    line.set_3d_properties(new_joints[:, 2])
    target_scatter._offsets3d = ([x], [y], [z])
    
    # Update the dynamic text box
    update_text(q1, q2, q3)
    
    fig.canvas.draw_idle()

s_x.on_changed(update)
s_y.on_changed(update)
s_z.on_changed(update)

plt.show()