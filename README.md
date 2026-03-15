# TO DO:
- Define, add, and subtract absolut encoder offsets. We are not doing that right now!
- Implement spin-wait loop and thread priority to improve timing performance.
- Implement a time over- or under-run exception.
- Get an equivalent inverse-kinematics solution working.

- Write a function to validate that the hardware watchdogs did infact get set up properly.

- Write main-ik-linear.py:
    - Ensure that the safety manager is running so we don't accidentally go out of bounds.
        - Double check this on startup!
    - Generate a basic front-to-back, side-to-side, and up-down linear gait.
    - Follow the gait at a 50Hz loop rate.