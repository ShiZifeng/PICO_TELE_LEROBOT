import time
import numpy as np
from xrobotoolkit_teleop.common.xr_client import XrClient

xr = XrClient()

while True:
    try:
        left_pose = xr.get_pose_by_name("left_controller")
        right_pose = xr.get_pose_by_name("right_controller")
        head_pose = xr.get_pose_by_name("headset")

        left_trigger = xr.get_key_value_by_name("left_trigger")
        right_trigger = xr.get_key_value_by_name("right_trigger")
        left_grip = xr.get_key_value_by_name("left_grip")
        right_grip = xr.get_key_value_by_name("right_grip")

        print("head:", np.round(head_pose, 3))
        print("left :", np.round(left_pose, 3), "trigger:", round(left_trigger, 3), "grip:", round(left_grip, 3))
        print("right:", np.round(right_pose, 3), "trigger:", round(right_trigger, 3), "grip:", round(right_grip, 3))
        print("-" * 80)

        time.sleep(0.2)

    except KeyboardInterrupt:
        break