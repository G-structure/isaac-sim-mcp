"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .crc32 import crc32_core, crc32_of_struct
from .gains import (
    G1_JOINT_LIMITS,
    G1_JOINT_NAMES,
    JointLimit,
    calibrate_gains,
    clamp_effort,
    clamp_velocity,
    get_limits,
    kd_to_drive_damping,
    kp_to_drive_stiffness,
)
from .motor_index_map import (
    PROBE_ISAAC_DOF_ORDER,
    UNITREE_G1_JOINT_NAMES,
    DofLayoutError,
    IndexMaps,
    assert_dof_layout,
    build_index_maps,
)
from .unitree_hg import (
    LOWCMD_SIZE,
    LOWSTATE_SIZE,
    NUM_MOTOR,
    G1JointIndex,
    IMUState,
    LowCmd,
    LowState,
    ModePR,
    MotorCmd,
    MotorState,
)
# Driver last: it depends on the codec + gains + index-map above. Its top-level is
# omni-free (omni/isaacsim/numpy are imported lazily in install()), so importing
# this package stays importable wherever those leaf modules are.
from .g1_lowcmd_driver import ControlMode, G1LowCmdDriver

__all__ = [
    "G1_JOINT_LIMITS",
    "G1_JOINT_NAMES",
    "JointLimit",
    "calibrate_gains",
    "clamp_effort",
    "clamp_velocity",
    "get_limits",
    "kd_to_drive_damping",
    "kp_to_drive_stiffness",
    "crc32_core",
    "crc32_of_struct",
    "UNITREE_G1_JOINT_NAMES",
    "PROBE_ISAAC_DOF_ORDER",
    "IndexMaps",
    "DofLayoutError",
    "build_index_maps",
    "assert_dof_layout",
    "ControlMode",
    "G1LowCmdDriver",
    "NUM_MOTOR",
    "ModePR",
    "G1JointIndex",
    "MotorCmd",
    "MotorState",
    "IMUState",
    "LowCmd",
    "LowState",
    "LOWCMD_SIZE",
    "LOWSTATE_SIZE",
]
