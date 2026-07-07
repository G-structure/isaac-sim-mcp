"""Vendored pure-Python mirror of the ``unitree_hg`` message contract.

This is the *in-process codec* the Isaac Sim G1 driver uses to pack/unpack and
CRC-validate LowCmd/LowState exactly as the real robot does, over the packed
unitree_sdk2 C-struct memory layout. It deliberately has no ROS/rclpy dependency
so it runs under plain ``python3`` for tests and layout validation.

The real ``unitree_hg`` / ``unitree_go`` / ``unitree_ros2`` message *packages*
(used for the actual DDS transport) are colcon-built inside the session
Dockerfile by a separate component; this module only provides the codec and CRC
that both sides must agree on byte-for-byte.

Layout provenance: field names/types/array lengths follow the unitree_hg ``.msg``
files; the packed byte layout (natural C alignment, little-endian) follows the
unitree_sdk2 hg IDL structs. LowCmd is 1004 bytes = 251 words -> 250 CRC words,
matching the ``(sizeof>>2)-1`` fact.

TODO(probe:crc-capture): the packed byte layout AND the CRC MUST be validated
against a recorded real LowCmd/LowState capture before trusting this on
hardware. Least-certain fields: the MotorState tail
(temperature/vol/sensor/motorstate/reserve) and IMUState.temperature -- these do
not affect the LowCmd path we drive but do affect LowState CRC verification.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple

from .crc32 import crc32_of_struct

NUM_MOTOR = 35  # unitree_hg fixes the array at 35 even though the G1 has 29 DoF


class ModePR(IntEnum):
    """LowCmd/LowState ``mode_pr`` command space (unitree_sdk2 convention)."""

    PR = 0  # serial pitch/roll ankle space, matches the URDF joints
    AB = 1  # parallel A/B linkage motor space


class G1JointIndex(IntEnum):
    """G1 29-DoF joint order (unitree_sdk2 G1JointIndex 0-28).

    TODO(probe:crc-capture): verify this index order against the on-image
    unitree_sdk2 headers / a live LowState mapping.
    """

    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


# ── Packed struct formats (little-endian, natural C alignment) ─────────────
# '<' means struct inserts no padding, so alignment gaps are spelled explicitly
# with 'x' bytes to reproduce the in-memory unitree_sdk2 struct byte-for-byte.

_MOTOR_CMD = struct.Struct("<B3xfffffI")           # 28 bytes
_MOTOR_STATE = struct.Struct("<B3xffffhhIIII4I")   # 56 bytes
_IMU_STATE = struct.Struct("<4f3f3f3fh2x")         # 56 bytes
_LOWCMD_HEAD = struct.Struct("<BB2x")              # 4 bytes
_LOWCMD_TAIL = struct.Struct("<4II")               # reserve[4] + crc = 20 bytes
_LOWSTATE_HEAD = struct.Struct("<IIBB2xI")         # 16 bytes
_LOWSTATE_TAIL = struct.Struct("<40s4II")          # wireless + reserve[4] + crc = 60 bytes

LOWCMD_SIZE = _LOWCMD_HEAD.size + NUM_MOTOR * _MOTOR_CMD.size + _LOWCMD_TAIL.size
LOWSTATE_SIZE = (
    _LOWSTATE_HEAD.size + _IMU_STATE.size + NUM_MOTOR * _MOTOR_STATE.size + _LOWSTATE_TAIL.size
)

assert _MOTOR_CMD.size == 28
assert _MOTOR_STATE.size == 56
assert _IMU_STATE.size == 56
assert LOWCMD_SIZE == 1004, LOWCMD_SIZE
assert LOWSTATE_SIZE == 2092, LOWSTATE_SIZE


@dataclass
class MotorCmd:
    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    reserve: int = 0

    def pack(self) -> bytes:
        return _MOTOR_CMD.pack(
            self.mode & 0xFF, self.q, self.dq, self.tau, self.kp, self.kd,
            self.reserve & 0xFFFFFFFF,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "MotorCmd":
        return cls(*_MOTOR_CMD.unpack(buf))


@dataclass
class MotorState:
    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    ddq: float = 0.0
    tau_est: float = 0.0
    temperature: Tuple[int, int] = (0, 0)
    vol: int = 0
    sensor: Tuple[int, int] = (0, 0)
    motorstate: int = 0
    reserve: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def pack(self) -> bytes:
        return _MOTOR_STATE.pack(
            self.mode & 0xFF, self.q, self.dq, self.ddq, self.tau_est,
            self.temperature[0], self.temperature[1],
            self.vol & 0xFFFFFFFF,
            self.sensor[0] & 0xFFFFFFFF, self.sensor[1] & 0xFFFFFFFF,
            self.motorstate & 0xFFFFFFFF,
            *(r & 0xFFFFFFFF for r in self.reserve),
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "MotorState":
        f = _MOTOR_STATE.unpack(buf)
        return cls(
            f[0], f[1], f[2], f[3], f[4], (f[5], f[6]), f[7], (f[8], f[9]),
            f[10], (f[11], f[12], f[13], f[14]),
        )


@dataclass
class IMUState:
    # TODO(probe:quaternion-order): component order is Unitree's (w, x, y, z);
    # confirm at runtime, it is not encoded in the .msg.
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gyroscope: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    accelerometer: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    temperature: int = 0

    def pack(self) -> bytes:
        return _IMU_STATE.pack(
            *self.quaternion, *self.gyroscope, *self.accelerometer, *self.rpy,
            self.temperature,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "IMUState":
        f = _IMU_STATE.unpack(buf)
        return cls(f[0:4], f[4:7], f[7:10], f[10:13], f[13])


@dataclass
class LowCmd:
    mode_pr: int = 0
    mode_machine: int = 0
    motor_cmd: List[MotorCmd] = field(default_factory=lambda: [MotorCmd() for _ in range(NUM_MOTOR)])
    reserve: Tuple[int, int, int, int] = (0, 0, 0, 0)
    crc: int = 0

    def pack(self) -> bytes:
        if len(self.motor_cmd) != NUM_MOTOR:
            raise ValueError(f"motor_cmd must have {NUM_MOTOR} entries, got {len(self.motor_cmd)}")
        parts = [_LOWCMD_HEAD.pack(self.mode_pr & 0xFF, self.mode_machine & 0xFF)]
        parts.extend(m.pack() for m in self.motor_cmd)
        parts.append(_LOWCMD_TAIL.pack(*(r & 0xFFFFFFFF for r in self.reserve), self.crc & 0xFFFFFFFF))
        return b"".join(parts)

    @classmethod
    def unpack(cls, buf: bytes) -> "LowCmd":
        if len(buf) != LOWCMD_SIZE:
            raise ValueError(f"LowCmd buffer must be {LOWCMD_SIZE} bytes, got {len(buf)}")
        off = _LOWCMD_HEAD.size
        mode_pr, mode_machine = _LOWCMD_HEAD.unpack_from(buf, 0)
        motors = []
        for _ in range(NUM_MOTOR):
            motors.append(MotorCmd.unpack(buf[off:off + _MOTOR_CMD.size]))
            off += _MOTOR_CMD.size
        r0, r1, r2, r3, crc = _LOWCMD_TAIL.unpack_from(buf, off)
        return cls(mode_pr, mode_machine, motors, (r0, r1, r2, r3), crc)

    def compute_crc(self) -> int:
        return crc32_of_struct(self.pack())

    def with_crc(self) -> "LowCmd":
        self.crc = self.compute_crc()
        return self

    def verify_crc(self) -> bool:
        return self.crc == self.compute_crc()


@dataclass
class LowState:
    version: Tuple[int, int] = (0, 0)
    mode_pr: int = 0
    mode_machine: int = 0
    tick: int = 0
    imu_state: IMUState = field(default_factory=IMUState)
    motor_state: List[MotorState] = field(default_factory=lambda: [MotorState() for _ in range(NUM_MOTOR)])
    wireless_remote: bytes = b"\x00" * 40
    reserve: Tuple[int, int, int, int] = (0, 0, 0, 0)
    crc: int = 0

    def pack(self) -> bytes:
        if len(self.motor_state) != NUM_MOTOR:
            raise ValueError(f"motor_state must have {NUM_MOTOR} entries, got {len(self.motor_state)}")
        parts = [
            _LOWSTATE_HEAD.pack(
                self.version[0] & 0xFFFFFFFF, self.version[1] & 0xFFFFFFFF,
                self.mode_pr & 0xFF, self.mode_machine & 0xFF, self.tick & 0xFFFFFFFF,
            ),
            self.imu_state.pack(),
        ]
        parts.extend(m.pack() for m in self.motor_state)
        wireless = bytes(self.wireless_remote[:40]).ljust(40, b"\x00")
        parts.append(_LOWSTATE_TAIL.pack(wireless, *(r & 0xFFFFFFFF for r in self.reserve), self.crc & 0xFFFFFFFF))
        return b"".join(parts)

    @classmethod
    def unpack(cls, buf: bytes) -> "LowState":
        if len(buf) != LOWSTATE_SIZE:
            raise ValueError(f"LowState buffer must be {LOWSTATE_SIZE} bytes, got {len(buf)}")
        v0, v1, mode_pr, mode_machine, tick = _LOWSTATE_HEAD.unpack_from(buf, 0)
        off = _LOWSTATE_HEAD.size
        imu = IMUState.unpack(buf[off:off + _IMU_STATE.size])
        off += _IMU_STATE.size
        motors = []
        for _ in range(NUM_MOTOR):
            motors.append(MotorState.unpack(buf[off:off + _MOTOR_STATE.size]))
            off += _MOTOR_STATE.size
        wireless, r0, r1, r2, r3, crc = _LOWSTATE_TAIL.unpack_from(buf, off)
        return cls((v0, v1), mode_pr, mode_machine, tick, imu, motors, wireless, (r0, r1, r2, r3), crc)

    def compute_crc(self) -> int:
        return crc32_of_struct(self.pack())

    def with_crc(self) -> "LowState":
        self.crc = self.compute_crc()
        return self

    def verify_crc(self) -> bool:
        return self.crc == self.compute_crc()


__all__ = [
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
