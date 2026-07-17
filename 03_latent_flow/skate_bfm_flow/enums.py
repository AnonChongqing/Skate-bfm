from enum import IntEnum


class SkateMode(IntEnum):
    PUSH = 0
    MOUNT = 1
    STEER = 2
    DISMOUNT = 3
    RECOVER = 4


MODE_NAMES = tuple(mode.name.lower() for mode in SkateMode)
