"""Compare a measured passage width with a wheelchair width (both in meters)."""
from math import isfinite


def compare_width(channel_width_m, wheelchair_width_m):
    """Return a message; equality does not pass. Missing measurements are unknown."""
    if not isfinite(wheelchair_width_m) or wheelchair_width_m <= 0:
        raise ValueError("Wheelchair width must be a positive finite number in meters")
    if channel_width_m is None or not isfinite(channel_width_m) or channel_width_m <= 0:
        return "Unable to determine: no valid passage width."
    return "Can pass!" if channel_width_m > wheelchair_width_m else "Cannot pass"


if __name__ == "__main__":
    # 测试数据：前端接入后，直接向 compare_width 传入真实数据。
    for channel, wheelchair in [(1.2, 0.65), (1.2, 1.2), (1.2, 1.3), (None, 0.65)]:
        print(f"Passage width={channel} m, wheelchair width={wheelchair} m: {compare_width(channel, wheelchair)}")
