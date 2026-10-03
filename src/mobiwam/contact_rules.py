"""Versioned monitoring exemptions. No physics or collision-mask writes."""
RULE_VERSION = "DR-v0.4-R3-exact-finger-pad"
FINGER_PAD_PAIR = frozenset(("gripper0_right_finger1_pad_collision",
                             "gripper0_right_finger2_pad_collision"))


def exempt_finger_pad_pair(a, b):
    return isinstance(a, str) and isinstance(b, str) and a != b and frozenset((a, b)) == FINGER_PAD_PAIR


def allowed_contact(a, b, phase, target):
    if exempt_finger_pad_pair(a, b):
        return True
    a, b = a or "", b or ""
    if ("floor" in a and b.startswith("mobilebase0_")) or ("floor" in b and a.startswith("mobilebase0_")):
        return True
    return bool(target and phase == "manipulate" and
                (("finger" in a and b.startswith(target)) or
                 ("finger" in b and a.startswith(target))))
