"""Development dispatch gates, separate from route outcome labels."""


def rejection_reason(preflight, route, require_full_plan=False):
    if route not in ('E', 'D', 'A'):
        raise ValueError('unknown route: ' + route)
    if route == 'D' and not preflight.get('D_prefix_executable', False):
        return 'no_ik_and_swept_prefix_valid_dock'
    if not require_full_plan:
        return None
    rows = [r for r in preflight.get('records', []) if r.get('route') == route]
    if len(rows) != 1:
        return 'missing_or_ambiguous_full_plan'
    row = rows[0]
    if not (row.get('hard_valid') is True and row.get('features', {}).get('hard_valid') == 1.0
            and row.get('collision', {}).get('valid') is True
            and row.get('source_qpos_unchanged') is True):
        return 'full_plan_not_hard_valid'
    return None
def target_finger_contact(contact, fixture_name, *, handle_only=False):
    """Match the bound fixture and a robot finger, never the palm or a neighbour."""
    if not fixture_name:
        raise ValueError('missing target fixture binding')
    names = [contact.get('geom1') or '', contact.get('geom2') or '']
    for target, finger in (names, names[::-1]):
        if (target.startswith(fixture_name + '_') and
                (not handle_only or 'handle' in target) and
                finger.startswith('gripper0_right_finger')):
            return True
    return False

