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
