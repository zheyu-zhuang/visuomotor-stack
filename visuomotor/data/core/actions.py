"""Canonical action representation semantics shared across integrations."""


def validate_action_rep(action_rep):
    """Normalize and validate a supported action representation."""
    action_rep = str(action_rep)
    if action_rep in ("absolute", "delta"):
        return action_rep
    raise ValueError(
        f"Unsupported action_rep: {action_rep!r}. Supported values are: "
        "'absolute', 'delta'."
    )
