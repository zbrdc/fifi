"""Helpers for disposable systemd scope tests only."""

import errno
import subprocess


def scope_state(unit):
    text = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "Id",
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "InvocationID",
            "-p",
            "ControlGroup",
        ],
        text=True,
        timeout=5,
    )
    return dict(line.split("=", 1) for line in text.splitlines())


def scope_terminated(data, unit, invocation, populated):
    if data.get("Id") != unit:
        raise ValueError("scope unit changed")
    load = data.get("LoadState")
    if load == "loaded":
        if data.get("InvocationID") != invocation:
            raise ValueError("scope invocation changed")
        if data.get("ActiveState") not in ("inactive", "failed"):
            return False
    elif load == "not-found":
        if data.get("InvocationID") not in ("", invocation):
            raise ValueError("removed scope identity changed")
    else:
        raise ValueError("scope status ambiguous")
    if populated is not None and type(populated) is not bool:
        raise ValueError("scope population ambiguous")
    return populated is not True


def scope_can_stop(data, unit, invocation):
    return (
        bool(invocation)
        and data.get("Id") == unit
        and data.get("LoadState") == "loaded"
        and data.get("InvocationID") == invocation
    )


def group_populated(group):
    try:
        fields = dict(
            line.split() for line in (group / "cgroup.events").read_text().splitlines()
        )
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ENODEV):
            try:
                group.stat()
            except OSError as status_error:
                if status_error.errno == errno.ENOENT:
                    return None
                raise
        raise
    if fields.get("populated") not in ("0", "1"):
        raise ValueError("scope population unavailable")
    return fields["populated"] == "1"
