"""Read-only external-volume gate. Never create a missing mount or fall back."""
import json
import os
from pathlib import Path
import plistlib
import subprocess

HERE = Path(__file__).resolve().parent


def validate_volume(info, mounted, available, plan):
    """Check observed diskutil/statvfs facts; separate for offline tests."""
    if not mounted:
        raise ValueError("Dedicated Models volume is not mounted; no fallback")
    if info.get("Internal") is not False:
        raise ValueError("Models storage must be verified external")
    if info.get("VolumeName") != plan["mount_name"]:
        raise ValueError("Wrong external volume; X9 is not the selected destination")
    if info.get("MountPoint") != plan["required_mount_path"]:
        raise ValueError("Models mount path does not match the frozen plan")
    if type(available) is not int or available < plan["launch_gate_free_bytes"]:
        raise ValueError("External free space is below the frozen launch gate")
    return {"mount": info["MountPoint"], "available_bytes": available,
            "required_free_bytes": plan["launch_gate_free_bytes"],
            "remaining_after_reservations_bytes":
                available - plan["persistent_phase_required_free_bytes"],
            "status": "storage_gate_passed_only", "execution_authorized": False}


def check_storage(plan=None):
    """Inspect the existing macOS mount, without creating any directories."""
    if plan is None:
        plan = json.loads((HERE / "storage-plan.json").read_text())
    mount = Path(plan["required_mount_path"])
    if mount.is_symlink() or not os.path.ismount(mount):
        raise ValueError("Dedicated Models volume missing or redirected; no fallback")
    target = Path(plan["planned_experiment_root"])
    if target.resolve() == mount.resolve() or mount.resolve() not in target.resolve().parents:
        raise ValueError("Experiment path escapes the dedicated Models volume")
    for value in plan["runtime_storage_environment"].values():
        if mount.resolve() not in Path(value).resolve().parents:
            raise ValueError("Cache or temporary path escapes the Models volume")
    output = subprocess.check_output(
        ["diskutil", "info", "-plist", str(mount)], stderr=subprocess.PIPE)
    info = plistlib.loads(output)
    stat = os.statvfs(mount)
    return validate_volume(info, True, stat.f_bavail * stat.f_frsize, plan)


if __name__ == "__main__":
    print(json.dumps(check_storage(), indent=2))
