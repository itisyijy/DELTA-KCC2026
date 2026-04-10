from __future__ import annotations


def load_manifest_records(manifest: object) -> tuple[list[dict], list[str]]:
    """Normalize supported manifest layouts into client records."""
    if isinstance(manifest, list):
        records = manifest
        channels: list[str] = []
    elif isinstance(manifest, dict) and isinstance(manifest.get("clients"), list):
        records = manifest["clients"]
        channels = [str(c) for c in manifest.get("channels", [])]
    elif isinstance(manifest, dict):
        records = [value for value in manifest.values() if isinstance(value, dict)]
        channels = []
    else:
        raise ValueError(f"Unsupported manifest type: {type(manifest)!r}")

    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Manifest records must be dictionaries.")
    return records, channels


def resolve_scale_stats(record: dict, channels: list[str]) -> tuple[float, float]:
    """Resolve inverse-transform stats from direct fields or channel_stats."""
    mean = record.get("mean_full", record.get("mean"))
    std = record.get("std_full", record.get("std"))
    if mean is not None and std is not None:
        return float(mean), float(std)

    channel_stats = record.get("channel_stats", {})
    if not isinstance(channel_stats, dict):
        return 0.0, 1.0

    ordered_channels = channels + [key for key in channel_stats if key not in channels]
    for channel in ordered_channels:
        stats = channel_stats.get(channel, {})
        if not isinstance(stats, dict):
            continue
        mean = stats.get("mean_full", stats.get("mean"))
        std = stats.get("std_full", stats.get("std"))
        if mean is not None and std is not None:
            return float(mean), float(std)
    return 0.0, 1.0
