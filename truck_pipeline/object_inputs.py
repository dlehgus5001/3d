"""Validate object-only RGB without changing the original pose/depth inputs."""
from pathlib import Path
from PIL import Image
from truck_pipeline.data import IMAGE_SUFFIXES


def resolve_object_images(frames, objects_dir):
    root = Path(objects_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Object RGB directory does not exist: {root}')
    candidates = {}
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.setdefault(path.stem, []).append(path)
    result, issues = {}, []
    for frame in frames:
        name = frame.get('filename', frame.get('img_name'))
        matches = candidates.get(Path(name).stem, [])
        if len(matches) != 1:
            issues.append(f'{name}: expected one matching object RGB, found {len(matches)} in {root}')
            continue
        path = matches[0]
        try:
            with Image.open(path) as image:
                image.load()
                expected_format = 'PNG' if path.suffix.lower() == '.png' else 'JPEG'
                if image.format != expected_format or image.mode not in ('RGB', 'RGBA'):
                    issues.append(f'{path.name}: expected a color RGB/RGBA {expected_format} image, not a binary mask')
                if image.size != (frame['width'], frame['height']):
                    issues.append(f'{path.name}: resolution {image.size} differs from original {(frame["width"], frame["height"])}; no crop/resize allowed')
        except (OSError, ValueError) as exc:
            issues.append(f'{path.name}: cannot decode object RGB: {exc}')
        result[name] = path
    if issues:
        raise ValueError('Object RGB validation failed:\n' + '\n'.join(issues))
    return result
