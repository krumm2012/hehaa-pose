"""Separate local calibration revisions, bound to actual input or camera address."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from ground_reference import normalize_calibration
from roi_stream_config import sanitize_stream_source


def source_binding(source, stream_id):
    path = Path(source)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return {'kind': 'video_sha256', 'stream_id': stream_id, 'source_id': digest.hexdigest()}
    if not str(source).lower().startswith(('rtsp://', 'rtsps://', 'http://', 'https://')):
        raise ValueError('地面标定输入视频不存在')
    # Keep query identity in the digest: ?channel=1 and ?channel=2 may be cameras.
    # Credentials in userinfo are excluded; neither address nor query is published.
    parsed = urlsplit(source)
    identity = sanitize_stream_source(source) + ('?' + parsed.query if parsed.query else '')
    return {'kind': 'camera_source', 'stream_id': stream_id,
            'source_id': hashlib.sha256(identity.encode()).hexdigest()}


class GroundCalibrationStore:
    def __init__(self, workspace):
        self.directory = Path(workspace) / 'data' / 'control_ground_calibrations'

    def _key(self, binding):
        return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()

    def load(self, binding):
        index = self.directory / (self._key(binding) + '.current.json')
        if not index.exists():
            return None
        revision = json.loads(index.read_text())['calibration_id']
        if (not isinstance(revision, str) or len(revision) != 64
                or any(c not in '0123456789abcdef' for c in revision)):
            raise ValueError('标定版本索引无效')
        document = normalize_calibration(json.loads((self.directory / (revision + '.json')).read_text()))
        if document['calibration_id'] != revision or document['binding'] != binding:
            raise ValueError('标定版本或来源不匹配')
        return document

    def save(self, document, binding):
        result = normalize_calibration(document)
        if result['binding'] != binding:
            raise ValueError('标定属于其他输入或机位，请在当前原图重新核对')
        # Validate before creating any files. Revisions are immutable; only index changes.
        self.directory.mkdir(parents=True, exist_ok=True)
        revision = self.directory / (result['calibration_id'] + '.json')
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        try:
            with revision.open('x') as handle:
                handle.write(text)
        except FileExistsError:
            if revision.read_text() != text:
                raise ValueError('已有标定版本内容冲突')
        index = self.directory / (self._key(binding) + '.current.json')
        with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({'calibration_id': result['calibration_id']}, handle)
        try:
            os.replace(temporary, index)
        finally:
            temporary.unlink(missing_ok=True)
        return result
