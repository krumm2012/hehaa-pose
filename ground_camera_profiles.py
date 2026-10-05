"""Explicit camera-level reuse of ground geometry; no inferred camera identity."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

from ground_reference import normalize_calibration

POLICY_VERSION = 'ground_camera_profile_v1_explicit_mapping_exact_size'


class GroundCameraProfiles:
    def __init__(self, workspace):
        self.directory = Path(workspace) / 'data' / 'control_ground_profiles'

    def _key(self, binding):
        return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()

    def save(self, document, camera_binding):
        calibration = normalize_calibration(document)
        if (camera_binding.get('kind') != 'camera_source'
                or calibration['binding']['stream_id'] != camera_binding.get('stream_id')):
            raise ValueError('导入标定的机位必须与目标机位相同')
        # Validate the target binding with the same strict contract before writing.
        normalize_calibration({**calibration, 'binding': camera_binding})
        result = {'schema': 'tennis.ground-camera-profile.v1', 'policy_version': POLICY_VERSION,
                  'application_scope': 'camera_profile', 'camera_binding': dict(camera_binding),
                  'source_calibration': calibration, 'accuracy_validated': False}
        canonical = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
        result['profile_id'] = hashlib.sha256(canonical.encode()).hexdigest()
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        self.directory.mkdir(parents=True, exist_ok=True)
        revision = self.directory / (result['profile_id']+'.json')
        try:
            with revision.open('x') as handle: handle.write(text)
        except FileExistsError:
            if revision.read_text() != text: raise ValueError('共享标定修订冲突')
        index = self.directory / (self._key(camera_binding)+'.current.json')
        with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({'profile_id': result['profile_id']}, handle)
        try: os.replace(temporary, index)
        finally: temporary.unlink(missing_ok=True)
        return result

    def load(self, camera_binding):
        index = self.directory / (self._key(camera_binding)+'.current.json')
        if not index.exists(): return None
        identity = json.loads(index.read_text())['profile_id']
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('共享标定版本索引无效')
        result = json.loads((self.directory / (identity+'.json')).read_text())
        unsigned = {k:v for k,v in result.items() if k != 'profile_id'}
        digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if (digest != identity or result.get('camera_binding') != camera_binding
                or result.get('schema') != 'tennis.ground-camera-profile.v1'
                or result.get('policy_version') != POLICY_VERSION):
            raise ValueError('共享标定内容或机位身份不匹配')
        # Recompute and validate geometry without changing the immutable revision.
        calibration = normalize_calibration(result['source_calibration'])
        if calibration['binding']['stream_id'] != camera_binding['stream_id']:
            raise ValueError('共享标定机位不匹配')
        result['source_calibration'] = calibration
        return result

    def resolve(self, input_binding, camera_binding):
        if input_binding['stream_id'] != camera_binding['stream_id']:
            return None
        if input_binding['kind'] == 'camera_source' and input_binding != camera_binding:
            return None
        profile = self.load(camera_binding)
        if profile is None: return None
        source = profile['source_calibration']
        # Only the explicitly mapped input receives a derived copy, not a rewritten source.
        calibration = normalize_calibration({**copy.deepcopy(source), 'binding': input_binding})
        application = {'policy_version': POLICY_VERSION, 'scope': 'camera_profile',
                       'profile_id': profile['profile_id'], 'camera_binding': camera_binding,
                       'source_calibration_id': source['calibration_id'],
                       'source_calibration_binding': source['binding'],
                       'input_binding': input_binding,
                       'geometry_reuse': 'operator_declared_same_camera_exact_image_size',
                       'accuracy_validated': False}
        return {'calibration': calibration, 'application': application}
