"""Experimental source-pixel racket crops; production remains full frame.

Regions depend only on frozen camera configuration, so they do not introduce a
same-frame body-inference dependency. Each crop is letterboxed independently by
the existing detector; observations are restored before association/comparison.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass


POLICY_VERSION = 'configured_racket_regions_v1'


@dataclass(frozen=True)
class SourceRegion:
    name: str
    box: tuple

    @property
    def size(self):
        x1, y1, x2, y2 = self.box
        return x2 - x1, y2 - y1

    def extract(self, frame):
        x1, y1, x2, y2 = self.box
        return frame[y1:y2, x1:x2]


def configured_regions(frame_size, front_view, mirror_view, policy='full', margin=256):
    """Plan full/union/split regions, with an explicit source-pixel margin.

    Unlike the human views, racket crops keep their source orientation and are
    not first resized to 540x720. Mirror flipping is for human-view presentation.
    """
    width, height = frame_size
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('Invalid source dimensions')
    if policy not in ('full', 'union', 'split'):
        raise ValueError('Unknown racket region policy')
    if type(margin) is not int or margin < 0:
        raise ValueError('Invalid source-pixel margin')
    if policy == 'full':
        return [SourceRegion('full', (0, 0, width, height))]

    def expanded(name, normalized):
        if (not isinstance(normalized, (list, tuple)) or len(normalized) != 4
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) and 0 <= v <= 1 for v in normalized)):
            raise ValueError('Missing/invalid ' + name + ' region')
        x1, y1, x2, y2 = normalized
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Empty ' + name + ' region')
        return SourceRegion(name, (
            max(0, math.floor(x1 * width) - margin),
            max(0, math.floor(y1 * height) - margin),
            min(width, math.ceil(x2 * width) + margin),
            min(height, math.ceil(y2 * height) + margin),
        ))

    front = expanded('front', front_view.get('default_roi'))
    back = expanded('mirror', mirror_view.get('reflection_roi'))
    if policy == 'split':
        return [front, back]
    a, b = front.box, back.box
    return [SourceRegion('union', (min(a[0], b[0]), min(a[1], b[1]),
                                   max(a[2], b[2]), max(a[3], b[3])))]


def restore_source_observation(observation, region, source_size):
    """Restore crop-local boxes AND all seven points without mutating raw data."""
    result = copy.deepcopy(observation)
    x1, y1, x2, y2 = region.box
    width, height = source_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Region lies outside source image')
    metadata = {'policy_version': POLICY_VERSION, 'name': region.name,
                'box': list(region.box), 'source_size': list(source_size),
                'horizontal_flip': False}
    result['input_region'] = metadata
    for candidate in result.get('candidates', []):
        box = candidate['box']
        candidate['box'] = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
        for point in candidate['keypoints'].values():
            point['x'] += x1
            point['y'] += y1
        candidate['input_region'] = copy.deepcopy(metadata)
        # A region hint is not an anatomical association or a new observation.
        candidate['coordinate_space'] = 'original_source_pixels'
    result['coordinate_space'] = 'original_source_pixels'
    return result
