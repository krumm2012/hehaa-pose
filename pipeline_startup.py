"""Warm Core ML kernels without advancing detection or motion state."""
import time
from PIL import Image


def warm_coreml_wrappers(wrappers):
    timings = []
    seen = set()
    for wrapper in wrappers:
        model = getattr(wrapper, 'model', None)
        if model is None or not hasattr(model, 'get_spec') or id(model) in seen:
            continue
        seen.add(id(model))
        inputs = {}
        for desc in model.get_spec().description.input:
            if desc.type.WhichOneof('Type') == 'imageType':
                image = desc.type.imageType
                inputs[desc.name] = Image.new('RGB', (image.width, image.height))
            elif desc.name == 'iouThreshold':
                inputs[desc.name] = wrapper.coreml_iou_threshold
            elif desc.name == 'confidenceThreshold':
                inputs[desc.name] = wrapper.coreml_confidence_threshold
            elif not desc.type.isOptional:
                raise ValueError('Unsupported warmup input: ' + desc.name)
        started = time.perf_counter()
        model.predict(inputs)
        timings.append((time.perf_counter() - started) * 1000)
    return timings
