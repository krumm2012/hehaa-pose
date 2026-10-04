"""Source frame identities shared by manual review and generated report editors."""

POLICY_VERSION = 'manual_annotation_identity_v1_no_coercion'
MAX_SAFE_FRAME_ID = 2**53 - 1


def require_manual_frame_id(value):
    if type(value) is not int or not 0 <= value <= MAX_SAFE_FRAME_ID:
        raise ValueError('帧号必须是非负安全整数，不能转换小数、布尔值或字符串')
    return value


def annotation_contract_script():
    """Inline the same validation in live and standalone reports; no network asset."""
    return r'''
    function sourceFrameValue(value, label, allowMissing = false) {
      if (value === null || value === undefined) {
        if (allowMissing) return null;
        throw new Error(`${label}缺失，请填写非负整数源帧号`);
      }
      if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
        throw new Error(`${label}必须是非负安全整数，不自动转换或取整`);
      }
      return value;
    }
    function integerField(card, field) {
      const input = card.querySelector(`[data-field="${field}"]`);
      if (!input) return null;
      input.setCustomValidity('');
      const raw = input.value.trim();
      if (raw === '' && !input.validity?.badInput) return null;
      try {
        if (!/^\d+$/.test(raw)) throw new Error('人工帧号只能填写非负整数，不自动取整');
        return sourceFrameValue(Number(raw), '人工帧号');
      } catch (error) {
        input.setCustomValidity(error.message);
        throw error;
      }
    }
    function annotationFrames(annotation) {
      if (!annotation || typeof annotation !== 'object' || Array.isArray(annotation)) {
        throw new Error('人工标注必须是对象');
      }
      const frames = annotation.frames ?? {};
      if (typeof frames !== 'object' || Array.isArray(frames)) throw new Error('标注frames必须是对象');
      const resolved = {};
      for (const key of ['start', 'contact', 'end']) {
        resolved[key] = Object.prototype.hasOwnProperty.call(frames, key)
          ? frames[key] : annotation[`${key}_frame`];
      }
      return resolved;
    }
    function validateAnnotationFrames(annotation, complete = false) {
      const frames = annotationFrames(annotation);
      const labels = {start:'开始帧', contact:'触球帧', end:'结束帧'};
      for (const key of ['start', 'contact', 'end']) {
        frames[key] = sourceFrameValue(frames[key], labels[key], !complete);
      }
      if ((frames.start !== null && frames.contact !== null && frames.start > frames.contact)
          || (frames.contact !== null && frames.end !== null && frames.contact > frames.end)
          || (frames.start !== null && frames.end !== null && frames.start > frames.end)) {
        throw new Error('帧号顺序必须满足开始帧 ≤ 触球帧 ≤ 结束帧');
      }
      return frames;
    }
    function validateAnnotationPayload(payload, complete = false) {
      if (!payload || payload.schema_version !== 'swing_manual_annotations_v2'
          || !Array.isArray(payload.events) || (complete && payload.events.length === 0)) {
        throw new Error('请选择包含事件的swing_manual_annotations_v2标注');
      }
      for (const event of payload.events) validateAnnotationFrames(event, complete);
      return payload;
    }
    function showAnnotationError(error) {
      const message = `标注帧号未通过校验：${error.message || String(error)}`;
      const readiness = document.getElementById('annotation-readiness');
      if (readiness) { readiness.dataset.state = 'blocked'; readiness.textContent = message; }
      const status = document.getElementById('annotation-status');
      if (status) status.textContent = message;
      const preview = document.getElementById('annotation-json');
      if (preview) preview.textContent = message;
    }
'''
