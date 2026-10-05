"""Source frame identities shared by manual review and generated report editors."""

POLICY_VERSION = 'manual_annotation_identity_v2_event_links'
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
    function sourceIdentityAttribute(text, label) {
      if (text === undefined || text === '') return null;
      if (typeof text !== 'string' || !/^(0|[1-9]\d*)$/.test(text)) {
        throw new Error(`${label}必须保留非负整数身份`);
      }
      // Only decode canonical DOM text written from a validated JSON integer.
      return sourceFrameValue(Number(text), label);
    }
    function annotationIdentity(annotation, index = 0) {
      if (!annotation || typeof annotation !== 'object' || Array.isArray(annotation)) {
        throw new Error('人工标注必须是对象');
      }
      for (const key of ['event_id', 'source_event_id']) {
        if (annotation[key] !== null && annotation[key] !== undefined) {
          sourceFrameValue(annotation[key], `事件身份 ${key}`);
        }
      }
      const source = Object.prototype.hasOwnProperty.call(annotation, 'source_event_id')
        ? annotation.source_event_id : annotation.event_id;
      const sourceId = sourceFrameValue(source, '来源事件ID', true);
      const id = annotation.annotation_id ?? (annotation.event_id == null
        ? `annotation-${index + 1}` : String(annotation.event_id));
      if (typeof id !== 'string' || !id.trim()) throw new Error('标注身份ID必须是非空字符串');
      return {annotation_id:id, source_event_id:sourceId};
    }
    function annotationFrames(annotation) {
      if (!annotation || typeof annotation !== 'object' || Array.isArray(annotation)) {
        throw new Error('人工标注必须是对象');
      }
      const frames = annotation.frames ?? {};
      if (typeof frames !== 'object' || Array.isArray(frames)) throw new Error('标注frames必须是对象');
      const resolved = {};
      for (const key of ['start', 'contact', 'end', 'peak']) {
        resolved[key] = Object.prototype.hasOwnProperty.call(frames, key)
          ? frames[key] : (Object.prototype.hasOwnProperty.call(annotation, `${key}_frame`)
            ? annotation[`${key}_frame`] : frames[`${key}_frame`]);
        for (const [mapping, name] of [[annotation, `${key}_frame`], [frames, key], [frames, `${key}_frame`]]) {
          if (Object.prototype.hasOwnProperty.call(mapping, name)) {
            sourceFrameValue(mapping[name], `源帧 ${name}`, true);
          }
        }
      }
      return resolved;
    }
    function validateAnnotationFrames(annotation, complete = false) {
      const frames = annotationFrames(annotation);
      const labels = {start:'开始帧', contact:'触球帧', end:'结束帧'};
      for (const key of ['start', 'contact', 'end']) {
        frames[key] = sourceFrameValue(frames[key], labels[key], !complete);
      }
      frames.peak = sourceFrameValue(frames.peak, '动作峰值帧', true);
      if ((frames.start !== null && frames.contact !== null && frames.start > frames.contact)
          || (frames.contact !== null && frames.end !== null && frames.contact > frames.end)
          || (frames.start !== null && frames.end !== null && frames.start > frames.end)) {
        throw new Error('帧号顺序必须满足开始帧 ≤ 触球帧 ≤ 结束帧');
      }
      return frames;
    }
    function validateAnnotationPayload(payload, complete = false, knownSourceIds = null) {
      if (!payload || payload.schema_version !== 'swing_manual_annotations_v2'
          || !Array.isArray(payload.events) || (complete && payload.events.length === 0)) {
        throw new Error('请选择包含事件的swing_manual_annotations_v2标注');
      }
      if (payload.timeline_review_complete !== undefined && typeof payload.timeline_review_complete !== 'boolean') {
        throw new Error('整段复核确认必须是布尔值');
      }
      if (payload.source != null && (typeof payload.source !== 'object' || Array.isArray(payload.source))) {
        throw new Error('标注来源必须是对象');
      }
      const identities = new Set(), sourceIds = new Set();
      const events = payload.events.map((event, index) => {
        validateAnnotationFrames(event, complete);
        const identity = annotationIdentity(event, index);
        if (identities.has(identity.annotation_id)) throw new Error(`标注身份重复: ${identity.annotation_id}`);
        identities.add(identity.annotation_id);
        const sourceId = identity.source_event_id;
        if (sourceId !== null) {
          if (sourceIds.has(sourceId)) throw new Error(`来源事件ID重复: ${sourceId}`);
          sourceIds.add(sourceId);
          if (knownSourceIds !== null && !knownSourceIds.includes(sourceId)) {
            throw new Error(`来源事件 #${sourceId} 在当前报告中不存在`);
          }
        }
        for (const key of ['valid_hit', 'needs_review', 'count_correct']) {
          if (event[key] !== undefined && typeof event[key] !== 'boolean') throw new Error(`${key}必须是布尔值`);
        }
        if (event.issue_tags != null && (!Array.isArray(event.issue_tags) || event.issue_tags.some(tag => typeof tag !== 'string'))) {
          throw new Error('问题标签必须是字符串数组');
        }
        return {...event, ...identity};
      });
      return {...payload, events};
    }
    function prepareAnnotationImport(payload) {
      const validated = validateAnnotationPayload(payload, false, modelEventIds);
      const modelCards = [...document.querySelectorAll('.event-card[data-annotation-card]')];
      const cardsBySource = new Map();
      for (const card of modelCards) {
        const sourceId = sourceIdentityAttribute(card.dataset.sourceEventId, '来源事件ID');
        if (sourceId === null || cardsBySource.has(sourceId)) throw new Error('页面来源事件身份缺失或重复');
        cardsBySource.set(sourceId, card);
      }
      const assignments = validated.events.map(annotation => {
        const card = annotation.source_event_id === null ? null : cardsBySource.get(annotation.source_event_id);
        if (annotation.source_event_id !== null && !card) throw new Error('来源事件在当前页面不存在');
        return {annotation, card};
      });
      const targeted = new Set(assignments.map(item => item.card).filter(Boolean));
      const remainingIds = new Set();
      for (const card of modelCards.filter(card => !targeted.has(card))) {
        const id = annotationIdentity({annotation_id:card.dataset.annotationId}).annotation_id;
        if (remainingIds.has(id)) throw new Error(`页面标注身份重复: ${id}`);
        remainingIds.add(id);
      }
      for (const {annotation} of assignments) {
        if (remainingIds.has(annotation.annotation_id)) throw new Error(`标注身份与当前卡片重复: ${annotation.annotation_id}`);
        remainingIds.add(annotation.annotation_id);
      }
      return {payload:validated, assignments};
    }
    function showAnnotationError(error) {
      const message = `标注身份或帧号未通过校验：${error.message || String(error)}`;
      const readiness = document.getElementById('annotation-readiness');
      if (readiness) { readiness.dataset.state = 'blocked'; readiness.textContent = message; }
      const status = document.getElementById('annotation-status');
      if (status) status.textContent = message;
      const preview = document.getElementById('annotation-json');
      if (preview) preview.textContent = message;
    }
'''
