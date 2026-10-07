    const data = JSON.parse(document.getElementById('report-data').textContent);
    const video = document.getElementById('swing-video');
    const timeline = document.getElementById('event-timeline');
    const scrubber = document.getElementById('frame-scrubber');
    const timelineStatus = document.getElementById('timeline-status');
    const navigation = (data.timeline || {}).source_time_navigation || {};
    const sourceTimes = navigation.status === 'reported_media_time' ? navigation.frames || [] : [];
    const timeByFrame = new Map(sourceTimes);
    const maxEventFrame = Math.max(0, ...data.events.flatMap(event => [event.start_frame, event.contact_frame, event.peak_frame, event.end_frame].filter(Number.isSafeInteger)));
    const totalFrames = Math.max(1, Number((data.timeline || {}).total_frames || 0), maxEventFrame + 1,
                                sourceTimes.length ? sourceTimes[sourceTimes.length - 1][0] + 1 : 0);
    const modelEventIds = data.events.map(event => event.event_id);
    let activeEventId = data.events.length ? data.events[0].event_id : null;
    scrubber.max = String(totalFrames - 1);
    scrubber.disabled = !sourceTimes.length;

    function frameAtMediaTime(seconds) {
      if (!sourceTimes.length || seconds < sourceTimes[0][1]) return null;
      let lo = 0, hi = sourceTimes.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (sourceTimes[mid][1] <= seconds) lo = mid + 1; else hi = mid;
      }
      return sourceTimes[Math.max(0, lo - 1)][0];
    }

    function percentForFrame(frame) {
      return Math.max(0, Math.min(100, (Number(frame) / Math.max(1, totalFrames - 1)) * 100));
    }
    function seekFrame(frame, shouldPlay = false) {
      const safeFrame = typeof frame === 'string'
        ? sourceIdentityAttribute(frame, '定位源帧') : sourceFrameValue(frame, '定位源帧', true);
      if (safeFrame === null) {
        timelineStatus.textContent = '缺少源帧锚点，无法准确定位。';
        return;
      }
      const seconds = timeByFrame.get(safeFrame);
      if (seconds === undefined) {
        timelineStatus.textContent = `第 ${safeFrame} 帧源时间不可核验，无法准确定位。`;
        return;
      }
      // Stay just inside the requested presentation interval: some media APIs
      // truncate currentTime to microseconds, otherwise selecting the prior frame.
      const position = sourceTimes.findIndex(([fid]) => fid === safeFrame);
      const next = sourceTimes[position + 1]?.[1];
      const inset = next > seconds ? Math.min(0.000002, (next - seconds) / 4) : 0;
      video.currentTime = seconds + inset;
      scrubber.value = String(safeFrame);
      updatePlaybackState(safeFrame);
      if (shouldPlay) video.play();
    }
    function selectEvent(eventId, seek = true) {
      const identity = typeof eventId === 'string'
        ? sourceIdentityAttribute(eventId, '来源事件ID') : sourceFrameValue(eventId, '来源事件ID');
      const event = data.events.find(item => item.event_id === identity);
      if (!event) return;
      activeEventId = event.event_id;
      document.querySelectorAll('[data-event-id]').forEach(node => node.classList.toggle('is-active', node.dataset.eventId === String(activeEventId)));
      if (seek) seekFrame(event.start_frame);
    }
    function marker(label, frame, className, eventId) {
      if (frame === null || frame === undefined) return '';
      sourceFrameValue(frame, '时间轴源帧');
      const position = percentForFrame(frame);
      return `<button type="button" class="timeline-marker ${className}" data-event-id="${eventId}" data-frame="${frame}" style="left:calc(${position}% - 5px)" aria-label="${label}：第 ${frame} 帧">${label}</button>`;
    }
    function renderTimeline() {
      const ruler = [0, .25, .5, .75, 1].map(ratio => {
        const frame = Math.round((totalFrames - 1) * ratio);
        return `<span style="left:${ratio * 100}%">${frame}</span>`;
      }).join('');
      const lanes = data.events.map(event => {
        const isShadow = Boolean(event.is_shadow_swing || ((event.evidence || {}).contact_analysis || {}).is_shadow_swing || (((event.evidence || {}).classification_context || {}).contact_analysis || {}).is_shadow_swing);
        const start = event.start_frame;
        const end = event.end_frame;
        if (start === null || end === null) {
          return `<div class="timeline-lane"><span class="timeline-label">事件 ${event.event_id}</span><div class="timeline-track">缺少起止源帧锚点</div></div>`;
        }
        const left = percentForFrame(start);
        const width = Math.max(1.5, percentForFrame(end) - left);
        const btnClass = isShadow ? 'timeline-event timeline-event--shadow' : 'timeline-event';
        return `<div class="timeline-lane${isShadow ? ' timeline-lane--shadow' : ''}" data-is-shadow="${isShadow}"><span class="timeline-label">事件 ${event.event_id}</span><div class="timeline-track"><button type="button" class="${btnClass}" data-event-id="${event.event_id}" data-frame="${start}" style="left:${left}%;width:${width}%" title="${event.stroke_type} · ${start}-${end}${isShadow ? ' (空挥试拍)' : ''}">${event.stroke_type}${isShadow ? ' (试拍)' : ''}</button>${marker('触球候选', event.contact_frame, 'timeline-marker--contact', event.event_id)}${marker('动作峰值', event.peak_frame, 'timeline-marker--peak', event.event_id)}<i class="timeline-playhead" aria-hidden="true"></i></div></div>`;
      }).join('');
      timeline.innerHTML = `<div class="timeline-ruler"><span class="timeline-label">帧号</span><div class="timeline-track timeline-track--ruler">${ruler}</div></div>${lanes || '<p>未检测到挥拍事件。</p>'}`;
      timeline.querySelectorAll('[data-frame]').forEach(button => button.addEventListener('click', () => {
        selectEvent(button.dataset.eventId, false);
        seekFrame(button.dataset.frame, button.classList.contains('timeline-event'));
      }));
    }
    function updatePlaybackState(frame) {
      const currentFrame = sourceFrameValue(frame, '播放源帧', true);
      if (currentFrame === null) {
        timelineStatus.textContent = '当前播放位置无法对应已知源帧。';
        return;
      }
      scrubber.value = String(currentFrame);
      const currentEvent = data.events.find(event => event.start_frame !== null && event.end_frame !== null
        && currentFrame >= event.start_frame && currentFrame <= event.end_frame);
      if (currentEvent) selectEvent(currentEvent.event_id, false);
      const phase = ((data.timeline || {}).frame_trace || []).find(trace => Number(trace.frame) === currentFrame)?.phase || 'ready';
      const seconds = timeByFrame.get(currentFrame);
      timelineStatus.textContent = `Frame ${currentFrame} · ${seconds === undefined ? '源时间不可核验' : seconds.toFixed(3) + 's（媒体PTS）'} · ${phase}`;
      timeline.querySelectorAll('.timeline-playhead').forEach(playhead => playhead.style.left = `${percentForFrame(currentFrame)}%`);
    }
    document.getElementById('raw-summary').textContent = JSON.stringify({
      paths: data.paths,
      summary: data.summary,
      events: data.events.map(e => ({
        event_id: e.event_id,
        stroke_type: e.stroke_type,
        frames: [e.start_frame, e.contact_frame, e.peak_frame, e.end_frame],
        quality_flags: e.quality_flags,
        diagnosis_tags: e.diagnosis_tags,
        scores: {
          overall: e.overall_score,
          contact: e.contact_score,
          preparation: e.preparation_score,
          follow_through: e.follow_through_score
        }
      }))
    }, null, 2);
    const manualEvents = document.getElementById('manual-events');
    const timelineReviewComplete = document.getElementById('timeline-review-complete');
    const annotationReadiness = document.getElementById('annotation-readiness');
    let manualCounter = 0;

    /* __ANNOTATION_CONTRACT_SCRIPT__ */

    function annotationFromCard(card) {
      const sourceEventId = sourceIdentityAttribute(card.dataset.sourceEventId, '来源事件ID');
      const original = data.events.find(event => event.event_id === sourceEventId) || {};
      return {
        annotation_id: card.dataset.annotationId,
        source_event_id: sourceEventId,
        predicted_stroke_type: original.stroke_type || null,
        actual_stroke_type: card.querySelector('[data-field="actual_stroke_type"]').value,
        count_correct: card.querySelector('[data-field="count_correct"]').checked,
        valid_hit: card.querySelector('[data-field="valid_hit"]').checked,
        needs_review: card.querySelector('[data-field="needs_review"]').checked,
        issue_tags: Array.from(card.querySelectorAll('[data-tag]:checked')).map(input => input.dataset.tag),
        note: card.querySelector('[data-field="note"]').value.trim(),
        frames: {
          start: integerField(card, 'start_frame'),
          contact: integerField(card, 'contact_frame'),
          peak: sourceIdentityAttribute(card.dataset.peakFrame, '动作峰值帧'),
          end: integerField(card, 'end_frame')
        },
        quality_flags: original.quality_flags || {}
      };
    }
    function collectAnnotations() {
      return {
        schema_version: 'swing_manual_annotations_v2',
        timeline_review_complete: timelineReviewComplete.checked,
        source: {...data.paths, reference_method: 'model_assisted_review', model_predictions_visible: true},
        summary: data.summary,
        events: Array.from(document.querySelectorAll('[data-annotation-card]')).map(annotationFromCard)
      };
    }
    function updateAnnotationReadiness(payload) {
      try { validateAnnotationPayload(payload, true); }
      catch (error) { showAnnotationError(error); return; }
      const pending = payload.events.filter(event => event.needs_review).length;
      if (pending > 0) {
        annotationReadiness.dataset.state = 'blocked';
        annotationReadiness.textContent = `还有 ${pending} 条“需要复核”，评估指标将保持 provisional。`;
      } else if (!payload.timeline_review_complete) {
        annotationReadiness.dataset.state = 'blocked';
        annotationReadiness.textContent = '请完整检查整段视频后勾选确认项。';
      } else {
        annotationReadiness.dataset.state = 'ready';
        annotationReadiness.textContent = '已满足复核对照条件，可以下载标注 JSON；独立准确性仍需验证。';
      }
    }
    function refreshAnnotations() {
      try {
        const payload = validateAnnotationPayload(collectAnnotations(), false, modelEventIds);
        updateAnnotationReadiness(payload);
        document.getElementById('annotation-json').textContent = JSON.stringify(payload, null, 2);
        return payload;
      } catch (error) { showAnnotationError(error); return null; }
    }
    function setField(card, field, value) {
      const input = card.querySelector(`[data-field="${field}"]`);
      if (!input) return;
      if (['start_frame','contact_frame','end_frame'].includes(field)) {
        input.value = value == null ? '' : String(sourceFrameValue(value, '人工帧号'));
        input.setCustomValidity('');
        return;
      }
      if (value === undefined || value === null) return;
      if (input.type === 'checkbox') {
        input.checked = Boolean(value);
      } else {
        input.value = String(value);
      }
    }
    function bindAnnotationInputs(root) {
      root.querySelectorAll('input, select, textarea').forEach(element => {
        element.addEventListener('change', refreshAnnotations);
      });
      root.querySelectorAll('textarea, input[type="number"]').forEach(element => {
        element.addEventListener('input', refreshAnnotations);
      });
    }
    function applyAnnotationToCard(card, imported) {
      const identity = annotationIdentity(imported);
      const frames = validateAnnotationFrames(imported);
      card.dataset.annotationId = identity.annotation_id;
      card.dataset.peakFrame = frames.peak == null ? '' : String(frames.peak);
      setField(card, 'actual_stroke_type', imported.actual_stroke_type);
      setField(card, 'count_correct', imported.count_correct);
      setField(card, 'valid_hit', imported.valid_hit);
      setField(card, 'needs_review', imported.needs_review);
      setField(card, 'note', imported.note || '');
      setField(card, 'start_frame', frames.start);
      setField(card, 'contact_frame', frames.contact);
      setField(card, 'end_frame', frames.end);
      const tags = new Set(imported.issue_tags || []);
      card.querySelectorAll('[data-tag]').forEach(input => input.checked = tags.has(input.dataset.tag));
    }
    function addMissedEvent(imported = null) {
      if (imported) { annotationIdentity(imported); validateAnnotationFrames(imported); }
      manualCounter += 1;
      while (document.querySelector(`[data-annotation-id="manual-${manualCounter}"]`)) manualCounter += 1;
      const card = document.createElement('article');
      const importedId = imported && (imported.annotation_id ?? imported.event_id);
      card.className = 'manual-event-card';
      card.dataset.annotationCard = '';
      card.dataset.annotationId = importedId != null ? String(importedId) : `manual-${manualCounter}`;
      const sourceId = imported ? annotationIdentity(imported).source_event_id : null;
      card.dataset.sourceEventId = sourceId == null ? '' : String(sourceId);
      card.innerHTML = `
        <div class="manual-event-head">
          <h3>人工补充挥拍</h3>
          <button class="remove-manual-event" type="button">删除</button>
        </div>
        <div class="annotation-box">
          <div class="annotation-frame-grid">
            <label>人工开始帧<input type="number" min="0" step="1" data-field="start_frame"></label>
            <label>人工触球帧<input type="number" min="0" step="1" data-field="contact_frame"></label>
            <label>人工结束帧<input type="number" min="0" step="1" data-field="end_frame"></label>
          </div>
          <label>人工类型
            <select class="annotation-stroke" data-field="actual_stroke_type">
              <option value="Forehand">Forehand</option>
              <option value="Backhand">Backhand</option>
              <option value="Two-Handed Backhand">Two-Handed Backhand</option>
              <option value="Serve">Serve</option>
              <option value="Volley">Volley</option>
              <option value="Unclear" selected>Unclear</option>
            </select>
          </label>
          <label><input type="checkbox" data-field="valid_hit" checked> 有效击球</label>
          <label><input type="checkbox" data-field="count_correct"> 计数正确</label>
          <label><input type="checkbox" data-field="needs_review"> 需要复核</label>
          <div class="annotation-tags">
            <label><input type="checkbox" data-tag="wrong_type"> 类型错误</label>
            <label><input type="checkbox" data-tag="contact_timing"> 触球帧偏差</label>
            <label><input type="checkbox" data-tag="event_boundary"> 边界偏差</label>
            <label><input type="checkbox" data-tag="missed_event"> 漏检事件</label>
          </div>
          <label>备注<textarea rows="2" data-field="note" placeholder="漏检原因、动作特点等"></textarea></label>
        </div>`;
      card.querySelector('.remove-manual-event').addEventListener('click', () => {
        card.remove();
        refreshAnnotations();
      });
      bindAnnotationInputs(card);
      if (imported) applyAnnotationToCard(card, imported);
      manualEvents.appendChild(card);
      refreshAnnotations();
      return card;
    }
    function applyImportedAnnotations(payload) {
      const plan = prepareAnnotationImport(payload);
      payload = plan.payload;
      timelineReviewComplete.checked = Boolean(payload.timeline_review_complete);
      manualEvents.replaceChildren();
      let applied = 0;
      for (const {annotation: imported, card} of plan.assignments) {
        if (card) {
          applyAnnotationToCard(card, imported);
        } else {
          addMissedEvent(imported);
        }
        applied += 1;
      }
      refreshAnnotations();
      document.getElementById('import-status').textContent = `已导入 ${applied} 条标注`;
      return applied;
    }
    function importAnnotations(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          applyImportedAnnotations(JSON.parse(reader.result));
        } catch (err) {
          document.getElementById('import-status').textContent = `导入失败: ${err.message}`;
        }
      };
      reader.readAsText(file);
    }
    function downloadAnnotations() {
      const payload = refreshAnnotations();
      if (!payload) return;
      try { validateAnnotationPayload(payload, true); }
      catch (error) { showAnnotationError(error); return; }
      const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'swing_manual_annotations_v2.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }
    document.querySelectorAll('.event-jump').forEach(button => button.addEventListener('click', () => selectEvent(button.dataset.eventId)));
    scrubber.addEventListener('input', event => seekFrame(event.target.value));
    document.getElementById('play-event').addEventListener('click', () => {
      const event = data.events.find(item => item.event_id === activeEventId);
      if (event) seekFrame(event.start_frame, true);
      else video.play();
    });
    let filterOnlyValidInReport = true;
    function setReportFilter(onlyValid) {
      filterOnlyValidInReport = onlyValid;
      const aside = document.querySelector('aside.events');
      const timelineEl = document.getElementById('event-timeline');
      const btnValid = document.getElementById('report-filter-valid');
      const btnAll = document.getElementById('report-filter-all');
      const badge = document.getElementById('report-events-count');

      if (btnValid && btnAll) {
        if (onlyValid) {
          btnValid.classList.add('active');
          btnAll.classList.remove('active');
          if (aside) aside.classList.add('filter-only-valid');
          if (timelineEl) timelineEl.classList.add('filter-only-valid');
        } else {
          btnAll.classList.add('active');
          btnValid.classList.remove('active');
          if (aside) aside.classList.remove('filter-only-valid');
          if (timelineEl) timelineEl.classList.remove('filter-only-valid');
        }
      }

      const total = (data.events || []).length;
      const validEvents = (data.events || []).filter(e => !e.is_shadow_swing && !((e.evidence || {}).contact_analysis || {}).is_shadow_swing && !(((e.evidence || {}).classification_context || {}).contact_analysis || {}).is_shadow_swing);
      const valid = validEvents.length;
      const shadow = total - valid;

      if (badge) {
        if (onlyValid) {
          badge.textContent = `${valid} 次有效击球${shadow > 0 ? ` (已过滤 ${shadow} 次空挥试拍)` : ''}`;
        } else {
          badge.textContent = `${total} 次记录 (含试拍)`;
        }
      }

      if (onlyValid && activeEventId != null) {
        const curr = (data.events || []).find(e => e.event_id === activeEventId);
        const isCurrShadow = curr && (curr.is_shadow_swing || ((curr.evidence || {}).contact_analysis || {}).is_shadow_swing || (((curr.evidence || {}).classification_context || {}).contact_analysis || {}).is_shadow_swing);
        if (isCurrShadow && validEvents.length > 0) {
          selectEvent(validEvents[0].event_id, false);
        }
      }
    }

    video.addEventListener('timeupdate', () => {
      const frame = frameAtMediaTime(video.currentTime);
      if (frame !== null) updatePlaybackState(frame);
    });
    bindAnnotationInputs(document);
    renderTimeline();
    updatePlaybackState(0);
    refreshAnnotations();
    setReportFilter(true);
