    const coachFeed = document.getElementById('live-coach-feed');
    const reportSummary = document.getElementById('report-summary');
    const refreshStatus = document.getElementById('refresh-status');
    const refreshStart = document.getElementById('refresh-start');
    const refreshStop = document.getElementById('refresh-stop');
    const annotationWorkspace = document.getElementById('annotation-workspace');
    const manualEvents = document.getElementById('manual-events');
    const timelineReviewComplete = document.getElementById('timeline-review-complete');
    const annotationStatus = document.getElementById('annotation-status');
    const annotationReadiness = document.getElementById('annotation-readiness');
    const manualReviewWorkspace = document.getElementById('manual-review-workflow');
    const manualReviewFile = document.getElementById('manual-review-file');
    const manualReviewMessage = document.getElementById('manual-review-message');
    const manualReviewState = document.getElementById('review-workflow-state');
    const manualReviewMetrics = document.getElementById('manual-review-metrics');
    const coachComparisons = document.getElementById('coach-comparisons');
    const evaluateImportedReview = document.getElementById('evaluate-imported-review');
    const evaluateCurrentReview = document.getElementById('evaluate-current-review');
    const annotationStorageKey = `tennis.swing.annotations.v2:${location.pathname}:${eventJsonUrl}`;
    const refreshStorageKey = `tennis.swing.auto-refresh.v1:${location.pathname}`;
    let autoRefreshEnabled = true;
    let coachFeedPending = false;
    let manualCounter = 0;
    let lastAnnotationInteraction = 0;
    let importedReviewPayload = null;
    let savedReviewFingerprint = null;
    let lastPublishedReviewState = null;
    let restoredBrowserDraft = false;

    try {
      autoRefreshEnabled = sessionStorage.getItem(refreshStorageKey) !== 'false';
    } catch (_error) {
      // Keep auto-refresh enabled when storage is unavailable (for example, restricted file:// pages).
    }

    function updateRefreshControls() {
      refreshStatus.dataset.state = autoRefreshEnabled ? 'running' : 'paused';
      refreshStatus.textContent = autoRefreshEnabled ? '自动刷新中' : '页面刷新已暂停';
      refreshStart.disabled = autoRefreshEnabled;
      refreshStop.disabled = !autoRefreshEnabled;
      refreshStart.setAttribute('aria-pressed', String(autoRefreshEnabled));
      refreshStop.setAttribute('aria-pressed', String(!autoRefreshEnabled));
    }

    function setAutoRefreshEnabled(enabled) {
      autoRefreshEnabled = Boolean(enabled);
      try {
        sessionStorage.setItem(refreshStorageKey, String(autoRefreshEnabled));
      } catch (_error) {
        // The controls still work for the current document when storage is unavailable.
      }
      updateRefreshControls();
      if (autoRefreshEnabled) {
        refreshCoachFeed();
        refreshPreview();
      }
    }


    /* __ANNOTATION_CONTRACT_SCRIPT__ */


    function annotationFromCard(card) {
      const sourceEventId = sourceIdentityAttribute(card.dataset.sourceEventId, '来源事件ID');
      const peakNumber = sourceIdentityAttribute(card.dataset.peakFrame, '动作峰值帧');
      return {
        ...humanReviewFromCard(card),
        annotation_id: card.dataset.annotationId,
        source_event_id: sourceEventId,
        predicted_stroke_type: card.dataset.predictedStrokeType || null,
        actual_stroke_type: card.querySelector('[data-field="actual_stroke_type"]').value,
        count_correct: card.querySelector('[data-field="count_correct"]').checked,
        valid_hit: card.querySelector('[data-field="valid_hit"]').checked,
        needs_review: card.querySelector('[data-field="needs_review"]').checked,
        issue_tags: [...card.querySelectorAll('[data-tag]:checked')].map(input => input.dataset.tag),
        note: card.querySelector('[data-field="note"]').value.trim(),
        frames: {
          start: integerField(card, 'start_frame'),
          contact: integerField(card, 'contact_frame'),
          peak: peakNumber,
          end: integerField(card, 'end_frame')
        }
      };
    }

    function collectAnnotations() {
      return {
        schema_version: 'swing_manual_annotations_v2',
        review_policy_version: 'human_coach_review_v3',
        timeline_review_complete: timelineReviewComplete.checked,
        source: { event_json: eventJsonUrl, reference_method: 'model_assisted_review', model_predictions_visible: true },
        events: [...document.querySelectorAll('[data-annotation-card]')].map(annotationFromCard)
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

    function saveAnnotations() {
      lastAnnotationInteraction = Date.now();
      try {
        const payload = validateAnnotationPayload(collectAnnotations(), false, modelEventIds);
        localStorage.setItem(annotationStorageKey, JSON.stringify(payload));
        updateAnnotationReadiness(payload);
        if (typeof markReviewDirty === 'function') markReviewDirty(payload);
        annotationStatus.textContent = `已在浏览器保存 ${payload.events.length} 条草稿`;
        return payload;
      } catch (error) { showAnnotationError(error); return null; }
    }

    function setAnnotationField(card, field, value) {
      const input = card.querySelector(`[data-field="${field}"]`);
      if (!input) return;
      if (['start_frame','contact_frame','end_frame'].includes(field)) {
        input.value = value == null ? '' : String(sourceFrameValue(value, '人工帧号'));
        input.setCustomValidity('');
        return;
      }
      if (value === undefined || value === null) return;
      if (input.type === 'checkbox') input.checked = Boolean(value);
      else input.value = String(value);
    }

    function applyAnnotation(card, imported) {
      const identity = annotationIdentity(imported);
      const frames = validateAnnotationFrames(imported);
      validateHumanReview(imported);
      applyHumanReview(card, imported);
      card.dataset.annotationId = identity.annotation_id;
      card.dataset.peakFrame = frames.peak == null ? '' : String(frames.peak);
      setAnnotationField(card, 'actual_stroke_type', imported.actual_stroke_type);
      setAnnotationField(card, 'count_correct', imported.count_correct);
      setAnnotationField(card, 'valid_hit', imported.valid_hit);
      setAnnotationField(card, 'needs_review', imported.needs_review);
      setAnnotationField(card, 'note', imported.note || '');
      setAnnotationField(card, 'start_frame', frames.start);
      setAnnotationField(card, 'contact_frame', frames.contact);
      setAnnotationField(card, 'end_frame', frames.end);
      const tags = new Set(imported.issue_tags || []);
      card.querySelectorAll('[data-tag]').forEach(input => input.checked = tags.has(input.dataset.tag));
    }

    function addMissedEvent(imported = null, persist = true) {
      if (imported) { annotationIdentity(imported); validateAnnotationFrames(imported); }
      manualCounter += 1;
      while (document.querySelector(`[data-annotation-id="manual-${manualCounter}"]`)) manualCounter += 1;
      const card = document.createElement('article');
      card.className = 'manual-event-card';
      card.dataset.annotationCard = '';
      card.dataset.annotationId = String(imported?.annotation_id ?? imported?.event_id ?? `manual-${manualCounter}`);
      const sourceId = imported ? annotationIdentity(imported).source_event_id : null;
      card.dataset.sourceEventId = sourceId == null ? '' : String(sourceId);
      card.dataset.predictedStrokeType = '';
      card.dataset.peakFrame = String(imported?.frames?.peak ?? '');
      card.innerHTML = `
        <div class="manual-event-heading"><h2>人工补充挥拍</h2><button type="button">删除</button></div>
        <div class="annotation-box">
          <div class="annotation-frames">
            <label>人工开始帧<input type="number" min="0" step="1" data-field="start_frame"></label>
            <label>触球候选 / 人工帧<input type="number" min="0" step="1" data-field="contact_frame"></label>
            <label>人工结束帧<input type="number" min="0" step="1" data-field="end_frame"></label>
          </div>
          <label>人工类型<select data-field="actual_stroke_type"><option>Forehand</option><option>Backhand</option><option>Two-Handed Backhand</option><option>Serve</option><option>Volley</option><option selected>Unclear</option></select></label>
          <div class="annotation-checks">
            <label><input type="checkbox" data-field="valid_hit" checked> 有效击球</label>
            <label><input type="checkbox" data-field="count_correct"> 计数正确</label>
            <label><input type="checkbox" data-field="needs_review"> 需要复核</label>
            <label><input type="checkbox" data-tag="missed_event" checked> 漏检事件</label>
            <label><input type="checkbox" data-tag="contact_timing"> 触球帧偏差</label>
            <label><input type="checkbox" data-tag="event_boundary"> 边界偏差</label>
          </div>
          <label>备注<textarea rows="2" data-field="note"></textarea></label>
          ${humanReviewFields()}
        </div>`;
      card.querySelector('button').addEventListener('click', () => { card.remove(); saveAnnotations(); });
      if (imported) applyAnnotation(card, imported);
      manualEvents.append(card);
      if (persist) saveAnnotations();
      return card;
    }

    function restoreAnnotations() {
      let payload;
      try { payload = JSON.parse(localStorage.getItem(annotationStorageKey) || 'null'); }
      catch (_error) { payload = null; }
      if (!payload || !Array.isArray(payload.events)) return;
      let plan;
      try { plan = prepareAnnotationImport(payload); }
      catch (error) { showAnnotationError(error); return; }
      payload = plan.payload;
      restoredBrowserDraft = true;
      timelineReviewComplete.checked = Boolean(payload.timeline_review_complete);
      manualEvents.replaceChildren();
      for (const {annotation: imported, card} of plan.assignments) {
        if (card) applyAnnotation(card, imported);
        else addMissedEvent(imported, false);
      }
      annotationStatus.textContent = `已恢复 ${payload.events.length} 条浏览器标注`;
    }

    function downloadAnnotations() {
      const payload = saveAnnotations();
      if (!payload) return;
      try { validateAnnotationPayload(payload, true); }
      catch (error) { showAnnotationError(error); return; }
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'swing_manual_annotations_v2.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }

    function handleAnnotationEdit(event) {
      if (event.target?.closest('[data-annotation-card]') || annotationWorkspace.contains(event.target)) {
        importedReviewPayload = null;
        evaluateImportedReview.disabled = true;
        saveAnnotations();
      }
    }
    document.addEventListener('input', handleAnnotationEdit);
    document.addEventListener('change', handleAnnotationEdit);
    document.getElementById('add-missed-event').addEventListener('click', () => addMissedEvent());
    document.getElementById('download-annotations').addEventListener('click', downloadAnnotations);
    restoreAnnotations();
    updateAnnotationReadiness(collectAnnotations());


    /* __REFERENCE_METRIC_SCRIPT__ */


    function setReviewStage(stage) {
      const order = ['import', 'validate', 'evaluate', 'coach'];
      const activeIndex = Math.max(0, order.indexOf(stage));
      order.forEach((name, index) => {
        document.getElementById(`review-stage-${name}`).dataset.active = String(index <= activeIndex);
      });
    }

    function setReviewStatus(label, state, message) {
      manualReviewState.textContent = label;
      manualReviewState.dataset.state = state;
      manualReviewMessage.textContent = message || '';
      manualReviewMessage.dataset.state = state === 'error' ? 'error' : '';
    }

    function coachColumn(title, advices, manual = false) {
      const column = document.createElement('div');
      column.className = `coach-column${manual ? ' manual' : ''}`;
      const heading = document.createElement('h3');
      heading.textContent = title;
      column.append(heading);
      const list = document.createElement('ol');
      for (const advice of advices || []) {
        const row = document.createElement('li');
        const message = document.createElement('span');
        message.textContent = advice.message || advice.code || '无建议';
        const confidence = document.createElement('small');
        const score = advice.confidence;
        const value = typeof score === 'number' && Number.isFinite(score) && score >= 0 && score <= 1 ? `${Math.round(score*100)}%` : '未提供';
        confidence.textContent = advice.category === 'review' ? '复核提示' : ` 证据参考 ${value} · 未校准`;
        row.append(message, confidence);
        list.append(row);
      }
      if (!list.children.length) {
        const row = document.createElement('li');
        row.textContent = '等待正式人工复核';
        list.append(row);
      }
      column.append(list);
      return column;
    }

    function reviewFingerprint(payload) {
      if (!payload) return null;
      const rows = payload.events.map(row => ({
        id:row.annotation_id, source:row.source_event_id ?? null, frames:validateAnnotationFrames(row),
        type:row.actual_stroke_type, valid:row.valid_hit ?? true, count:row.count_correct ?? true,
        pending:row.needs_review ?? false, tags:[...(row.issue_tags || [])].sort(), note:row.note || '',
        contact:row.contact_review || 'unreviewed',
        coach:['reviewer','verdict','observed_fact','suggestion','evidence'].map(k => row.coach_review?.[k] || (k === 'verdict' ? 'unreviewed' : '')),
        ratings:['preparation','positioning','contact','coordination','recovery'].map(k => row.practice_review?.ratings?.[k] ?? null),
        confirmed:row.practice_review?.confirmed === true,
      })).sort((a,b) => String(a.id).localeCompare(String(b.id)));
      return JSON.stringify({complete:payload.timeline_review_complete === true, rows});
    }

    function markReviewDirty(payload) {
      if (savedReviewFingerprint !== null && reviewFingerprint(payload) !== savedReviewFingerprint) {
        setReviewStatus('草稿已修改，待重新评估', 'dirty', '下方对照属于上次提交；本次修改已保存到浏览器，点击“保存并评估当前审核”更新。');
        manualReviewMetrics.hidden = true;
        coachComparisons.style.opacity = '.55';
      } else if (manualReviewState.dataset.state === 'dirty' && lastPublishedReviewState) {
        renderReviewState(lastPublishedReviewState);
      }
    }

    function renderReviewState(state) {
      lastPublishedReviewState = state;
      const status = state?.status || 'waiting_for_annotations';
      const validation = state?.validation || null;
      const summary = state?.evaluation?.summary || null;
      if (status === 'finalized') {
        setReviewStatus('事件对照已完成', 'finalized', '已按审核边界重算规则建议；触球确认和人工教练意见分别记录，事件对照完成不代表教练评分完成。');
        setReviewStage('coach');
      } else if (status === 'needs_review') {
        const pending = validation?.pending_count ?? 0;
        setReviewStatus('仍需人工复核', 'needs_review', `已生成 provisional 评估；还有 ${pending} 条需要复核，暂不生成正式人工 Coach。`);
        setReviewStage('evaluate');
      } else {
        setReviewStatus('等待标注', 'waiting', '请选择下载的 swing_manual_annotations_v2.json，或直接评估页面当前标注。');
        setReviewStage('import');
      }

      manualReviewMetrics.replaceChildren();
      if (summary) {
        const metrics = reviewMetricRows(state.evaluation);
        for (const [label, value] of metrics) {
          const item = document.createElement('div');
          const caption = document.createElement('span');
          caption.textContent = label;
          const number = document.createElement('strong');
          number.textContent = value;
          item.append(caption, number);
          manualReviewMetrics.append(item);
        }
        const note = document.createElement('p');
        note.textContent = reviewReferenceNote(state.evaluation);
        const human = state.human_review_summary;
        if (human) {
          note.textContent += ` 事件可比 ${summary.true_positive_count}/${summary.predicted_event_count}；类型可比 ${summary.stroke_type_comparable}/${summary.true_positive_count}；触球已确认 ${human.contact_confirmed_count}/${human.event_count}；教练已审核 ${human.coach_reviewed_count}/${human.event_count}；人工总分已确认 ${human.coach_score_confirmed_count}/${human.event_count}。`;
        }
        manualReviewMetrics.append(note);
        manualReviewMetrics.hidden = false;
      } else {
        manualReviewMetrics.hidden = true;
      }

      coachComparisons.replaceChildren();
      for (const comparison of state?.comparisons || []) {
        const card = document.createElement('article');
        card.className = 'coach-comparison';
        const head = document.createElement('div');
        head.className = 'coach-comparison-head';
        const title = document.createElement('strong');
        title.textContent = `Swing #${comparison.source_event_id ?? comparison.event_id} · ${comparison.stroke_type || 'Unknown'}${comparison.excluded_event ? ' · 已排除非有效事件' : ''}`;
        const changes = document.createElement('span');
        changes.textContent = comparison.changed_fields?.length
          ? `变化：${comparison.changed_fields.join('、')}`
          : '人工边界与 Coach 未变化';
        head.append(title, changes);
        const columns = document.createElement('div');
        columns.className = 'coach-columns';
        columns.append(
          coachColumn('实时 Coach（原始）', comparison.original_coach, false),
          coachColumn('人工边界重算（系统规则）', comparison.manual_coach, true),
        );
        const review = comparison.human_coach_review || {};
        const humanColumn = document.createElement('div');
        humanColumn.className = 'coach-column manual';
        const humanTitle = document.createElement('h3'); humanTitle.textContent = '人工教练意见';
        const humanText = document.createElement('p');
        const verdictLabels = {unreviewed:'未审核',supported:'证据支持',rejected:'不采纳',insufficient:'证据不足'};
        const score = comparison.human_practice_score;
        humanText.textContent = `${verdictLabels[review.verdict] || '未审核'} · 审核人：${review.reviewer || '未填写'}\n事实：${review.observed_fact || '未填写'}\n证据：${review.evidence || '未填写'}\n建议：${review.suggestion || '未填写'}\n人工五维评分：${score?.status === 'coach_confirmed' ? score.score + '/100' : '草稿 / 未评'}`;
        humanText.style.whiteSpace = 'pre-wrap';
        humanColumn.append(humanTitle, humanText); columns.append(humanColumn);
        card.append(head, columns);
        coachComparisons.append(card);
      }
      coachComparisons.style.opacity = '1';
      if (state.review_policy_version !== 'human_coach_review_v3' && state.status !== 'waiting_for_annotations') {
        setReviewStatus('历史审核，待按 V3 重算', 'needs_review', '旧结果保留。点击“保存并评估当前审核”更新事件口径；不会自动确认触球或教练评分。');
        manualReviewMetrics.hidden = true;
      }
      if (state.annotations) {
        savedReviewFingerprint = reviewFingerprint(state.annotations);
        markReviewDirty(collectAnnotations());
      }
    }

    function applyImportedReviewToEditor(payload) {
      if (!payload || !Array.isArray(payload.events)) return;
      let plan;
      try { plan = prepareAnnotationImport(payload); }
      catch (error) { showAnnotationError(error); return; }
      payload = plan.payload;
      timelineReviewComplete.checked = Boolean(payload.timeline_review_complete);
      manualEvents.replaceChildren();
      for (const {annotation: imported, card} of plan.assignments) {
        if (card) applyAnnotation(card, imported);
        else addMissedEvent(imported, false);
      }
      saveAnnotations();
      return payload;
    }

    function manualReviewReportHeaders() {
      const headers = {};
      const artifactPrefix = '/artifacts/';
      if (location.pathname.startsWith(artifactPrefix)) {
        const reportPath = decodeURIComponent(location.pathname.slice(artifactPrefix.length));
        if (reportPath) headers['X-Manual-Review-Report'] = reportPath;
      }
      return headers;
    }

    async function manualReviewRequestHeaders() {
      const headers = manualReviewReportHeaders();
      headers['Content-Type'] = 'application/json';
      try {
        const response = await fetch('/api/config', { cache: 'no-store' });
        if (!response.ok) return headers;
        const config = await response.json();
        if (config && config.token) headers['X-Control-Token'] = String(config.token);
      } catch (_error) {
        // The standalone manual-review server does not require a control token.
      }
      return headers;
    }

    async function submitManualReview(payload) {
      try { payload = validateAnnotationPayload(payload, true, modelEventIds); }
      catch (error) {
        showAnnotationError(error);
        setReviewStatus('标注校验失败', 'error', error.message);
        return;
      }
      if (location.protocol === 'file:') {
        setReviewStatus('需要本地服务', 'error', '请运行 manual_review_workflow.py 后从 http://127.0.0.1 打开本页。');
        return;
      }
      setReviewStatus('处理中', 'waiting', '正在校验标注并重新聚合逐帧证据…');
      setReviewStage('validate');
      evaluateImportedReview.disabled = true;
      evaluateCurrentReview.disabled = true;
      try {
        const response = await fetch('/api/manual-review/evaluate', {
          method: 'POST',
          headers: await manualReviewRequestHeaders(),
          body: JSON.stringify(payload),
        });
        const state = await response.json();
        if (!response.ok) throw new Error(state.message || state.error || '人工校准失败');
        renderReviewState(state);
      } catch (error) {
        setReviewStatus('校准失败', 'error', error.message || String(error));
        setReviewStage('validate');
      } finally {
        evaluateImportedReview.disabled = !importedReviewPayload;
        evaluateCurrentReview.disabled = false;
      }
    }

    manualReviewFile.addEventListener('change', async event => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      try {
        const payload = JSON.parse(await file.text());
        if (payload.schema_version !== 'swing_manual_annotations_v2' || !Array.isArray(payload.events)) {
          throw new Error('请选择 swing_manual_annotations_v2 JSON');
        }
        const applied = applyImportedReviewToEditor(payload);
        if (!applied) throw new Error('标注身份或帧号未通过校验，原草稿保留');
        importedReviewPayload = applied;
        const pending = payload.events.filter(item => item.needs_review).length;
        evaluateImportedReview.disabled = false;
        setReviewStatus(
          pending ? '导入完成，仍需复核' : '导入完成',
          pending ? 'needs_review' : 'waiting',
          `已读取 ${payload.events.length} 条标注；${pending ? `其中 ${pending} 条仍需复核。` : '可以生成复核对照。'}`,
        );
        setReviewStage('import');
      } catch (error) {
        importedReviewPayload = null;
        evaluateImportedReview.disabled = true;
        setReviewStatus('导入失败', 'error', error.message || String(error));
      }
    });
    evaluateImportedReview.addEventListener('click', () => submitManualReview(importedReviewPayload));
    evaluateCurrentReview.addEventListener('click', () => {
      try { submitManualReview(collectAnnotations()); }
      catch (error) { showAnnotationError(error); setReviewStatus('标注校验失败', 'error', error.message); }
    });

    async function loadManualReviewState() {
      if (location.protocol === 'file:') {
        renderReviewState({ status: 'waiting_for_annotations' });
        manualReviewMessage.textContent = '当前是 file:// 页面；运行 manual_review_workflow.py 后可生成评估和人工 Coach。';
        return;
      }
      try {
        const response = await fetch('/api/manual-review/state', {
          cache: 'no-store',
          headers: manualReviewReportHeaders(),
        });
        const state = await response.json();
        if (!response.ok) throw new Error(state.message || state.error || '人工校准服务不可用');
        if (!restoredBrowserDraft && state.annotations) applyImportedReviewToEditor(state.annotations);
        renderReviewState(state);
      } catch (error) {
        setReviewStatus('服务不可用', 'error', error.message || String(error));
      }
    }
    loadManualReviewState();

    function renderCoachFeed(documentPayload) {
      const events = Array.isArray(documentPayload.events) ? documentPayload.events : [];
      const summary = documentPayload.summary || {};
      reportSummary.textContent = `${events.length} events · frame ${summary.latest_frame ?? -1}`;
      coachFeed.replaceChildren();
      const title = document.createElement('h2');
      title.textContent = '实时 Coach';
      coachFeed.append(title);
      if (!events.length) {
        const waiting = document.createElement('p');
        waiting.textContent = '等待第一条建议…';
        coachFeed.append(waiting);
        return;
      }
      const latest = events[events.length - 1];
      const meta = document.createElement('p');
      meta.textContent = `Swing #${latest.event_id} · ${latest.stroke_type || 'Unknown'}`;
      coachFeed.append(meta);
      const advices = Array.isArray(latest.coach_advices) && latest.coach_advices.length
        ? latest.coach_advices
        : (latest.coach_advice ? [latest.coach_advice] : []);
      const list = document.createElement('ol');
      for (const advice of advices.slice(0, 3)) {
        if (!advice || !advice.message) continue;
        const row = document.createElement('li');
        const message = document.createElement('strong');
        message.textContent = advice.message;
        row.append(message);
        list.append(row);
      }
      if (list.children.length) coachFeed.append(list);
    }

    async function refreshCoachFeed() {
      if (!autoRefreshEnabled || coachFeedPending) return;
      coachFeedPending = true;
      try {
        const response = await fetch(`${eventJsonUrl}?t=${Date.now()}`, { cache: 'no-store' });
        if (autoRefreshEnabled && response.ok) renderCoachFeed(await response.json());
      } catch (_error) {
        // file:// reports cannot fetch siblings; the slower HTML refresh below remains available.
      } finally {
        coachFeedPending = false;
      }
    }

    const preview = document.getElementById('roi-preview');
    const previewSource = preview ? preview.getAttribute('src').split('?')[0] : '';
    function refreshPreview() {
      if (autoRefreshEnabled && preview) preview.src = previewSource + '?t=' + Date.now();
    }

    refreshStart.addEventListener('click', () => setAutoRefreshEnabled(true));
    refreshStop.addEventListener('click', () => setAutoRefreshEnabled(false));
    updateRefreshControls();
    if (autoRefreshEnabled) {
      refreshCoachFeed();
      refreshPreview();
    }
    setInterval(refreshCoachFeed, 200);
    setInterval(refreshPreview, 1000);
    setInterval(() => {
      if (!autoRefreshEnabled) return;
      const playing = [...document.querySelectorAll('video')].some(video => !video.paused && !video.ended);
      const editingAnnotations = Date.now() - lastAnnotationInteraction < 15000;
      const annotationFocused = annotationWorkspace.contains(document.activeElement);
      const workflowFocused = manualReviewWorkspace.contains(document.activeElement);
      if (!playing && !editingAnnotations && !annotationFocused && !workflowFocused) location.reload();
    }, 3000);
