// Shared by independent and assisted boards, with separate storage identities.
const annotationIdentity = JSON.stringify([data.schema, data.source_sha256,
  data.prediction_sha256 || null, data.frames.map(f => [f.frame_id, f.width, f.height]), names]);
const draftStorageKey = 'tennis.joint-draft.v1:' + annotationIdentity;
function validateAnnotationDraft(candidate) {
  if (!candidate || candidate.schema !== data.schema || candidate.source_sha256 !== data.source_sha256 ||
      candidate.coordinate_space !== data.coordinate_space || candidate.frame_index_base !== data.frame_index_base ||
      candidate.prediction_sha256 !== data.prediction_sha256 ||
      JSON.stringify(candidate.frames) !== JSON.stringify(data.frames)) throw Error('来源、模式或帧范围不匹配');
  if (candidate.annotation_mode !== data.annotation_mode || candidate.independent_reference !== data.independent_reference)
    throw Error('独立标注与辅助复核不能互相导入');
  if (!candidate.labels || typeof candidate.labels !== 'object' || Array.isArray(candidate.labels)) throw Error('标签格式错误');
  if (candidate.confirmed === true && (!candidate.annotator_id || typeof candidate.annotator_id !== 'string')) throw Error('确认标签需要标注者编号');
  if (candidate.confirmed === true && data.model_suggestions && Object.values(candidate.labels).some(p => p.reviewed !== true)) throw Error('辅助建议仍有未复核项');
  if (data.require_complete_review && candidate.confirmed === true &&
      (Object.keys(candidate.labels).length !== data.frames.length * 2 * names.length ||
       Object.values(candidate.labels).some(p => p.reviewed !== true))) throw Error('完整审核需要逐项完成所有帧双视角关节');
  if (candidate.prediction_scale !== data.prediction_scale) throw Error('模型尺度不匹配');
  for (const [k, p] of Object.entries(candidate.labels)) {
    const [fid, view, joint] = k.split(':');
    const f = data.frames.find(f => String(f.frame_id) === fid);
    if (!f || !['front','back'].includes(view) || !names.includes(joint) || !p || typeof p.visible !== 'boolean') throw Error('未知帧、视角或关节');
    if (p.visible && (![p.x,p.y].every(v => typeof v === 'number' && Number.isFinite(v)) || p.x < 0 || p.y < 0 || p.x >= f.width || p.y >= f.height)) throw Error('点坐标越界');
    if (!p.visible && (p.x !== null || p.y !== null)) throw Error('不可辨认点必须保留空坐标');
    if (data.schema === 'tennis.independent-joint-labels.v1' && (candidate.model_suggestions || p.origin?.includes('model'))) throw Error('独立标签不能包含模型建议');
  }
  if (data.model_suggestions && JSON.stringify(candidate.model_suggestions) !== JSON.stringify(data.model_suggestions)) throw Error('原始模型建议已改变');
  return candidate;
}
function annotationGeometry(labels) {
  return JSON.stringify(Object.entries(labels).sort(([a],[b]) => a.localeCompare(b))
    .map(([k,p]) => [k,p.visible,p.x,p.y,p.reason || null]));
}
// Immutable observations are supplied by this source-bound page, not repeated
// in each saved revision. Keep the old key for safe in-place migration.
const annotationBaselineLabels = JSON.parse(JSON.stringify(data.labels));
function annotationStamp(snapshot) {
  return {annotator_id:snapshot.annotator_id || null, confirmed:snapshot.confirmed === true,
          draft_revision:snapshot.draft_revision || 0, saved_at:snapshot.saved_at || null};
}
function annotationDelta(before, after) {
  const changes={}, removed=[];
  for (const [key,point] of Object.entries(after))
    if (JSON.stringify(point)!==JSON.stringify(before[key])) changes[key]=point;
  for (const key of Object.keys(before)) if (!(key in after)) removed.push(key);
  return {changes,removed};
}
function applyAnnotationDelta(before, delta) {
  if (!delta || !delta.changes || typeof delta.changes!=='object' || Array.isArray(delta.changes) ||
      !Array.isArray(delta.removed)) throw Error('草稿修订格式无效');
  const labels=JSON.parse(JSON.stringify(before));
  for (const key of delta.removed) delete labels[key];
  for (const [key,point] of Object.entries(delta.changes)) labels[key]=JSON.parse(JSON.stringify(point));
  return labels;
}
function encodeAnnotationStorage(withHistory=true) {
  const current={...annotationStamp(data),...annotationDelta(annotationBaselineLabels,data.labels)};
  const historyReverse=[];
  // Deltas go backwards from the latest state; no history entry repeats all
  // 2000 labels, model suggestions, source metadata or review priorities.
  if (withHistory) for (let i=draftHistory.length-2;i>=0;i--)
    historyReverse.push({...annotationStamp(draftHistory[i]),
      ...annotationDelta(draftHistory[i+1].labels,draftHistory[i].labels)});
  return {schema:'tennis.annotation-storage.v2',identity:annotationIdentity,current,historyReverse};
}
function decodeAnnotationStorage(stored) {
  if (stored.schema!=='tennis.annotation-storage.v2') {
    validateAnnotationDraft(stored.current);
    return {current:stored.current,history:(stored.history||[]).slice(-20).filter(candidate=>{
      try { validateAnnotationDraft(candidate);return true; } catch { return false; }
    })};
  }
  if (stored.identity!==annotationIdentity) throw Error('草稿来源不匹配');
  const current={...data,...annotationStamp(stored.current),
                 labels:applyAnnotationDelta(annotationBaselineLabels,stored.current)};
  validateAnnotationDraft(current);
  const history=[JSON.parse(JSON.stringify(current))];let latest=current;
  for (const delta of (stored.historyReverse||[]).slice(0,19)) {
    const prior={...data,...annotationStamp(delta),labels:applyAnnotationDelta(latest.labels,delta)};
    validateAnnotationDraft(prior);history.push(prior);latest=prior;
  }
  return {current,history:history.reverse()};
}
let savedDraftSignature = '', lastAttemptSignature='', draftHistory = [], lastLabels = annotationGeometry(data.labels);
const originalAnnotationDraw = draw;
function persistAnnotationDraft(force=false) {
  data.annotator_id = $('annotator').value.trim() || null;
  data.confirmed = $('confirm').checked;
  const signature = JSON.stringify([data.labels, data.annotator_id, data.confirmed]);
  if (!force && (signature===savedDraftSignature || signature===lastAttemptSignature)) return;
  if (signature!==lastAttemptSignature) {
    data.draft_revision = (data.draft_revision || 0) + 1;
    data.saved_at = new Date().toISOString();
    draftHistory.push(JSON.parse(JSON.stringify(data)));
    draftHistory = draftHistory.slice(-20);
  }
  lastAttemptSignature=signature;
  try {
    let historyReduced=false;
    try { localStorage.setItem(draftStorageKey,JSON.stringify(encodeAnnotationStorage())); }
    catch (error) {
      if (error.name!=='QuotaExceededError') throw error;
      // Retry only our own current draft. Never delete another board's data.
      localStorage.setItem(draftStorageKey,JSON.stringify(encodeAnnotationStorage(false)));
      historyReduced=true;
    }
    savedDraftSignature = signature;
    $('draft-state').textContent = '本地草稿已保存 · 修订 ' + data.draft_revision +
      (historyReduced?' · 本地历史已精简，完整历史可导出':'');
  } catch (e) {
    $('draft-state').textContent = e.name==='QuotaExceededError'
      ? '本地空间不足，当前修改仍在本页。请导出草稿备份；不要关闭或刷新。'
      : '本地保存失败，当前修改仍在本页。请导出草稿备份。';
  }
}
draw = function() {
  const labels = annotationGeometry(data.labels);
  if (labels !== lastLabels) { $('confirm').checked = false; lastLabels = labels; }
  originalAnnotationDraw();
  $('state').textContent += '\n进度 ' + Object.keys(data.labels).length + '/' + (data.frames.length * 2 * names.length);
  persistAnnotationDraft();
};
function restoreAnnotationDraft(candidate) {
  validateAnnotationDraft(candidate);
  // Source metadata and model suggestions remain the board's immutable inputs.
  data.labels = JSON.parse(JSON.stringify(candidate.labels));
  data.draft_revision = candidate.draft_revision || 0;
  $('annotator').value = candidate.annotator_id || '';
  $('confirm').checked = candidate.confirmed === true;
  lastLabels = annotationGeometry(data.labels); savedDraftSignature = ''; lastAttemptSignature = '';
  draw();
}
$('draft-import').onchange = async e => {
  try {
    const candidate = validateAnnotationDraft(JSON.parse(await e.target.files[0].text()));
    if (!confirm('导入将替换当前填写内容，原草稿保留在最近20个修订中。继续？')) return;
    persistAnnotationDraft(); restoreAnnotationDraft(candidate);
  } catch (err) { $('draft-state').textContent = '导入被拒绝：' + err.message; }
  e.target.value = '';
};
$('draft-history').onclick = () => {
  const blob = new Blob([JSON.stringify({schema:'tennis.joint-draft-history.v1', identity:annotationIdentity, history:draftHistory},null,2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'joint_draft_history.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href),1000);
};
$('annotator').oninput = () => persistAnnotationDraft(); $('confirm').onchange = () => persistAnnotationDraft();
$('draft-retry').onclick = () => persistAnnotationDraft(true);
$('frame-prev').onclick = () => { $('frame').selectedIndex = Math.max(0, $('frame').selectedIndex-1); draw(); };
$('frame-next').onclick = () => { $('frame').selectedIndex = Math.min(data.frames.length-1, $('frame').selectedIndex+1); draw(); };
$('next-missing').onclick = () => {
  for (const f of data.frames) for (const view of ['front','back']) for (const joint of names) {
    if (!data.labels[[f.frame_id,view,joint].join(':')]) { $('frame').value=f.frame_id; $('view').value=view; $('joint').value=joint; draw(); return; }
  }
  $('draft-state').textContent = '所有项均已填写；仍需复核';
};
document.addEventListener('keydown', e => {
  if (['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); $('frame-prev').click(); }
  if (e.key === 'ArrowRight') { e.preventDefault(); $('frame-next').click(); }
});
try {
  const stored = JSON.parse(localStorage.getItem(draftStorageKey) || 'null');
  if (stored) { const decoded=decodeAnnotationStorage(stored); draftHistory=decoded.history; restoreAnnotationDraft(decoded.current); }
} catch (err) { $('draft-state').textContent = '草稿未恢复：' + err.message; }
