// Exercise the same preview functions used by SSE/polling and the ROI editor.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const core = fs.readFileSync('static/js/control_panel_core.js', 'utf8');
const roi = fs.readFileSync('static/js/roi_editor.js', 'utf8');
const code = core.slice(core.indexOf('    function applyPreviewBlob('), core.indexOf('    async function startAnalysis('))
  + roi.slice(roi.indexOf('    async function toggleRoiEditor('), roi.indexOf('    function startRepointRoi('));
function deferred() {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
}
function harness() {
  const elements = new Map();
  const $ = id => {
    if (!elements.has(id)) elements.set(id, { hidden: true, style: {}, classList: { add() {}, remove() {} } });
    return elements.get(id);
  };
  const image = $('preview-image');
  Object.assign(image, { src: 'analysis-composite', decode: async () => {} });
  $('settings-form').reportValidity = () => true;
  let liveFetches = 0;
  const c = vm.createContext({ $, state: {}, roiEditorState: { active: false, loading: false, courtPoints: [], mirrorPoints: [] },
    URL: { createObjectURL: b => b.kind, revokeObjectURL() {} }, Date,
    fetch: async () => { liveFetches++; return { ok: true, blob: async () => ({ kind: 'analysis-composite' }) }; },
    api: async () => ({ blob: async () => ({ kind: 'source-frame' }) }),
    formPayload: () => ({}), showToast() {}, syncCanvasGeometry() {}, selectedStream: () => ({}),
    initRoiEditorFromStream() {}, parseSidebarPoints: () => [], sourceToPreview: p => p, setEditTarget() {},
  });
  vm.runInContext(code, c);
  return { c, $, image, run: text => vm.runInContext(text, c), liveFetches: () => liveFetches };
}
(async () => {
  // A running session's repeated status events must leave the source-space editor alone.
  let h = harness();
  h.c.roiEditorState.active = true;
  h.image.src = 'source-frame';
  await h.run('refreshLivePreview()');
  assert.equal(h.image.src, 'source-frame', 'live polling replaced the ROI source frame with a composite');
  assert.equal(h.liveFetches(), 0);

  // Check ownership after BOTH awaits, not only at request dispatch.
  for (const phase of ['fetch', 'blob']) {
    h = harness();
    const wait = deferred();
    h.c.fetch = phase === 'fetch'
      ? () => wait.promise
      : async () => ({ ok: true, blob: () => wait.promise });
    const pending = h.run('refreshLivePreview()');
    await Promise.resolve();
    h.c.roiEditorState.active = true;
    h.image.src = 'source-frame';
    wait.resolve(phase === 'fetch' ? { ok: true, blob: async () => ({ kind: 'analysis-composite' }) } : { kind: 'analysis-composite' });
    await pending;
    assert.equal(h.image.src, 'source-frame', `late live ${phase} response overwrote the editor`);
    assert.equal(h.run('refreshLivePreview.pending'), false);
  }

  // A live request from before entry must stay stale even if it arrives after exit.
  h = harness();
  const live = deferred();
  h.c.fetch = () => live.promise;
  const pendingLive = h.run('refreshLivePreview()');
  await h.run('refreshPreview(false, true)');
  live.resolve({ ok: true, blob: async () => ({ kind: 'stale-composite' }) });
  await pendingLive;
  assert.equal(h.image.src, 'source-frame', 'an older live response survived a source preview transition');

  // An older manual annotated preview may not overwrite the newer clean preview either.
  h = harness();
  const old = deferred();
  let request = 0;
  h.c.api = () => ++request === 1 ? old.promise : Promise.resolve({ blob: async () => ({ kind: 'source-frame' }) });
  const oldPreview = h.run('refreshPreview(false, false)');
  await h.run('refreshPreview(false, true)');
  old.resolve({ blob: async () => ({ kind: 'stale-annotated-frame' }) });
  await oldPreview;
  assert.equal(h.image.src, 'source-frame');

  // Enter via the real toggle; neither fetch nor image decoding may expose a canvas on the composite.
  h = harness();
  const clean = deferred(), decoded = deferred();
  h.c.api = () => clean.promise;
  h.image.decode = () => decoded.promise;
  const entry = h.run('toggleRoiEditor(true)');
  assert.equal(h.c.roiEditorState.loading, true);
  await h.run('refreshLivePreview()');
  assert.equal(h.liveFetches(), 0, 'editor entry did not reserve preview ownership');
  clean.resolve({ blob: async () => ({ kind: 'source-frame' }) });
  await new Promise(setImmediate);
  assert.equal(h.c.roiEditorState.active, false, 'editing started before the clean image decoded');
  assert.equal(h.$('roi-canvas').hidden, true);
  decoded.resolve();
  await entry;
  assert.equal(h.c.roiEditorState.active, true);
  assert.equal(h.c.roiEditorState.loading, false);
  assert.equal(h.$('roi-canvas').hidden, false);
  await h.run('refreshLivePreview()');
  assert.equal(h.image.src, 'source-frame');
  await h.run('toggleRoiEditor(false)');
  assert.equal(h.$('roi-canvas').hidden, true);
  await h.run('refreshLivePreview()');
  assert.equal(h.image.src, 'analysis-composite', 'normal live preview did not resume after editing');

  // Failed source capture / JPEG decoding must not enable drawing on the last analysis frame.
  for (const phase of ['capture', 'decode']) {
    h = harness();
    if (phase === 'capture') h.c.api = async () => { throw Error('capture unavailable'); };
    else h.image.decode = async () => { throw Error('invalid JPEG'); };
    await h.run('toggleRoiEditor(true)');
    assert.equal(h.c.roiEditorState.active, false, `${phase} failure enabled the editor`);
    assert.equal(h.c.roiEditorState.loading, false);
    assert.equal(h.$('roi-canvas').hidden, true);
  }
  console.log('PASS: ROI source ownership, in-flight fetch/blob races, stale manual preview, decode readiness, failure handling, and live resume.');
})().catch(e => { console.error(e); process.exitCode = 1; });
