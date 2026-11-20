    let compareToken = "";
    let pageViewMode = "forehand"; // "forehand" | "backhand" | "ab"
    let globalProfileData = null;
    let globalGridPreset = "conflict_4"; // "conflict_4" | "timeline_4" | "best_worst_4" | "custom_2x2"
    let globalTrendMetricKey = "shoulder_turn_change";
    let globalSlots = []; // Max 4 entries: { label, side, sessionId, eventId }
    const gridFocusX = { 0: 76, 1: 76, 2: 76, 3: 76 };

    let latestCompareSessions = [];
    let latestPersonalAttribution = null;
    let selectedCompareSessionA = "";
    let selectedCompareSessionB = "";
    let compareCropMode = "front_focus"; // "front_focus" | "back_focus" | "raw"
    const compareFocusX = { A: 76, B: 76 };
    const slotDetectedDualView = { A: null, B: null };

    let batchAllSessions = [];
    const batchCheckedIds = new Set();
    let dedupEnabled = true;
    let dedupModalData = null;
    const dedupCheckedIds = new Set();
    let globalTrendViewMode = "auto"; // "auto" | "per_swing" | "per_session"
    let trajectorySummaryBySide = { forehand: null, backhand: null };
    let activeTrajectorySwingList = [];
    let activeTrajectoryIndex = -1;
    let selectedRadarSwingKey = ""; // "" = global overview, or "${session_id}#${event_id}"
    let selectedGlobalAiChoice = "deepseek::deepseek-chat";

    const CORE_COMPARE_METRICS = [
      { key: "shoulder_turn_change", label: "转肩角度变化", unit: "°", refMax: 90, higherBetter: true },
      { key: "hip_shoulder_separation", label: "肩髋分离角", unit: "°", refMax: 45, higherBetter: true },
      { key: "preparation_knee_flexion", label: "准备期屈膝角", unit: "°", refMax: 175, higherBetter: false },
      { key: "arm_extension", label: "持拍侧肘伸展", unit: "°", refMax: 170, higherBetter: true },
      { key: "contact_lateral_distance", label: "触球离体距离", unit: "bw", refMax: 1.5, higherBetter: true },
      { key: "weight_transfer", label: "躯干重心横移", unit: "bw", refMax: 0.8, higherBetter: true },
      { key: "balance_drift", label: "回位期重心漂移", unit: "bw", refMax: 0.6, higherBetter: false },
      { key: "racket_max_speed", label: "拍框峰值速度", unit: "px/s", refMax: 1800, higherBetter: true },
    ];

    const TREND_METRIC_OPTIONS = [
      { key: "shoulder_turn_change", label: "转肩角度变化", unit: "°" },
      { key: "hip_shoulder_separation", label: "肩髋分离角", unit: "°" },
      { key: "arm_extension", label: "持拍侧肘伸展", unit: "°" },
      { key: "preparation_knee_flexion", label: "准备期屈膝角", unit: "°" },
      { key: "contact_lateral_distance", label: "触球离体距离", unit: "bw" },
      { key: "weight_transfer", label: "躯干重心横移", unit: "bw" },
      { key: "balance_drift", label: "回位期重心漂移", unit: "bw" },
      { key: "racket_max_speed", label: "拍框峰值速度", unit: "px/s" },
      { key: "consistency_index", label: "动作定型度", unit: "%" },
    ];

    function escapeHtml(value) {
      const div = document.createElement("div");
      div.textContent = value == null ? "" : String(value);
      return div.innerHTML;
    }

    function showToast(message, error = false) {
      const toast = document.getElementById("toast");
      if (!toast) return;
      toast.textContent = message;
      toast.className = "toast visible" + (error ? " error" : "");
      clearTimeout(showToast.timer);
      showToast.timer = setTimeout(() => { toast.className = "toast"; }, 3600);
    }

    async function api(path, options = {}) {
      const headers = { ...(options.headers || {}) };
      if (options.method === "POST") {
        headers["Content-Type"] = headers["Content-Type"] || "application/json";
        if (compareToken) headers["X-Control-Token"] = compareToken;
      }
      const response = await fetch(path, { ...options, headers, cache: "no-store" });
      if (!response.ok) {
        let message = `${response.status} ${response.statusText}`;
        try {
          const payload = await response.json();
          message = payload.error || message;
        } catch (_) {}
        throw new Error(message);
      }
      return response;
    }

    function activeStrokeSide() {
      return pageViewMode === "backhand" ? "backhand" : "forehand";
    }

    function syncUrlParams() {
      if (typeof window === "undefined" || !window.history?.replaceState) return;
      const playerSel = document.getElementById("compare-player-select");
      const strokeSel = document.getElementById("compare-stroke-select");
      const daysSel = document.getElementById("compare-days-select");
      const params = new URLSearchParams(window.location.search);
      params.set("mode", pageViewMode);
      if (playerSel?.value) params.set("player_id", playerSel.value);
      else params.delete("player_id");
      if (daysSel?.value && daysSel.value !== "all") params.set("days", daysSel.value);
      else params.delete("days");
      if (!dedupEnabled) params.set("dedup", "0");
      else params.delete("dedup");
      if (pageViewMode === "ab" && strokeSel?.value && strokeSel.value !== "all") {
        params.set("stroke_type", strokeSel.value);
      } else {
        params.delete("stroke_type");
      }
      const qs = params.toString();
      const nextUrl = window.location.pathname + (qs ? `?${qs}` : "");
      window.history.replaceState(null, "", nextUrl);
    }

    function updateModeTabCounts() {
      const fhBtn = document.getElementById("mode-forehand-btn");
      const bhBtn = document.getElementById("mode-backhand-btn");
      const fhCnt = globalProfileData?.forehand?.total_swings;
      const bhCnt = globalProfileData?.backhand?.total_swings;
      if (fhBtn) fhBtn.textContent = fhCnt != null ? `🎾 正手全局 (${fhCnt}拍)` : "🎾 正手全局分析";
      if (bhBtn) bhBtn.textContent = bhCnt != null ? `🛡️ 反手全局 (${bhCnt}拍)` : "🛡️ 反手全局分析";
    }

    function setPageViewMode(mode) {
      if (mode === "global") mode = "forehand";
      pageViewMode = ["forehand", "backhand", "ab"].includes(mode) ? mode : "forehand";
      const fhBtn = document.getElementById("mode-forehand-btn");
      const bhBtn = document.getElementById("mode-backhand-btn");
      const abBtn = document.getElementById("mode-ab-btn");
      if (fhBtn) fhBtn.classList.toggle("active", pageViewMode === "forehand");
      if (bhBtn) bhBtn.classList.toggle("active", pageViewMode === "backhand");
      if (abBtn) abBtn.classList.toggle("active", pageViewMode === "ab");

      const daysSel = document.getElementById("compare-days-select");
      const strokeSel = document.getElementById("compare-stroke-select");
      const dedupLabel = document.getElementById("compare-dedup-toggle")?.closest("label");
      if (daysSel) daysSel.hidden = pageViewMode === "ab";
      if (dedupLabel) dedupLabel.hidden = pageViewMode === "ab";
      if (strokeSel) strokeSel.hidden = pageViewMode !== "ab";

      syncUrlParams();
      if (pageViewMode !== "ab" && globalProfileData) {
        const sideKey = activeStrokeSide();
        selectedRadarSwingKey = "";
        globalSlots = buildDefaultGlobalSlots(sideKey, globalGridPreset === "custom_2x2" ? "conflict_4" : globalGridPreset);
        renderSingleStrokeDashboard(sideKey);
        if (!trajectorySummaryBySide[sideKey]) {
          loadTrajectorySummary(sideKey);
        }
      } else {
        loadActiveViewData();
      }
    }

    async function loadComparePlayers(preferredPlayerId = "") {
      const sel = document.getElementById("compare-player-select");
      if (!sel) return;
      const prev = preferredPlayerId || sel.value;
      const response = await api("/api/players");
      const data = await response.json();
      const players = data.players || [];
      [sel, document.getElementById("batch-target-player")].forEach(target => {
        if (!target) return;
        const targetPrev = target === sel ? prev : (target.value || prev);
        target.innerHTML = "";
        players.forEach(p => {
          const opt = document.createElement("option");
          opt.value = p.player_id;
          opt.textContent = `${p.display_name}${p.session_count ? ` (${p.session_count})` : ""}`;
          target.appendChild(opt);
        });
        if (targetPrev && Array.from(target.options).some(o => o.value === targetPrev)) {
          target.value = targetPrev;
        }
      });
    }

    async function createNewComparePlayer() {
      const name = window.prompt("请输入新学员姓名（例如：张三）：");
      if (!name || !name.trim()) return;
      try {
        const res = await api("/api/players", {
          method: "POST",
          body: JSON.stringify({ display_name: name.trim() }),
        });
        const created = await res.json();
        await loadComparePlayers(created.player_id || "");
        syncUrlParams();
        await loadActiveViewData();
        showToast(`已创建学员：${created.display_name || name.trim()}`);
      } catch (err) {
        showToast(`创建学员失败：${err.message}`, true);
      }
    }

    /* ============================================================
     * Shared Crop Transform Helper (Single-View vs Side-by-Side)
     * ============================================================ */
    function computeMediaTransformCss(isDual, cropMode, focusX, heightCss = "100%") {
      const hStyle = typeof heightCss === "number" ? `${heightCss}px` : heightCss;
      if (cropMode === "raw") {
        return {
          css: `width:100%; height:${hStyle}; object-fit:contain; transform:none; display:block;`,
          badge: isDual ? "双视角全画幅 (1080×720)" : "单视角全画幅 (16:9)",
          showPan: false,
        };
      }
      if (cropMode === "front_focus") {
        if (isDual) {
          return {
            css: `width:200%; max-width:none; height:${hStyle}; object-fit:cover; object-position:left center; transform:translateX(0%); display:block;`,
            badge: "双视角 · 正面全身 (含步伐)",
            showPan: false,
          };
        }
        return {
          css: `width:100%; height:${hStyle}; object-fit:cover; object-position:${focusX}% center; transform:none; display:block;`,
          badge: `单视角 · 正面全身 (焦点 ${focusX}%)`,
          showPan: true,
        };
      }
      if (isDual) {
        return {
          css: `width:200%; max-width:none; height:${hStyle}; object-fit:cover; object-position:left center; transform:translateX(-50%); display:block;`,
          badge: "双视角 · 背面镜面全身",
          showPan: false,
        };
      }
      return {
        css: `width:100%; height:${hStyle}; object-fit:cover; object-position:${focusX}% center; transform:scale(1.2); transform-origin:${focusX}% 50%; display:block;`,
        badge: `单视角 · 镜面全身 (焦点 ${focusX}%)`,
        showPan: true,
      };
    }

    /* ============================================================
     * Single-Stroke Multi-Session Analysis (Forehand OR Backhand)
     * ============================================================ */
    function resolveGlobalSwing(side, sessionId, eventId) {
      if (!globalProfileData) return null;
      const sidePool = (globalProfileData[side]?.swings) || [];
      if (!sidePool.length) return null;
      const exact = sidePool.find(sw => sw.session_id === sessionId && sw.event_id === Number(eventId));
      if (exact) return exact;
      const inSess = sidePool.find(sw => sw.session_id === sessionId);
      return inSess || sidePool[0];
    }

    function buildDefaultGlobalSlots(sideKey, preset) {
      const sideData = globalProfileData?.[sideKey] || {};
      const pool = sideData.swings || [];
      const pre = sideData.video_presets || {};
      const sideName = sideKey === "backhand" ? "反手" : "正手";

      const mkSlot = (label, refObj, fallbackIdx = 0) => {
        if (refObj?.session_id) {
          return { label, side: sideKey, sessionId: refObj.session_id, eventId: refObj.event_id };
        }
        const fb = pool[Math.max(0, Math.min(pool.length - 1, fallbackIdx))] || null;
        return fb
          ? { label, side: sideKey, sessionId: fb.session_id, eventId: fb.event_id }
          : { label, side: sideKey, sessionId: "", eventId: 0 };
      };

      if (preset === "timeline_4") {
        const tl = pre.timeline_4 || [];
        const titles = [`${sideName} · 最早基准`, `${sideName} · 早期阶段`, `${sideName} · 近期阶段`, `${sideName} · 最新状态`];
        return [0, 1, 2, 3].map(i => mkSlot(titles[i], tl[i], i));
      }

      if (preset === "best_worst_4") {
        const withMedia = pool.filter(sw => sw.clip_url || sw.impact_freeze_url);
        const src = withMedia.length ? withMedia : pool;
        const ranked = [...src].sort((a, b) => {
          const sa = a.racket_max_speed_px_s ?? ((a.arm_extension || 0) + (a.shoulder_turn_change || 0));
          const sb = b.racket_max_speed_px_s ?? ((b.arm_extension || 0) + (b.shoulder_turn_change || 0));
          return sa - sb;
        });
        const best1 = ranked[ranked.length - 1] || null;
        const best2 = ranked[Math.max(0, ranked.length - 2)] || best1;
        const worst1 = ranked[0] || null;
        const worst2 = ranked[Math.min(ranked.length - 1, 1)] || worst1;
        return [
          mkSlot(`${sideName} · 最佳发力 #1`, best1),
          mkSlot(`${sideName} · 次佳对照 #2`, best2),
          mkSlot(`${sideName} · 最弱待改 #1`, worst1),
          mkSlot(`${sideName} · 偏弱对照 #2`, worst2),
        ];
      }

      // Default: "conflict_4" — cross-session discrepancy across same stroke side
      const cf = pre.conflict_4 || pre.timeline_4 || [];
      const metricTag = pre.conflict_metric_label ? ` (${pre.conflict_metric_label})` : "";
      const titles = [
        `极差最低场${metricTag}`,
        `偏低对照场`,
        `偏高对照场`,
        `极差最高场${metricTag}`,
      ];
      return [0, 1, 2, 3].map(i => mkSlot(titles[i], cf[i], i));
    }

    function setGlobalGridPreset(preset) {
      const sideKey = activeStrokeSide();
      globalGridPreset = preset;
      if (preset !== "custom_2x2" || globalSlots.length !== 4) {
        globalSlots = buildDefaultGlobalSlots(sideKey, preset === "custom_2x2" ? "conflict_4" : preset);
      }
      renderSingleStrokeDashboard(sideKey);
    }

    function jumpGridToConflictSessions(minSid, maxSid, metricKey, metricLabel) {
      const sideKey = activeStrokeSide();
      const sideData = globalProfileData?.[sideKey] || {};
      const pool = sideData.swings || [];
      if (!pool.length) return;

      const pickRepInSession = (sid) => {
        const group = pool.filter(sw => sw.session_id === sid);
        if (!group.length) return null;
        const withMedia = group.filter(sw => sw.clip_url || sw.impact_freeze_url);
        const src = withMedia.length ? withMedia : group;
        const vals = src.map(sw => sw[metricKey]).filter(v => v != null);
        if (!vals.length) return src[0];
        vals.sort((a, b) => a - b);
        const med = vals[Math.floor(vals.length / 2)];
        return src.reduce((best, cur) => (
          Math.abs((cur[metricKey] ?? med) - med) < Math.abs((best[metricKey] ?? med) - med) ? cur : best
        ), src[0]);
      };

      const minSw = pickRepInSession(minSid);
      const maxSw = pickRepInSession(maxSid);
      const otherSids = [];
      (sideData.trend_series || []).forEach(pt => {
        if (pt.session_id !== minSid && pt.session_id !== maxSid) otherSids.push(pt.session_id);
      });
      const midSw1 = otherSids.length ? pickRepInSession(otherSids[0]) : minSw;
      const midSw2 = otherSids.length > 1 ? pickRepInSession(otherSids[otherSids.length - 1]) : maxSw;

      const picked = [minSw, midSw1, midSw2, maxSw].filter(Boolean);
      const labels = [
        `最低 ${metricLabel} 场`,
        `中间对照 A`,
        `中间对照 B`,
        `最高 ${metricLabel} 场`,
      ];
      globalGridPreset = "custom_2x2";
      if (TREND_METRIC_OPTIONS.some(m => m.key === metricKey)) {
        globalTrendMetricKey = metricKey;
      }
      globalSlots = [0, 1, 2, 3].map(i => {
        const sw = picked[Math.min(i, picked.length - 1)];
        return {
          label: labels[i],
          side: sideKey,
          sessionId: sw?.session_id || "",
          eventId: sw?.event_id || 0,
        };
      });
      renderSingleStrokeDashboard(sideKey);
      const sec = document.getElementById("grid-2x2-section");
      if (sec?.scrollIntoView) sec.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function setGlobalCropMode(mode) {
      compareCropMode = mode;
      [0, 1, 2, 3].forEach(idx => {
        const slot = globalSlots[idx];
        const sw = slot ? resolveGlobalSwing(slot.side, slot.sessionId, slot.eventId) : null;
        applyGridMediaTransform(idx, sw?.view_mode || "single_view");
      });
      renderSingleStrokeDashboard(activeStrokeSide());
    }

    function applyGridMediaTransform(slotIdx, fallbackViewMode = "single_view") {
      const mediaEl = document.getElementById(`cmp-grid-video-${slotIdx}`) || document.getElementById(`cmp-grid-img-${slotIdx}`);
      const badgeEl = document.getElementById(`cmp-grid-badge-${slotIdx}`);
      const panWrap = document.getElementById(`cmp-grid-pan-${slotIdx}`);
      if (!mediaEl) return;

      const w = mediaEl.videoWidth || mediaEl.naturalWidth || 0;
      const h = mediaEl.videoHeight || mediaEl.naturalHeight || 0;
      let isDual = fallbackViewMode === "dual_view";
      if (w > 0 && h > 0) {
        isDual = (w / h) <= 1.58;
      }
      const fx = gridFocusX[slotIdx] ?? 76;
      const tf = computeMediaTransformCss(isDual, compareCropMode, fx, "100%");
      mediaEl.style.cssText = tf.css;
      if (badgeEl) badgeEl.textContent = tf.badge;
      if (panWrap) panWrap.hidden = !tf.showPan;
    }

    function nudgeGridFocus(slotIdx, delta) {
      const cur = gridFocusX[slotIdx] ?? 76;
      gridFocusX[slotIdx] = Math.max(15, Math.min(90, cur + delta));
      const slot = globalSlots[slotIdx];
      const sw = slot ? resolveGlobalSwing(slot.side, slot.sessionId, slot.eventId) : null;
      applyGridMediaTransform(slotIdx, sw?.view_mode || "single_view");
    }

    function onChangeGridSlotSession(slotIdx) {
      const sessSel = document.getElementById(`grid-sess-${slotIdx}`);
      if (!sessSel || !globalSlots[slotIdx]) return;
      const side = globalSlots[slotIdx].side;
      const pool = (globalProfileData?.[side]?.swings || []).filter(sw => sw.session_id === sessSel.value);
      const pick = pool[0] || null;
      globalSlots[slotIdx].sessionId = sessSel.value;
      globalSlots[slotIdx].eventId = pick?.event_id || 0;
      renderSingleStrokeDashboard(activeStrokeSide());
    }

    function onChangeGridSlotSwing(slotIdx) {
      const swSel = document.getElementById(`grid-swing-${slotIdx}`);
      if (!swSel || !globalSlots[slotIdx]) return;
      globalSlots[slotIdx].eventId = Number(swSel.value);
      renderSingleStrokeDashboard(activeStrokeSide());
    }

    function syncGridPlayClips(rate = 0.25) {
      [0, 1, 2, 3].forEach(i => {
        const v = document.getElementById(`cmp-grid-video-${i}`);
        if (!v) return;
        const cSec = parseFloat(v.dataset.contactSec || "0") || 0;
        v.pause();
        v.playbackRate = rate;
        v.currentTime = Math.max(0, cSec - 0.5);
        v.play().catch(() => {});
      });
    }

    function syncGridPauseClips() {
      [0, 1, 2, 3].forEach(i => {
        const v = document.getElementById(`cmp-grid-video-${i}`);
        if (v) v.pause();
      });
    }

    function syncGridSeekContact() {
      [0, 1, 2, 3].forEach(i => {
        const v = document.getElementById(`cmp-grid-video-${i}`);
        if (!v) return;
        const cSec = parseFloat(v.dataset.contactSec || "0") || 0;
        v.pause();
        v.currentTime = Math.max(0, cSec);
      });
    }

    function syncGridStepFrames(deltaFrames) {
      const stepSec = deltaFrames * 0.04;
      [0, 1, 2, 3].forEach(i => {
        const v = document.getElementById(`cmp-grid-video-${i}`);
        if (!v) return;
        v.pause();
        const dur = Number.isFinite(v.duration) ? v.duration : 9999;
        v.currentTime = Math.max(0, Math.min(dur, (v.currentTime || 0) + stepSec));
      });
    }

    function setGlobalTrendMetric(metricKey) {
      globalTrendMetricKey = metricKey;
      renderSingleStrokeDashboard(activeStrokeSide());
    }

    function renderGridSlotCardHtml(slot, idx) {
      const sw = resolveGlobalSwing(slot.side, slot.sessionId, slot.eventId);
      const isFh = slot.side === "forehand";
      const accentColor = isFh ? "#38bdf8" : "#4ade80";
      const sideLabel = isFh ? "正手" : "反手";
      const sidePool = globalProfileData?.[slot.side]?.swings || [];

      const sessMap = new Map();
      sidePool.forEach(item => {
        if (!sessMap.has(item.session_id)) sessMap.set(item.session_id, item.display_time);
      });
      const sessList = Array.from(sessMap.entries());
      const curSid = sw?.session_id || "";
      const swingsInSess = sidePool.filter(item => item.session_id === curSid);

      const titleText = slot.label || `${sideLabel}槽位 #${idx + 1}`;

      if (!sw) {
        return `
          <div style="background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:10px;">
            <strong style="font-size:12px; color:${accentColor}; display:block; margin-bottom:8px;">${escapeHtml(titleText)}</strong>
            <div class="cmp-grid-stage" style="display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:12px;">
              暂无${sideLabel}样本
            </div>
          </div>
        `;
      }

      const startF = sw.start_frame || 0;
      const contactF = sw.contact_frame ?? startF;
      const offsetSec = Math.max(0, (contactF - startF) / 25.0).toFixed(2);
      const mediaInner = sw.clip_url ? `
        <video id="cmp-grid-video-${idx}" src="${sw.clip_url}" poster="${sw.impact_freeze_url || ''}"
               muted controls loop playsinline preload="metadata"
               data-contact-sec="${offsetSec}"
               onloadedmetadata="applyGridMediaTransform(${idx}, '${sw.view_mode || 'single_view'}')"
               style="width:100%; height:100%; object-fit:contain; display:block;"></video>
      ` : (sw.impact_freeze_url ? `
        <img id="cmp-grid-img-${idx}" src="${sw.impact_freeze_url}" alt="触球定格"
             onload="applyGridMediaTransform(${idx}, '${sw.view_mode || 'single_view'}')"
             style="width:100%; height:100%; object-fit:contain; display:block;"/>
      ` : `<div style="height:100%; display:flex; align-items:center; justify-content:center; color:var(--muted);">无视频/定格图</div>`);

      setTimeout(() => applyGridMediaTransform(idx, sw.view_mode || "single_view"), 20);

      const sessOpts = sessList.map(([sid, dt]) => (
        `<option value="${escapeHtml(sid)}" ${sid === curSid ? "selected" : ""}>${escapeHtml(dt)}</option>`
      )).join("");
      const swOpts = swingsInSess.map(item => {
        const spd = item.racket_max_speed_px_s != null ? ` · ${Math.round(item.racket_max_speed_px_s)}px/s` : "";
        return `<option value="${item.event_id}" ${item.event_id === sw.event_id ? "selected" : ""}>#${item.event_id}${spd}</option>`;
      }).join("");

      const turnVal = sw.shoulder_turn_change != null ? `${Number(sw.shoulder_turn_change).toFixed(1)}°` : "—";
      const sepVal = sw.hip_shoulder_separation != null ? `${Number(sw.hip_shoulder_separation).toFixed(1)}°` : "—";
      const extVal = sw.arm_extension != null ? `${Number(sw.arm_extension).toFixed(1)}°` : "—";
      const kneeVal = sw.preparation_knee_flexion != null ? `${Number(sw.preparation_knee_flexion).toFixed(1)}°` : "—";
      const spdVal = sw.racket_max_speed_px_s != null ? `${Math.round(sw.racket_max_speed_px_s)} px/s` : "—";

      return `
        <div style="background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:8px; display:flex; flex-direction:column;">
          <div style="display:flex; flex-direction:column; gap:5px; margin-bottom:6px;">
            <strong style="font-size:11.5px; color:${accentColor}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${escapeHtml(titleText)}">${escapeHtml(titleText)} [${sw.view_mode === "dual_view" ? "双视角" : "单视角"}]</strong>
            <div style="display:flex; gap:4px; align-items:center;">
              <select id="grid-sess-${idx}" onchange="onChangeGridSlotSession(${idx})" style="flex:1; min-width:0; height:24px; font-size:11px; background:#0a1018; border:1px solid var(--line); border-radius:5px; color:var(--text); padding:0 4px;">${sessOpts}</select>
              <select id="grid-swing-${idx}" onchange="onChangeGridSlotSwing(${idx})" style="width:92px; flex:0 0 auto; height:24px; font-size:11px; background:#0a1018; border:1px solid var(--line); border-radius:5px; color:var(--text); padding:0 4px;">${swOpts}</select>
            </div>
          </div>
          <div class="cmp-grid-stage">
            ${mediaInner}
            <div style="position:absolute; top:6px; left:6px; right:6px; display:flex; justify-content:space-between; align-items:center; pointer-events:none;">
              <span id="cmp-grid-badge-${idx}" style="font-size:10px; padding:2px 6px; border-radius:4px; background:rgba(0,0,0,0.72); color:#e2e8f0; border:1px solid rgba(255,255,255,0.14);">视角加载中...</span>
              <span id="cmp-grid-pan-${idx}" style="pointer-events:auto;" hidden>
                <button type="button" onclick="nudgeGridFocus(${idx}, -8)" style="font-size:10px; padding:1px 6px; background:rgba(0,0,0,0.75); color:#fff; border:1px solid rgba(255,255,255,0.22); border-radius:3px; cursor:pointer;">◀左</button>
                <button type="button" onclick="nudgeGridFocus(${idx}, +8)" style="font-size:10px; padding:1px 6px; background:rgba(0,0,0,0.75); color:#fff; border:1px solid rgba(255,255,255,0.22); border-radius:3px; cursor:pointer;">右▶</button>
              </span>
            </div>
          </div>
          <div style="display:flex; flex-direction:column; font-size:11px; color:var(--muted); margin-top:6px; gap:2px; line-height:1.4;">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:4px;">
              <span>${escapeHtml(sw.display_time)} · 触球 ${sw.contact_frame ?? "—"}F</span>
              <button type="button" class="button" style="min-height:22px; padding:0 7px; font-size:10.5px;" onclick="openTrajectoryModal('${escapeHtml(sw.session_id)}', ${sw.event_id})" title="查看本拍逐帧平滑拟合曲线与异常剔除明细">📈 拟合曲线</button>
            </div>
            <span>转肩 ${turnVal} · 分离 ${sepVal} · 肘角 ${extVal}</span>
            <span>屈膝 ${kneeVal} · 拍速 ${spdVal}</span>
          </div>
        </div>
      `;
    }

    const RADAR_8_AXES = [
      { key: "shoulder_turn_change", label: "转肩", unit: "°", refMax: 65, digits: 1 },
      { key: "hip_shoulder_separation", label: "分离", unit: "°", refMax: 20, digits: 1 },
      { key: "arm_extension", label: "肘展", unit: "°", refMax: 180, digits: 0 },
      { key: "preparation_knee_flexion", label: "屈膝", unit: "°", refMax: 65, digits: 1 },
      { key: "stance", label: "站位", unit: "°", refMax: 90, digits: 0 },
      { key: "takeback_depth", label: "引拍", unit: "", refMax: 2.5, digits: 2 },
      { key: "weight_transfer", label: "横移", unit: "bw", refMax: 1.8, digits: 2 },
      { key: "racket_max_speed", label: "拍速", unit: "", refMax: 5000, digits: 0 },
    ];

    function buildTrajMapForSide(sideKey) {
      const m = new Map();
      const list = trajectorySummaryBySide[sideKey]?.swings || [];
      list.forEach(it => m.set(`${it.session_id}#${it.event_id}`, it));
      return m;
    }

    function getSwingRadarValues(sw, diag) {
      const out = {};
      RADAR_8_AXES.forEach(ax => {
        const refV = diag?.summary?.[ax.key]?.refined;
        let v = refV != null && Number.isFinite(Number(refV)) ? Number(refV) : null;
        if (v == null) {
          const rawFallback = ax.key === "racket_max_speed" ? (sw.racket_max_speed ?? sw.racket_max_speed_px_s) : sw[ax.key];
          if (rawFallback != null && Number.isFinite(Number(rawFallback))) {
            v = Number(rawFallback);
          }
        }
        out[ax.key] = v;
      });
      return out;
    }

    function getSideRadarBaseline(sideKey, sideData, trajMap) {
      const swings = sideData?.swings || [];
      const perKey = {};
      RADAR_8_AXES.forEach(ax => { perKey[ax.key] = []; });
      swings.forEach(sw => {
        const diag = trajMap.get(`${sw.session_id}#${sw.event_id}`) || null;
        if (diag && diag.tier === "C") return;
        const vals = getSwingRadarValues(sw, diag);
        RADAR_8_AXES.forEach(ax => {
          if (vals[ax.key] != null) perKey[ax.key].push(vals[ax.key]);
        });
      });
      const med = {};
      RADAR_8_AXES.forEach(ax => {
        const m = medianOfNums(perKey[ax.key]);
        med[ax.key] = m != null ? m : (sideData?.metrics?.[ax.key]?.median ?? null);
      });
      return med;
    }

    function formatRadarAxisVal(val, ax) {
      if (val == null || !Number.isFinite(Number(val))) return "—";
      const n = Number(val);
      return ax.digits === 0 ? `${Math.round(n)}${ax.unit}` : `${n.toFixed(ax.digits)}${ax.unit}`;
    }

    function selectRadarSwing(key) {
      selectedRadarSwingKey = key || "";
      renderSingleStrokeDashboard(activeStrokeSide());
    }

    function stepRadarSwing(delta) {
      const sideKey = activeStrokeSide();
      const swings = globalProfileData?.[sideKey]?.swings || [];
      if (!swings.length) return;
      const keys = ["", ...swings.map(s => `${s.session_id}#${s.event_id}`)];
      const curIdx = Math.max(0, keys.indexOf(selectedRadarSwingKey));
      const nextIdx = (curIdx + delta + keys.length) % keys.length;
      selectedRadarSwingKey = keys[nextIdx];
      renderSingleStrokeDashboard(sideKey);
    }

    function loadSwingIntoGridSlot(sessionId, eventId) {
      const sideKey = activeStrokeSide();
      if (globalSlots.length < 4) {
        globalSlots = buildDefaultGlobalSlots(sideKey, "conflict_4");
      }
      globalGridPreset = "custom_2x2";
      globalSlots[0] = {
        label: `指定复盘 (#${eventId})`,
        side: sideKey,
        sessionId: String(sessionId),
        eventId: Number(eventId),
      };
      renderSingleStrokeDashboard(sideKey);
      const sec = document.getElementById("grid-2x2-section");
      if (sec) sec.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function computeFittedPersonalAttribution(sideKey, sideData) {
      const swings = sideData?.swings || [];
      const trajMap = buildTrajMapForSide(sideKey);
      const hasFitted = trajMap.size > 0;
      if (!hasFitted) {
        return { ...(sideData.personal_attribution || {}), fitted: false };
      }
      const valid = [];
      swings.forEach(sw => {
        const key = `${sw.session_id}#${sw.event_id}`;
        const diag = trajMap.get(key) || null;
        if (diag && diag.tier === "C") return;
        const vals = getSwingRadarValues(sw, diag);
        if (vals.racket_max_speed != null && Number.isFinite(vals.racket_max_speed)) {
          valid.push({ sw, vals });
        }
      });
      if (valid.length < 4) {
        return { ...(sideData.personal_attribution || {}), fitted: false };
      }
      valid.sort((a, b) => b.vals.racket_max_speed - a.vals.racket_max_speed);
      const k = Math.max(2, Math.floor(valid.length * 0.25));
      const topGrp = valid.slice(0, k);
      const botGrp = valid.slice(valid.length - k);
      const topSpd = medianOfNums(topGrp.map(x => x.vals.racket_max_speed));
      const botSpd = medianOfNums(botGrp.map(x => x.vals.racket_max_speed));

      const specs = [
        { key: "shoulder_turn_change", label: "转肩角度变化", unit: "°" },
        { key: "preparation_knee_flexion", label: "准备期屈膝角", unit: "°" },
        { key: "arm_extension", label: "持拍侧肘伸展", unit: "°" },
        { key: "hip_shoulder_separation", label: "肩髋分离角", unit: "°" },
        { key: "weight_transfer", label: "躯干重心横移", unit: "bw" },
        { key: "takeback_depth", label: "镜面引拍深度", unit: "" },
      ];
      const factors = [];
      specs.forEach(sp => {
        const allV = valid.map(x => x.vals[sp.key]).filter(v => v != null && Number.isFinite(v)).sort((a, b) => a - b);
        const tV = topGrp.map(x => x.vals[sp.key]).filter(v => v != null && Number.isFinite(v));
        const bV = botGrp.map(x => x.vals[sp.key]).filter(v => v != null && Number.isFinite(v));
        if (allV.length < 4 || !tV.length || !bV.length) return;
        const tMed = medianOfNums(tV);
        const bMed = medianOfNums(bV);
        const q1 = allV[Math.floor(allV.length * 0.25)];
        const q3 = allV[Math.floor(allV.length * 0.75)];
        const spread = Math.max(q3 - q1, Math.abs(medianOfNums(allV)) * 0.08, 1e-3);
        const delta = tMed - bMed;
        const effect = Math.abs(delta) / spread;
        factors.push({
          key: sp.key,
          label: sp.label,
          unit: sp.unit,
          top_median: Number(tMed.toFixed(2)),
          bottom_median: Number(bMed.toFixed(2)),
          delta: Number(delta.toFixed(2)),
          effect_ratio: Number(effect.toFixed(2)),
        });
      });
      factors.sort((a, b) => b.effect_ratio - a.effect_ratio);
      const top2 = factors.slice(0, 2);
      const top2Str = top2.map(f => {
        const sign = f.delta >= 0 ? "+" : "";
        return `【${f.label}】（好球 ${f.top_median}${f.unit} vs 弱球 ${f.bottom_median}${f.unit}，差异 ${sign}${f.delta}${f.unit}）`;
      }).join(" 与 ");
      return {
        available: true,
        fitted: true,
        sample_count: valid.length,
        insight: `基于拟合修复后的 ${valid.length} 拍有效样本（已剔除 Tier C 误切）：前 25% 高质球（拍速中位 ${Math.round(topSpd)} px/s）相比后 25% 弱球（${Math.round(botSpd)} px/s），差异最显著的发力开关是 ${top2Str || "各维度较接近"}。`,
        factors,
      };
    }

    function buildBiomechanicalFlaws(sideKey, sideData) {
      const swings = sideData?.swings || [];
      const trajMap = buildTrajMapForSide(sideKey);
      const valid = [];
      swings.forEach(sw => {
        const key = `${sw.session_id}#${sw.event_id}`;
        const diag = trajMap.get(key) || null;
        if (diag && diag.tier === "C") return;
        const vals = getSwingRadarValues(sw, diag);
        valid.push({ sw, vals });
      });
      const n = valid.length;
      if (!n) return { total: 0, flaws: [] };
      const mid = Math.floor(n / 2);

      const rules = [
        {
          id: "low_sep",
          label: "肩髋分离不足 (< 10° · 缺少髋先肩后蓄力)",
          check: ({ vals }) => vals.hip_shoulder_separation != null && vals.hip_shoulder_separation < 10.0,
        },
        {
          id: "low_turn",
          label: "转肩幅度偏小 (< 30° · 侧身蓄力不够充分)",
          check: ({ vals }) => vals.shoulder_turn_change != null && vals.shoulder_turn_change < 30.0,
        },
        {
          id: "shallow_knee",
          label: "准备期屈膝偏浅 (< 35° · 下肢蹬转不足)",
          check: ({ vals }) => vals.preparation_knee_flexion != null && vals.preparation_knee_flexion < 35.0,
        },
        {
          id: "bent_elbow",
          label: "触球略缩肘 (< 155° · 击球点偏挤)",
          check: ({ vals }) => vals.arm_extension != null && vals.arm_extension < 155.0,
        },
        {
          id: "high_drift",
          label: "回位期重心漂移偏大 (> 0.45 bw · 随挥失衡)",
          check: ({ sw }) => sw.balance_drift != null && Number(sw.balance_drift) > 0.45,
        },
        {
          id: "low_weight",
          label: "躯干迎前横移偏小 (< 0.40 bw · 重心未压前)",
          check: ({ vals }) => vals.weight_transfer != null && vals.weight_transfer < 0.40,
        },
      ];

      const flaws = [];
      rules.forEach(r => {
        const hits = valid.map(x => (r.check(x) ? 1 : 0));
        const cnt = hits.reduce((a, b) => a + b, 0);
        if (!cnt) return;
        const rate = Math.round((cnt / n) * 1000) / 10;
        let earlyRate = null, recentRate = null, deltaPct = null;
        if (n >= 4 && mid > 0) {
          const eCnt = hits.slice(0, mid).reduce((a, b) => a + b, 0);
          const rCnt = hits.slice(mid).reduce((a, b) => a + b, 0);
          earlyRate = Math.round((eCnt / mid) * 1000) / 10;
          recentRate = Math.round((rCnt / (n - mid)) * 1000) / 10;
          deltaPct = Math.round((recentRate - earlyRate) * 10) / 10;
        }
        flaws.push({
          label: r.label,
          count: cnt,
          rate_pct: rate,
          early_rate_pct: earlyRate,
          recent_rate_pct: recentRate,
          delta_pct: deltaPct,
        });
      });
      flaws.sort((a, b) => b.rate_pct - a.rate_pct);
      return { total: n, flaws };
    }

    function buildSingleSideRadarSvg(sideKey, sideData, accentColor) {
      const swings = sideData?.swings || [];
      const sidePrefix = sideKey === "backhand" ? "反" : "正";
      const trajMap = buildTrajMapForSide(sideKey);
      const baseline = getSideRadarBaseline(sideKey, sideData, trajMap);

      let activeSw = null;
      let activeSwIdx = -1;
      if (selectedRadarSwingKey) {
        activeSwIdx = swings.findIndex(s => `${s.session_id}#${s.event_id}` === selectedRadarSwingKey);
        if (activeSwIdx >= 0) activeSw = swings[activeSwIdx];
      }
      const activeDiag = activeSw ? (trajMap.get(selectedRadarSwingKey) || null) : null;
      const activeVals = activeSw ? getSwingRadarValues(activeSw, activeDiag) : null;

      const cx = 160, cy = 148, rMax = 88;
      const toPt = (idx, ratio) => {
        const angle = -Math.PI / 2 + (idx * 2 * Math.PI) / RADAR_8_AXES.length;
        const r = Math.max(0.08, Math.min(1.0, ratio)) * rMax;
        return `${(cx + r * Math.cos(angle)).toFixed(1)},${(cy + r * Math.sin(angle)).toFixed(1)}`;
      };
      const gridRings = [0.33, 0.66, 1.0].map(level => (
        `<polygon points="${RADAR_8_AXES.map((_, i) => toPt(i, level)).join(" ")}" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="1"/>`
      )).join("");

      const displayVals = activeVals || baseline;
      const axisLines = RADAR_8_AXES.map((ax, i) => {
        const angle = -Math.PI / 2 + (i * 2 * Math.PI) / RADAR_8_AXES.length;
        const x2 = cx + rMax * Math.cos(angle);
        const y2 = cy + rMax * Math.sin(angle);
        const lx = cx + (rMax + 34) * Math.cos(angle);
        const ly = cy + (rMax + 24) * Math.sin(angle);
        const valStr = formatRadarAxisVal(displayVals[ax.key], ax);
        return `
          <line x1="${cx}" y1="${cy}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
          <text x="${lx.toFixed(1)}" y="${(ly - 5).toFixed(1)}" fill="#cbd5e1" font-size="10" font-weight="600" text-anchor="middle">${ax.label}</text>
          <text x="${lx.toFixed(1)}" y="${(ly + 8).toFixed(1)}" fill="${activeVals ? accentColor : '#94a3b8'}" font-size="9.5" text-anchor="middle">${valStr}</text>
        `;
      }).join("");

      const polyFromObj = (obj) => RADAR_8_AXES.map((ax, i) => {
        const v = obj?.[ax.key];
        const ratio = (v != null && Number.isFinite(Number(v))) ? (Number(v) / ax.refMax) : 0.08;
        return toPt(i, ratio);
      }).join(" ");

      const polyFromField = (fieldName) => RADAR_8_AXES.map((ax, i) => {
        const m = sideData?.metrics?.[ax.key];
        const v = m?.[fieldName] ?? baseline[ax.key] ?? 0;
        return toPt(i, Number(v) / ax.refMax);
      }).join(" ");

      const ptsBaseline = polyFromObj(baseline);
      const fillAccent = accentColor === "#38bdf8" ? "rgba(56,189,248,0.22)" : "rgba(74,222,128,0.22)";
      const polygonsHtml = activeVals
        ? `
          <polygon points="${ptsBaseline}" fill="rgba(255,255,255,0.05)" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="4,3"/>
          <polygon points="${polyFromObj(activeVals)}" fill="${fillAccent}" stroke="${accentColor}" stroke-width="2.3"/>
        `
        : `
          <polygon points="${polyFromField("early_median")}" fill="rgba(251,191,36,0.11)" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="4,3"/>
          <polygon points="${ptsBaseline}" fill="${fillAccent}" stroke="${accentColor}" stroke-width="2.2"/>
        `;

      const selectOpts = [
        `<option value="" ${!selectedRadarSwingKey ? "selected" : ""}>全局 8 维中位 (全部 ${swings.length} 拍)</option>`,
        ...swings.map((sw, idx) => {
          const k = `${sw.session_id}#${sw.event_id}`;
          const dg = trajMap.get(k);
          const tTag = dg ? ` [Tier ${dg.tier}]` : "";
          return `<option value="${escapeHtml(k)}" ${k === selectedRadarSwingKey ? "selected" : ""}>${sidePrefix}${idx + 1} (#${sw.event_id})${tTag}</option>`;
        }),
      ].join("");

      const legendHtml = activeSw
        ? `
          <span><span style="display:inline-block;width:10px;height:2px;background:#94a3b8;vertical-align:middle;margin-right:4px;"></span>全局中位 (虚线)</span>
          <span><span style="display:inline-block;width:10px;height:2px;background:${accentColor};vertical-align:middle;margin-right:4px;"></span>${sidePrefix}${activeSwIdx + 1} (#${activeSw.event_id}) 实测</span>
        `
        : `
          <span><span style="display:inline-block;width:10px;height:2px;background:#fbbf24;vertical-align:middle;margin-right:4px;"></span>早期阶段 (虚线)</span>
          <span><span style="display:inline-block;width:10px;height:2px;background:${accentColor};vertical-align:middle;margin-right:4px;"></span>拟合全局中位 (实线)</span>
        `;

      const actionBtn = activeSw
        ? `<div style="text-align:center; margin-top:6px;"><button type="button" class="button" style="min-height:24px; padding:0 10px; font-size:11px;" onclick="openTrajectoryModal('${escapeHtml(activeSw.session_id)}', ${activeSw.event_id})">📈 查看 ${sidePrefix}${activeSwIdx + 1} 逐帧拟合曲线</button></div>`
        : "";

      return `
        <div style="display:flex; align-items:center; justify-content:center; gap:4px; margin-bottom:4px;">
          <button type="button" class="button" style="min-height:26px; padding:0 7px; font-size:11px;" onclick="stepRadarSwing(-1)" title="上一拍">◀</button>
          <select onchange="selectRadarSwing(this.value)" style="flex:1; max-width:210px; height:26px; font-size:11.5px; background:#0a1018; border:1px solid var(--line); border-radius:6px; color:var(--text); padding:0 6px;">${selectOpts}</select>
          <button type="button" class="button" style="min-height:26px; padding:0 7px; font-size:11px;" onclick="stepRadarSwing(1)" title="下一拍">▶</button>
        </div>
        <svg viewBox="0 0 320 296" width="300" height="278" style="display:block; margin:0 auto;">
          ${gridRings}
          ${axisLines}
          ${polygonsHtml}
        </svg>
        <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px; font-size:11px; color:var(--muted); margin-top:2px;">
          ${legendHtml}
        </div>
        <div style="text-align:center; font-size:10.5px; color:var(--muted); margin-top:2px;">8 维全景：转肩·分离·肘展·屈膝·站位·引拍·横移·拍速（优先取拟合值）</div>
        ${actionBtn}
      `;
    }

    function buildPerSwingRadarGalleryHtml(sideKey, sideData, accentColor) {
      const swings = sideData?.swings || [];
      if (!swings.length) return "";
      const sidePrefix = sideKey === "backhand" ? "反" : "正";
      const sideShort = sideKey === "backhand" ? "反手" : "正手";
      const trajMap = buildTrajMapForSide(sideKey);
      const baseline = getSideRadarBaseline(sideKey, sideData, trajMap);

      const cx = 64, cy = 62, rMax = 44;
      const toPt = (idx, ratio) => {
        const angle = -Math.PI / 2 + (idx * 2 * Math.PI) / RADAR_8_AXES.length;
        const r = Math.max(0.08, Math.min(1.0, ratio)) * rMax;
        return `${(cx + r * Math.cos(angle)).toFixed(1)},${(cy + r * Math.sin(angle)).toFixed(1)}`;
      };
      const basePoly = RADAR_8_AXES.map((ax, i) => {
        const v = baseline[ax.key];
        return toPt(i, (v != null && Number.isFinite(Number(v))) ? (Number(v) / ax.refMax) : 0.08);
      }).join(" ");
      const miniRings = [0.5, 1.0].map(lv => (
        `<polygon points="${RADAR_8_AXES.map((_, i) => toPt(i, lv)).join(" ")}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="0.8"/>`
      )).join("");
      const miniLabels = RADAR_8_AXES.map((ax, i) => {
        const angle = -Math.PI / 2 + (i * 2 * Math.PI) / RADAR_8_AXES.length;
        const lx = cx + (rMax + 11) * Math.cos(angle);
        const ly = cy + (rMax + 9) * Math.sin(angle);
        return `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" fill="#8d9caf" font-size="7.5" text-anchor="middle" dominant-baseline="middle">${ax.label}</text>`;
      }).join("");

      const cardsHtml = swings.map((sw, idx) => {
        const key = `${sw.session_id}#${sw.event_id}`;
        const diag = trajMap.get(key) || null;
        const vals = getSwingRadarValues(sw, diag);
        const isSel = key === selectedRadarSwingKey;
        const tier = diag?.tier || "A";
        const tierColor = tier === "A" ? "#4ade80" : (tier === "B" ? "#fbbf24" : "#f87171");
        const swColor = tier === "C" ? "#f87171" : accentColor;
        const swFill = tier === "C" ? "rgba(248,113,113,0.16)" : (accentColor === "#38bdf8" ? "rgba(56,189,248,0.22)" : "rgba(74,222,128,0.22)");
        const swPoly = RADAR_8_AXES.map((ax, i) => {
          const v = vals[ax.key];
          return toPt(i, (v != null && Number.isFinite(Number(v))) ? (Number(v) / ax.refMax) : 0.08);
        }).join(" ");

        return `
          <div onclick="selectRadarSwing('${escapeHtml(key)}')" style="cursor:pointer; padding:8px; border-radius:9px; background:${isSel ? 'rgba(56,189,248,0.10)' : 'rgba(0,0,0,0.20)'}; border:1px solid ${isSel ? accentColor : 'rgba(255,255,255,0.08)'}; transition:border-color 0.15s;">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:4px; margin-bottom:2px; font-size:11px;">
              <strong style="color:${isSel ? accentColor : 'var(--text)'};">${sidePrefix}${idx + 1} <span style="color:var(--muted); font-weight:400;">#${sw.event_id}</span></strong>
              <span style="display:inline-flex; align-items:center; gap:4px;">
                <span style="font-size:9.5px; padding:0 5px; border-radius:3px; background:${tierColor}22; color:${tierColor};">T${tier}</span>
                <button type="button" onclick="event.stopPropagation(); openTrajectoryModal('${escapeHtml(sw.session_id)}', ${sw.event_id})" style="background:rgba(255,255,255,0.06); border:1px solid var(--line); border-radius:4px; color:var(--text); font-size:10px; padding:0 5px; height:18px; cursor:pointer;" title="打开逐帧拟合曲线">📈</button>
              </span>
            </div>
            <svg viewBox="0 0 128 124" style="width:100%; max-width:144px; height:auto; display:block; margin:0 auto;">
              ${miniRings}
              ${miniLabels}
              <polygon points="${basePoly}" fill="none" stroke="rgba(148,163,184,0.45)" stroke-width="1" stroke-dasharray="2,2"/>
              <polygon points="${swPoly}" fill="${swFill}" stroke="${swColor}" stroke-width="1.6"/>
            </svg>
            <div style="font-size:10px; color:var(--muted); line-height:1.4; text-align:center; margin-top:2px;">
              转肩 ${formatRadarAxisVal(vals.shoulder_turn_change, RADAR_8_AXES[0])} · 屈膝 ${formatRadarAxisVal(vals.preparation_knee_flexion, RADAR_8_AXES[3])}<br>
              肘展 ${formatRadarAxisVal(vals.arm_extension, RADAR_8_AXES[2])} · 拍速 ${formatRadarAxisVal(vals.racket_max_speed, RADAR_8_AXES[7])}
            </div>
          </div>
        `;
      }).join("");

      return `
        <details open style="padding:12px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:10px; margin-bottom:14px;">
          <summary style="cursor:pointer; color:var(--text); font-size:13px; font-weight:600;">
            🕸️ 每拍${sideShort} 8 维动作雷达缩略图墙（共 ${swings.length} 拍 · 灰虚线=全局中位 · 点击任意拍叠加放大，点 📈 看逐帧拟合）
          </summary>
          <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(168px, 1fr)); gap:8px; margin-top:10px;">
            ${cardsHtml}
          </div>
        </details>
      `;
    }

    function medianOfNums(arr) {
      const nums = (arr || []).filter(v => v != null && Number.isFinite(Number(v))).map(Number).sort((a, b) => a - b);
      if (!nums.length) return null;
      const mid = Math.floor(nums.length / 2);
      return nums.length % 2 === 1 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
    }

    function buildSingleSideTrendChartSvg(sideKey, sideData, metricKey, accentColor) {
      const cfg = TREND_METRIC_OPTIONS.find(m => m.key === metricKey) || TREND_METRIC_OPTIONS[0];
      const sideShort = sideKey === "backhand" ? "反手" : "正手";
      const pts = (sideData?.trend_series || []).filter(p => p[metricKey] != null);
      if (!pts.length) {
        return `<div style="padding:30px 12px; text-align:center; color:var(--muted); font-size:12px;">当前范围暂无${sideShort}「${escapeHtml(cfg.label)}」的场次趋势样本。</div>`;
      }

      // Compute per-session fitted medians from trajectorySummaryBySide[sideKey] (Tier A+B only)
      const trajData = trajectorySummaryBySide[sideKey];
      const fittedKeys = new Set(["arm_extension", "preparation_knee_flexion", "shoulder_turn_change", "racket_max_speed"]);
      const sessFittedMap = new Map();
      if (fittedKeys.has(metricKey) && trajData?.swings?.length) {
        const bySess = new Map();
        trajData.swings.forEach(it => {
          if (it.tier === "C") return;
          const v = it.summary?.[metricKey]?.refined;
          if (v != null && Number.isFinite(Number(v))) {
            if (!bySess.has(it.session_id)) bySess.set(it.session_id, []);
            bySess.get(it.session_id).push(Number(v));
          }
        });
        bySess.forEach((arr, sid) => {
          const med = medianOfNums(arr);
          if (med != null) sessFittedMap.set(sid, Number(med.toFixed(1)));
        });
      }
      const hasFitted = sessFittedMap.size > 0;

      const overallMed = sideData?.metrics?.[metricKey]?.median;
      const vals = [];
      pts.forEach(p => {
        vals.push(Number(p[metricKey]));
        if (sessFittedMap.has(p.session_id)) vals.push(sessFittedMap.get(p.session_id));
      });
      if (overallMed != null) vals.push(Number(overallMed));
      let minV = Math.min(...vals);
      let maxV = Math.max(...vals);
      if (Math.abs(maxV - minV) < 1e-6) {
        minV -= 1;
        maxV += 1;
      } else {
        const pad = (maxV - minV) * 0.14;
        minV -= pad;
        maxV += pad;
      }

      const W = 820, H = 224, padL = 54, padR = 24, padT = 22, padB = 40;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const xOf = (idx) => (pts.length <= 1 ? padL + plotW / 2 : padL + (idx / (pts.length - 1)) * plotW);
      const yOf = (v) => padT + (1 - (v - minV) / (maxV - minV)) * plotH;

      const gridLines = [0, 0.5, 1.0].map(t => {
        const y = padT + t * plotH;
        const val = maxV - t * (maxV - minV);
        const label = Math.abs(val) >= 100 ? Math.round(val) : val.toFixed(1);
        return `
          <line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
          <text x="${padL - 8}" y="${y.toFixed(1)}" fill="#8d9caf" font-size="10" text-anchor="end" dominant-baseline="middle">${label}</text>
        `;
      }).join("");

      const medLine = overallMed != null ? (() => {
        const my = yOf(Number(overallMed)).toFixed(1);
        return `
          <line x1="${padL}" y1="${my}" x2="${W - padR}" y2="${my}" stroke="rgba(251,191,36,0.45)" stroke-width="1.1" stroke-dasharray="5,4"/>
          <text x="${W - padR}" y="${Math.max(12, Number(my) - 5)}" fill="#fbbf24" font-size="10" text-anchor="end">${sideShort}原始中位 ${overallMed} ${escapeHtml(cfg.unit)}</text>
        `;
      })() : "";

      const rawCoords = pts.map((p, idx) => `${xOf(idx).toFixed(1)},${yOf(Number(p[metricKey])).toFixed(1)}`).join(" ");
      const fitCoords = hasFitted
        ? pts.map((p, idx) => sessFittedMap.has(p.session_id) ? `${xOf(idx).toFixed(1)},${yOf(sessFittedMap.get(p.session_id)).toFixed(1)}` : null).filter(Boolean).join(" ")
        : "";

      const dots = pts.map((p, idx) => {
        const cx = xOf(idx).toFixed(1);
        const rawV = Number(p[metricKey]);
        const fitV = sessFittedMap.get(p.session_id);
        const primaryV = hasFitted && fitV != null ? fitV : rawV;
        const cy = yOf(primaryV).toFixed(1);
        const viewTag = p.view_mode === "dual_view" ? "双视角" : "单视角";
        const shortDt = String(p.display_time || "").slice(5, 16);
        const tip = `场次${idx + 1} · ${p.display_time} [${sideShort} ${p.count}拍 · ${viewTag}] | 原始中位: ${rawV} ${cfg.unit}` +
          (fitV != null ? ` → 拟合中位: ${fitV} ${cfg.unit}` : "") +
          "（点击在新标签查看该场次）";
        return `
          ${hasFitted && fitV != null ? `<circle cx="${cx}" cy="${yOf(rawV).toFixed(1)}" r="3" fill="#64748b"/>` : ""}
          <circle cx="${cx}" cy="${cy}" r="5" fill="${accentColor}" stroke="#0a1018" stroke-width="1.5" style="cursor:pointer;" onclick="openSessionInConsole('${escapeHtml(p.session_id)}')">
            <title>${escapeHtml(tip)}</title>
          </circle>
          <text x="${cx}" y="${H - 22}" fill="#94a3b8" font-size="9.5" text-anchor="middle">场${idx + 1} (${shortDt})</text>
        `;
      }).join("");

      return `
        <svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; display:block;">
          ${gridLines}
          ${medLine}
          ${pts.length >= 2 ? `<polyline points="${rawCoords}" fill="none" stroke="${hasFitted ? '#64748b' : accentColor}" stroke-width="${hasFitted ? 1.4 : 2.2}" ${hasFitted ? 'stroke-dasharray="4,3"' : ''} stroke-linejoin="round"/>` : ""}
          ${hasFitted && pts.length >= 2 && fitCoords ? `<polyline points="${fitCoords}" fill="none" stroke="${accentColor}" stroke-width="2.5" stroke-linejoin="round"/>` : ""}
          ${dots}
          <text x="${W / 2}" y="${H - 6}" fill="#8d9caf" font-size="10" text-anchor="middle">
            ${sideShort}跨场次趋势 · 共 ${pts.length} 场${hasFitted ? " · 灰虚线=原始场次中位 实线=拟合场次中位(剔除误切)" : ""}${pts.length <= 1 ? "（需 ≥2 场不同训练视频才会连成跨场趋势线）" : " · 点击圆点打开该场次"}
          </text>
        </svg>
      `;
    }

    function setGlobalTrendViewMode(mode) {
      globalTrendViewMode = mode;
      renderSingleStrokeDashboard(activeStrokeSide());
    }

    async function loadTrajectorySummary(sideKey) {
      const playerSel = document.getElementById("compare-player-select");
      const daysSel = document.getElementById("compare-days-select");
      if (!playerSel) return;
      const pid = playerSel.value || "default";
      const days = daysSel ? daysSel.value : "all";
      try {
        const qs = new URLSearchParams({
          player_id: pid,
          stroke_side: sideKey,
          dedup: dedupEnabled ? "1" : "0",
        });
        if (days && days !== "all") qs.set("days", days);
        const res = await api(`/api/players/trajectory-summary?${qs.toString()}`);
        trajectorySummaryBySide[sideKey] = await res.json();
        if (pageViewMode !== "ab" && activeStrokeSide() === sideKey) {
          renderSingleStrokeDashboard(sideKey);
        }
      } catch (_) {}
    }

    function buildPerSwingFittedChartSvg(sideKey, sideData, metricKey, accentColor) {
      const cfg = TREND_METRIC_OPTIONS.find(m => m.key === metricKey) || TREND_METRIC_OPTIONS[0];
      const rawSwings = sideData?.swings || [];
      if (!rawSwings.length) {
        return `<div style="padding:26px 12px; text-align:center; color:var(--muted); font-size:12px;">当前范围暂无逐拍样本。</div>`;
      }
      const trajData = trajectorySummaryBySide[sideKey];
      const trajMap = new Map();
      (trajData?.swings || []).forEach(it => {
        trajMap.set(`${it.session_id}#${it.event_id}`, it);
      });
      const fittedKeys = new Set(["arm_extension", "preparation_knee_flexion", "shoulder_turn_change", "racket_max_speed"]);
      const hasFittedForMetric = fittedKeys.has(metricKey) && trajMap.size > 0;
      const capSpeed = metricKey === "racket_max_speed" ? 6500 : null;

      const items = rawSwings.map(sw => {
        const diag = trajMap.get(`${sw.session_id}#${sw.event_id}`) || null;
        const sumEntry = diag?.summary?.[metricKey];
        let rawVal = sumEntry?.raw ?? sw[metricKey];
        if (rawVal != null && capSpeed != null && rawVal > capSpeed) rawVal = capSpeed;
        const refVal = hasFittedForMetric && diag && diag.tier !== "C" ? sumEntry?.refined : null;
        return {
          session_id: sw.session_id,
          event_id: sw.event_id,
          display_time: sw.display_time,
          raw: rawVal != null && Number.isFinite(Number(rawVal)) ? Number(rawVal) : null,
          refined: refVal != null && Number.isFinite(Number(refVal)) ? Number(refVal) : null,
          tier: diag?.tier || "A",
          rejected: diag?.rejected_count || 0,
          interp: diag?.interpolated_count || 0,
          shift: diag?.shift_frames || 0,
        };
      });

      const vals = [];
      items.forEach(it => {
        if (it.raw != null) vals.push(it.raw);
        if (it.refined != null) vals.push(it.refined);
      });
      if (!vals.length) {
        return `<div style="padding:26px 12px; text-align:center; color:var(--muted); font-size:12px;">当前指标「${escapeHtml(cfg.label)}」暂无逐拍数值。</div>`;
      }
      let minV = Math.min(...vals);
      let maxV = Math.max(...vals);
      if (Math.abs(maxV - minV) < 1e-6) { minV -= 1; maxV += 1; }
      else { const pad = (maxV - minV) * 0.14; minV -= pad; maxV += pad; }

      const W = 820, H = 248, padL = 54, padR = 24, padT = 30, padB = 40;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const xOf = (idx) => (items.length <= 1 ? padL + plotW / 2 : padL + (idx / (items.length - 1)) * plotW);
      const yOf = (v) => padT + (1 - (v - minV) / (maxV - minV)) * plotH;
      const sidePrefix = sideKey === "backhand" ? "反" : "正";

      // Partition swings into chronological session spans to distinguish sessions visually
      const sessSpans = [];
      items.forEach((it, idx) => {
        const last = sessSpans[sessSpans.length - 1];
        if (!last || last.session_id !== it.session_id) {
          sessSpans.push({
            session_id: it.session_id,
            display_time: it.display_time,
            startIdx: idx,
            endIdx: idx,
            vals: [],
          });
        } else {
          last.endIdx = idx;
        }
        const vForMed = hasFittedForMetric ? (it.tier !== "C" ? it.refined : null) : it.raw;
        if (vForMed != null) sessSpans[sessSpans.length - 1].vals.push(vForMed);
      });

      const stepX = items.length > 1 ? plotW / (items.length - 1) : plotW;
      const sessionBandsHtml = sessSpans.map((sp, sIdx) => {
        const x1 = sIdx === 0 ? padL : (xOf(sp.startIdx) - stepX * 0.5);
        const x2 = sIdx === sessSpans.length - 1 ? (W - padR) : (xOf(sp.endIdx) + stepX * 0.5);
        const cx = (x1 + x2) / 2;
        const fill = sIdx % 2 === 0 ? "rgba(56,189,248,0.035)" : "rgba(168,85,247,0.045)";
        const divider = sIdx > 0
          ? `<line x1="${x1.toFixed(1)}" y1="${padT - 14}" x2="${x1.toFixed(1)}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.22)" stroke-width="1" stroke-dasharray="3,3"/>`
          : "";
        const med = medianOfNums(sp.vals);
        const medBar = med != null
          ? `<line x1="${(x1 + 4).toFixed(1)}" y1="${yOf(med).toFixed(1)}" x2="${(x2 - 4).toFixed(1)}" y2="${yOf(med).toFixed(1)}" stroke="rgba(251,191,36,0.55)" stroke-width="1.3" stroke-dasharray="5,3"/>`
          : "";
        const shortDt = String(sp.display_time || "").slice(5, 16);
        const countInSpan = sp.endIdx - sp.startIdx + 1;
        const labelTxt = `场次${sIdx + 1} · ${shortDt} (${sidePrefix}${countInSpan}拍${med != null ? ` · 中位${med.toFixed(1)}` : ""})`;
        return `
          <rect x="${x1.toFixed(1)}" y="${padT}" width="${Math.max(4, x2 - x1).toFixed(1)}" height="${plotH}" fill="${fill}"/>
          ${divider}
          ${medBar}
          <text x="${cx.toFixed(1)}" y="${padT - 8}" fill="#cbd5e1" font-size="9.5" text-anchor="middle">${escapeHtml(labelTxt)}</text>
        `;
      }).join("");

      const gridLines = [0, 0.5, 1.0].map(t => {
        const y = padT + t * plotH;
        const val = maxV - t * (maxV - minV);
        const label = Math.abs(val) >= 100 ? Math.round(val) : val.toFixed(1);
        return `
          <line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
          <text x="${padL - 8}" y="${(y + 3).toFixed(1)}" fill="#8d9caf" font-size="10" text-anchor="end">${label}</text>
        `;
      }).join("");

      // Break polylines per session so intra-session swings never falsely merge across different days
      const rawPathsHtml = sessSpans.map(sp => {
        const ptsStr = [];
        for (let i = sp.startIdx; i <= sp.endIdx; i++) {
          if (items[i].raw != null) ptsStr.push(`${xOf(i).toFixed(1)},${yOf(items[i].raw).toFixed(1)}`);
        }
        return ptsStr.length >= 2
          ? `<polyline points="${ptsStr.join(" ")}" fill="none" stroke="${hasFittedForMetric ? '#64748b' : accentColor}" stroke-width="${hasFittedForMetric ? 1.3 : 2.1}" ${hasFittedForMetric ? 'stroke-dasharray="4,3"' : ''} stroke-linejoin="round"/>`
          : "";
      }).join("");

      const refPathsHtml = hasFittedForMetric ? sessSpans.map(sp => {
        const ptsStr = [];
        for (let i = sp.startIdx; i <= sp.endIdx; i++) {
          if (items[i].refined != null) ptsStr.push(`${xOf(i).toFixed(1)},${yOf(items[i].refined).toFixed(1)}`);
        }
        return ptsStr.length >= 2
          ? `<polyline points="${ptsStr.join(" ")}" fill="none" stroke="${accentColor}" stroke-width="2.4" stroke-linejoin="round"/>`
          : "";
      }).join("") : "";

      const dotsHtml = items.map((it, i) => {
        const cx = xOf(i).toFixed(1);
        const showVal = hasFittedForMetric ? (it.tier === "C" ? it.raw : (it.refined ?? it.raw)) : it.raw;
        if (showVal == null) return "";
        const cy = yOf(showVal).toFixed(1);
        const tierColor = it.tier === "A" ? "#4ade80" : (it.tier === "B" ? "#fbbf24" : "#f87171");
        const tip = `${sidePrefix}${i + 1} (Event #${it.event_id}) [${it.display_time}] | 原始: ${it.raw ?? "—"} ${cfg.unit}` +
          (hasFittedForMetric ? ` → 拟合: ${it.refined ?? "误切不计入"} ${cfg.unit} | Tier ${it.tier} (剔除${it.rejected}帧/插值${it.interp}帧)` : "") +
          "（点击查看本拍逐帧拟合曲线）";
        if (hasFittedForMetric && it.tier === "C") {
          return `
            <g style="cursor:pointer;" onclick="openTrajectoryModal('${escapeHtml(it.session_id)}', ${it.event_id})">
              <line x1="${Number(cx) - 4}" y1="${Number(cy) - 4}" x2="${Number(cx) + 4}" y2="${Number(cy) + 4}" stroke="#f87171" stroke-width="2"/>
              <line x1="${Number(cx) + 4}" y1="${Number(cy) - 4}" x2="${Number(cx) - 4}" y2="${Number(cy) + 4}" stroke="#f87171" stroke-width="2"/>
              <circle cx="${cx}" cy="${cy}" r="7" fill="transparent"><title>${escapeHtml(tip)}</title></circle>
            </g>
          `;
        }
        return `
          <circle cx="${cx}" cy="${cy}" r="4.5" fill="${hasFittedForMetric ? tierColor : accentColor}" stroke="#0a1018" stroke-width="1.3" style="cursor:pointer;" onclick="openTrajectoryModal('${escapeHtml(it.session_id)}', ${it.event_id})">
            <title>${escapeHtml(tip)}</title>
          </circle>
        `;
      }).join("");

      const xTicks = items.map((it, i) => {
        if (items.length > 22 && i % 2 !== 0 && i !== items.length - 1) return "";
        return `<text x="${xOf(i).toFixed(1)}" y="${H - 22}" fill="#8d9caf" font-size="9" text-anchor="middle">${sidePrefix}${i + 1}</text>`;
      }).join("");

      const tc = trajData?.tier_counts;
      const tierInfo = tc
        ? ` · Tier A高置信 ${tc.A}拍 / B插值 ${tc.B}拍 / C误切 ${tc.C}拍`
        : (trajData === null ? " · 正在加载每拍拟合曲线..." : "");

      return `
        <svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; display:block;">
          ${sessionBandsHtml}
          ${gridLines}
          ${rawPathsHtml}
          ${refPathsHtml}
          ${dotsHtml}
          ${xTicks}
          <text x="${W / 2}" y="${H - 6}" fill="#8d9caf" font-size="10" text-anchor="middle">
            仅${sidePrefix}手 共 ${sessSpans.length} 场 ${items.length} 拍 · 黄虚线=各场中位${hasFittedForMetric ? " · 灰虚线=原始值 实线=PCHIP+SG拟合 · 绿=高置信 黄=插值 红×=误切" : ""}${tierInfo}
          </text>
        </svg>
      `;
    }

    const GLOBAL_AI_MODEL_OPTIONS = [
      // 1. DeepSeek 系列 (V4 Flash 极速 + R1 深度推理)
      { value: "deepseek::deepseek-flash", label: "DeepSeek · deepseek-flash (V4 Flash 极速/1M上下文/原生多模态)" },
      { value: "deepseek::deepseek-reasoner", label: "DeepSeek · deepseek-reasoner (R1 深度动力链思维链推演)" },
      { value: "deepseek::deepseek-chat", label: "DeepSeek · deepseek-chat (V3 通用基座)" },

      // 2. Google Gemini 系列 (对标/超越 V4 Flash 的下一代旗舰)
      { value: "gemini::gemini-2.5-flash", label: "Google · Gemini 2.5 Flash (全新 Flash Thinking / 1M上下文+动态推理)" },
      { value: "gemini::gemini-2.5-pro", label: "Google · Gemini 2.5 Pro (顶尖重型推理多模态旗舰)" },
      { value: "gemini::gemini-2.0-flash", label: "Google · Gemini 2.0 Flash (极速成熟高并发)" },

      // 3. OpenAI 系列 (极速思维链推理 + 最新全能旗舰)
      { value: "openai::o3-mini", label: "OpenAI · o3-mini (轻快级思维链推理之王 / 运动力学因果链极致)" },
      { value: "openai::gpt-4.5-preview", label: "OpenAI · GPT-4.5 (超大规模直觉感知旗舰)" },
      { value: "openai::gpt-4.1", label: "OpenAI · GPT-4.1 (新一代代码与逻辑全能)" },
      { value: "openai::o1", label: "OpenAI · o1 (深度物理因果推演)" },
      { value: "openai::gpt-4o", label: "OpenAI · GPT-4o (经典全能多模态主力)" },
      { value: "openai::gpt-4o-mini", label: "OpenAI · GPT-4o-mini (轻量经济)" },

      // 4. Anthropic Claude (顶尖教练处方设计)
      { value: "openai::claude-3-7-sonnet", label: "Claude · 3.7 Sonnet (顶尖教练逻辑与分期处方 / 需中转)" },
    ];

    function onChangeGlobalAiModel(val) {
      if (val) selectedGlobalAiChoice = val;
    }

    async function generateGlobalAiReview() {
      const btn = document.getElementById("gen-global-ai-btn");
      const outEl = document.getElementById("global-ai-review-body");
      const playerSel = document.getElementById("compare-player-select");
      const daysSel = document.getElementById("compare-days-select");
      const modelSel = document.getElementById("global-ai-model-select");
      if (!btn || !outEl || !playerSel) return;
      if (modelSel?.value) selectedGlobalAiChoice = modelSel.value;
      const [provider, model] = String(selectedGlobalAiChoice || "deepseek::deepseek-chat").split("::");
      const providerLabelMap = {
        openai: "OpenAI",
        gemini: "Google Gemini",
        deepseek: "DeepSeek",
      };
      const providerName = `${providerLabelMap[provider] || provider} (${model})`;
      const sideKey = activeStrokeSide();
      const sideCn = sideKey === "backhand" ? "反手" : "正手";
      btn.disabled = true;
      btn.textContent = `⏳ ${providerName} 正在分析全部${sideCn}数据...`;
      outEl.style.color = "var(--muted)";
      outEl.textContent = `正在调用 ${providerName} 提炼该学员多次${sideCn}训练的共性特征、跨场次前后矛盾及动作处方...`;
      try {
        const res = await api("/api/players/global-review", {
          method: "POST",
          body: JSON.stringify({
            player_id: playerSel.value || "default",
            days: daysSel?.value || "all",
            stroke_side: sideKey,
            dedup: dedupEnabled,
            provider: provider || "deepseek",
            model: model || "",
          }),
        });
        const data = await res.json();
        const sideDataRef = sideKey === "backhand" ? (globalProfileData?.backhand) : (globalProfileData?.forehand);
        if (sideDataRef) sideDataRef.latest_review = data;

        const vpaths = Array.isArray(data.video_paths) && data.video_paths.length
          ? data.video_paths.map(p => escapeHtml(p.split("/").pop())).join(", ")
          : `${(data.session_ids || []).length} 场训练会话`;
        const revId = data.review_id ? ` · 存档ID: <code>${escapeHtml(data.review_id)}</code>` : "";

        outEl.style.color = "var(--text)";
        outEl.innerHTML = `
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; font-size:11px; color:var(--muted); margin-bottom:8px; padding-bottom:6px; border-bottom:1px dashed var(--line);">
            <div>
              分析范围：${sideCn} · 引擎：<code>${escapeHtml(data.provider || provider)}</code> · 模型：<code>${escapeHtml(data.model || model || "")}</code>${revId}
            </div>
            <div>
              💾 已持久化入库 · 关联视频：<strong style="color:var(--text);">${vpaths}</strong> · 生成时间：${escapeHtml(data.generated_at || "")}
            </div>
          </div>
          <div style="white-space:pre-wrap; line-height:1.68; font-size:13px; color:var(--text);">${escapeHtml(data.review_text || "")}</div>
        `;
      } catch (err) {
        outEl.style.color = "#f87171";
        outEl.textContent = `生成总评失败：${err.message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = `🎾 生成主教练人话评语与处方`;
      }
    }

    function renderSingleStrokeDashboard(sideKey) {
      const body = document.getElementById("player-compare-body");
      if (!body || !globalProfileData) return;
      const isFh = sideKey !== "backhand";
      const sideData = isFh ? (globalProfileData.forehand || {}) : (globalProfileData.backhand || {});
      const sideCn = isFh ? "正手 (Forehand)" : "反手 (Backhand)";
      const sideShort = isFh ? "正手" : "反手";
      const accent = isFh ? "#38bdf8" : "#4ade80";

      if (!sideData.total_swings) {
        body.innerHTML = `
          <div style="padding:36px 16px; text-align:center;">
            <strong style="display:block; font-size:15px; color:var(--text); margin-bottom:8px;">
              「${escapeHtml(globalProfileData.player?.display_name || "该学员")}」在当前范围内暂无${sideCn}实战记录
            </strong>
            <span style="display:block; font-size:12px; color:var(--muted); margin-bottom:14px;">
              可切换另一侧查看，或点击「📂 批量归属会话」将挂在「默认球员」下的历史训练批量归入该学员。
            </span>
            <button type="button" class="button primary" onclick="openBatchAssignModal()">📂 批量归属历史会话</button>
          </div>
        `;
        return;
      }

      if (globalSlots.length !== 4 || globalSlots[0]?.side !== sideKey) {
        globalSlots = buildDefaultGlobalSlots(sideKey, globalGridPreset === "custom_2x2" ? "conflict_4" : globalGridPreset);
      }

      const commonalities = sideData.commonalities || [];
      const contradictions = sideData.contradictions || [];
      const repSpd = sideData.metrics?.racket_max_speed;
      const dd = globalProfileData.dedup || {};
      const dupCount = Number(dd.duplicate_session_count || 0);
      const dedupBannerHtml = dupCount > 0 ? `
        <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; padding:9px 12px; margin-bottom:12px; border-radius:9px; background:${dd.enabled ? 'rgba(56,189,248,0.08)' : 'rgba(251,191,36,0.08)'}; border:1px solid ${dd.enabled ? 'rgba(56,189,248,0.28)' : 'rgba(251,191,36,0.3)'}; font-size:12px;">
          <span style="color:var(--text);">
            ${dd.enabled
              ? `🔀 <strong>已按原视频自动去重</strong>：原始 ${dd.raw_session_count} 场分析包含 ${dd.duplicate_group_count} 组同视频重跑，已优先保留双视角/最新代表场 <strong>${dd.unique_session_count} 场</strong>（隐藏 ${dupCount} 场重复记录），防止同视频拉偏共性均值。`
              : `⚠️ <strong>已关闭视频去重</strong>：当前合并了全部 ${dd.raw_session_count} 场原始记录（含 ${dupCount} 场同源视频重跑）。`}
          </span>
          <button type="button" class="button" style="min-height:28px; padding:0 10px; font-size:12px;" onclick="openDedupModal()">🧹 查看去重明细 / 归档清理</button>
        </div>
      ` : "";

      const kpiHtml = `
        ${dedupBannerHtml}
        <div class="kpi-strip">
          <div class="kpi-card"><span>${sideShort}训练场次</span><strong style="color:${accent};">${sideData.session_count || 0} 场</strong><small>${sideCn}</small></div>
          <div class="kpi-card"><span>${sideShort}实战拍数</span><strong>${sideData.total_swings || 0} 拍</strong><small>双视角 ${sideData.dual_view_swings || 0} · 单视角 ${sideData.single_view_swings || 0}</small></div>
          <div class="kpi-card"><span>动作定型度</span><strong style="color:${accent};">${sideData.consistency_index != null ? `${sideData.consistency_index}%` : "—"}</strong><small>越高代表多次动作越统一</small></div>
          <div class="kpi-card"><span>代表拍速 (中位)</span><strong>${repSpd ? `${Math.round(repSpd.median)} px/s` : "—"}</strong><small>${repSpd ? `P25~P75: ${Math.round(repSpd.p25)}~${Math.round(repSpd.p75)}` : "暂无有效拍速"}</small></div>
          <div class="kpi-card"><span>共性 / 矛盾数</span><strong>${commonalities.length} / <span style="color:${contradictions.length ? '#fbbf24' : '#4ade80'};">${contradictions.length}</span></strong><small>跨场稳定性自动体检</small></div>
        </div>
      `;

      // Commonality & Contradiction dual panel
      const commonHtml = commonalities.length ? commonalities.map(c => {
        const isFlaw = c.type === "persistent_flaw";
        const tag = isFlaw
          ? `<span style="font-size:10px; padding:1px 6px; border-radius:4px; background:rgba(248,113,113,0.16); color:#f87171; border:1px solid rgba(248,113,113,0.3);">高频共性毛病</span>`
          : `<span style="font-size:10px; padding:1px 6px; border-radius:4px; background:rgba(74,222,128,0.14); color:#4ade80; border:1px solid rgba(74,222,128,0.3);">动作稳定习惯</span>`;
        return `
          <div style="padding:7px 9px; border-radius:8px; background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.06); margin-bottom:7px;">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:3px;">
              <strong style="font-size:12.5px; color:var(--text);">${escapeHtml(c.label)}</strong>
              ${tag}
            </div>
            <div style="font-size:12px; color:var(--muted); line-height:1.45;">${escapeHtml(c.summary)}</div>
          </div>
        `;
      }).join("") : `<div style="font-size:12px; color:var(--muted); padding:8px 0;">样本场次较少或波动较大，暂未检测到稳定跨场共性。</div>`;

      const contraHtml = contradictions.length ? contradictions.map(c => {
        const isHigh = c.severity === "high";
        const tagColor = isHigh ? "#f87171" : "#fbbf24";
        const tagText = c.type === "trend_reversal" ? "趋势反转" : (c.type === "view_heterogeneity" ? "视角口径" : (isHigh ? "显著前后矛盾" : "跨场波动"));
        const jumpBtn = (c.type === "session_drift" && c.min_session?.session_id && c.max_session?.session_id)
          ? `<button type="button" class="button" style="min-height:24px; padding:0 8px; font-size:11px; margin-top:5px;" onclick="jumpGridToConflictSessions('${escapeHtml(c.min_session.session_id)}', '${escapeHtml(c.max_session.session_id)}', '${escapeHtml(c.key)}', '${escapeHtml(c.label)}')">🎬 在 1×4 视频对比极差场次</button>`
          : "";
        return `
          <div style="padding:7px 9px; border-radius:8px; background:rgba(251,191,36,0.04); border:1px solid rgba(251,191,36,0.18); margin-bottom:7px;">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:3px;">
              <strong style="font-size:12.5px; color:var(--text);">${escapeHtml(c.label)}</strong>
              <span style="font-size:10px; padding:1px 6px; border-radius:4px; background:rgba(0,0,0,0.3); color:${tagColor}; border:1px solid ${tagColor}44;">${tagText}</span>
            </div>
            <div style="font-size:12px; color:var(--muted); line-height:1.45;">${escapeHtml(c.summary)}</div>
            ${jumpBtn}
          </div>
        `;
      }).join("") : `<div style="font-size:12px; color:#4ade80; padding:8px 0;">多次${sideShort}训练高度一致，未检出明显前后矛盾或跨场漂移 🎉</div>`;

      // Full metric table for THIS side only
      const metricRowsHtml = Object.entries(sideData.metrics || {}).map(([key, m]) => {
        const scopeTag = m.view_scope === "dual_view_only"
          ? `<span style="font-size:10px; padding:1px 5px; border-radius:4px; background:rgba(168,85,247,0.16); color:#c084fc; margin-left:4px;">仅双视角</span>`
          : "";
        const cScore = m.consistency_score != null
          ? `<span style="color:${m.consistency_score >= 60 ? '#4ade80' : (m.consistency_score < 45 ? '#f87171' : '#fbbf24')}; font-weight:600;">${m.consistency_score}%</span>`
          : "—";
        let trendCell = `<span style="color:var(--muted);">—</span>`;
        if (m.trend_delta != null) {
          const improved = m.higher_better ? m.trend_delta > 0 : m.trend_delta < 0;
          const tc = m.trend_delta === 0 ? "var(--muted)" : (improved ? "#4ade80" : "#f87171");
          const sign = m.trend_delta > 0 ? "+" : "";
          trendCell = `<span style="color:${tc}; font-weight:600;">${sign}${m.trend_delta} ${escapeHtml(m.unit)}</span> <small style="color:var(--muted);">(${m.early_median}→${m.recent_median})</small>`;
        }
        return `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:6px 8px;"><strong>${escapeHtml(m.label)}</strong>${scopeTag}</td>
            <td style="padding:6px 8px; color:${accent}; font-weight:600;">${m.median} ${escapeHtml(m.unit)} <small style="color:var(--muted); font-weight:400;">±${m.iqr} (n=${m.count})</small></td>
            <td style="padding:6px 8px; color:var(--muted);">${m.min} ~ ${m.max} ${escapeHtml(m.unit)}</td>
            <td style="padding:6px 8px;">${cScore}</td>
            <td style="padding:6px 8px;">${trendCell}</td>
          </tr>
        `;
      }).join("");

      const trendBtnsHtml = TREND_METRIC_OPTIONS.map(opt => (
        `<button type="button" onclick="setGlobalTrendMetric('${opt.key}')" style="height:26px; padding:0 9px; font-size:11px; border-radius:6px; border:1px solid ${globalTrendMetricKey === opt.key ? 'rgba(56,189,248,0.55)' : 'var(--line)'}; background:${globalTrendMetricKey === opt.key ? 'rgba(56,189,248,0.18)' : 'rgba(255,255,255,0.03)'}; color:${globalTrendMetricKey === opt.key ? '#fff' : 'var(--muted)'}; cursor:pointer;">${escapeHtml(opt.label)}</button>`
      )).join("");

      // Personal attribution (recomputed on PCHIP+SG fitted Tier A+B swings when available)
      const attr = computeFittedPersonalAttribution(sideKey, sideData);
      const attrBadge = attr.fitted
        ? `<span style="font-size:10px; padding:1px 6px; border-radius:4px; background:rgba(74,222,128,0.16); color:#4ade80; margin-left:6px; font-weight:500;">已基于拟合数据重算</span>`
        : "";
      const attrHtml = attr.available ? `
        <div style="flex:1; min-width:300px; padding:12px; background:rgba(56,189,248,0.05); border:1px solid rgba(56,189,248,0.20); border-radius:10px;">
          <strong style="font-size:13px; color:${accent}; display:block; margin-bottom:6px;">🔑 ${sideShort}个人好球 vs 弱球归因 (有效 ${attr.sample_count} 拍)${attrBadge}</strong>
          <div style="font-size:12px; line-height:1.55; color:var(--text); margin-bottom:8px;">${escapeHtml(attr.insight || "")}</div>
          <div style="display:flex; flex-wrap:wrap; gap:6px;">
            ${(attr.factors || []).slice(0, 5).map(f => `
              <span style="font-size:11px; padding:3px 8px; background:rgba(255,255,255,0.05); border-radius:5px;">
                <strong>${escapeHtml(f.label)}</strong>: 好球 ${f.top_median}${escapeHtml(f.unit)} vs 弱球 ${f.bottom_median}${escapeHtml(f.unit)}
                <span style="color:#4ade80;">(效应 ${f.effect_ratio}x)</span>
              </span>
            `).join("")}
          </div>
        </div>
      ` : "";

      // Separate real biomechanical technique flaws from camera/capture diagnostic notices
      const bioFlawInfo = buildBiomechanicalFlaws(sideKey, sideData);
      const captureMsgs = new Set(["动作证据不足需复核", "减少球拍遮挡", "触球候选需确认"]);
      const rawWarns = sideData.coach_warnings || [];
      const captureNotes = rawWarns.filter(w => captureMsgs.has(String(w.message || "").trim()) || /遮挡|复核|候选/.test(String(w.message || "")));
      const warnsHtml = `
        <div style="flex:1; min-width:300px; padding:12px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:10px;">
          <strong style="font-size:13px; color:var(--text); display:block; margin-bottom:6px;">🎯 ${sideShort}动作技术瑕疵体检 (基于有效 ${bioFlawInfo.total} 拍 · 早期→近期)</strong>
          ${bioFlawInfo.flaws.length ? bioFlawInfo.flaws.map(w => {
            const trendStr = (w.early_rate_pct != null && w.recent_rate_pct != null)
              ? (() => {
                  const d = w.delta_pct;
                  const c = d < 0 ? "#4ade80" : (d > 0 ? "#f87171" : "var(--muted)");
                  const s = d > 0 ? "+" : "";
                  return `<span style="color:var(--muted); margin-left:6px;">(${w.early_rate_pct}%→${w.recent_rate_pct}% <strong style="color:${c};">${s}${d}%</strong>)</span>`;
                })()
              : "";
            return `
              <div style="display:flex; justify-content:space-between; gap:8px; font-size:12px; padding:4px 0; border-bottom:1px dashed rgba(255,255,255,0.06);">
                <span>${escapeHtml(w.label)}</span>
                <span style="white-space:nowrap;"><strong>${w.rate_pct}%</strong> (${w.count}/${bioFlawInfo.total})${trendStr}</span>
              </div>
            `;
          }).join("") : `<div style="font-size:12px; color:#4ade80; padding:4px 0;">未检出高频技术瑕疵 🎉</div>`}
          ${captureNotes.length ? `
            <div style="margin-top:8px; padding-top:7px; border-top:1px solid rgba(255,255,255,0.08);">
              <div style="font-size:11px; color:var(--muted); margin-bottom:4px;">📷 拍摄/识别质量提示（非动作技术瑕疵）</div>
              <div style="display:flex; flex-wrap:wrap; gap:6px;">
                ${captureNotes.map(cn => `
                  <span style="font-size:11px; padding:2px 7px; border-radius:4px; background:rgba(255,255,255,0.04); color:var(--muted);">
                    ${escapeHtml(cn.message)} ${cn.rate_pct}% (${cn.count}/${sideData.total_swings})
                  </span>
                `).join("")}
              </div>
            </div>
          ` : ""}
        </div>
      `;

      // Per-session timeline table (prefer fitted Tier A+B medians when available)
      const trajMapForTables = buildTrajMapForSide(sideKey);
      const sideSwingsAll = sideData.swings || [];
      const sidePrefixTbl = isFh ? "正" : "反";
      const sessionRowsHtml = (sideData.trend_series || []).slice().reverse().map(pt => {
        const sessSwings = sideSwingsAll.filter(s => s.session_id === pt.session_id);
        const validSessVals = [];
        let tierCCnt = 0;
        sessSwings.forEach(s => {
          const dg = trajMapForTables.get(`${s.session_id}#${s.event_id}`) || null;
          if (dg && dg.tier === "C") { tierCCnt += 1; return; }
          validSessVals.push(getSwingRadarValues(s, dg));
        });
        const fitMed = (k, fallback) => {
          const m = medianOfNums(validSessVals.map(v => v[k]));
          return m != null ? Number(m.toFixed(1)) : fallback;
        };
        const turnV = fitMed("shoulder_turn_change", pt.shoulder_turn_change);
        const sepV = fitMed("hip_shoulder_separation", pt.hip_shoulder_separation);
        const armV = fitMed("arm_extension", pt.arm_extension);
        const kneeV = fitMed("preparation_knee_flexion", pt.preparation_knee_flexion);
        const spdV = fitMed("racket_max_speed", pt.racket_max_speed);
        const countNote = tierCCnt > 0 ? `${validSessVals.length}/${pt.count} 拍` : `${pt.count} 拍`;
        return `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:6px 8px; white-space:nowrap;">${escapeHtml(pt.display_time)}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${pt.view_mode === "dual_view" ? "双视角" : "单视角"}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${countNote}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${turnV != null ? `${turnV}°` : "—"}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${sepV != null ? `${sepV}°` : "—"}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${armV != null ? `${armV}°` : "—"}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${kneeV != null ? `${kneeV}°` : "—"}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${spdV != null ? `${Math.round(spdV)} px/s` : "—"}</td>
            <td style="padding:6px 8px; white-space:nowrap;">${pt.consistency_index != null ? `${pt.consistency_index}%` : "—"}</td>
            <td style="padding:6px 8px; white-space:nowrap;"><button type="button" class="button" style="min-height:24px;padding:0 8px;font-size:11px;white-space:nowrap;" onclick="openSessionInConsole('${escapeHtml(pt.session_id)}')">打开会话 ↗</button></td>
          </tr>
        `;
      }).join("");

      const perSwingTableRowsHtml = sideSwingsAll.map((sw, idx) => {
        const key = `${sw.session_id}#${sw.event_id}`;
        const dg = trajMapForTables.get(key) || null;
        const vals = getSwingRadarValues(sw, dg);
        const tier = dg?.tier || "A";
        const tierColor = tier === "A" ? "#4ade80" : (tier === "B" ? "#fbbf24" : "#f87171");
        const tierTxt = tier === "A" ? "高置信" : (tier === "B" ? "插值" : "误切");
        return `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05); ${tier === 'C' ? 'opacity:0.55;' : ''}">
            <td style="padding:5px 8px; white-space:nowrap; font-weight:600; color:${accent};">${sidePrefixTbl}${idx + 1} <small style="color:var(--muted); font-weight:400;">#${sw.event_id}</small></td>
            <td style="padding:5px 8px; white-space:nowrap;"><span style="font-size:10px; padding:1px 6px; border-radius:4px; background:${tierColor}22; color:${tierColor};">T${tier} ${tierTxt}</span></td>
            <td style="padding:5px 8px; white-space:nowrap;">${vals.shoulder_turn_change != null ? `${vals.shoulder_turn_change.toFixed(1)}°` : "—"}</td>
            <td style="padding:5px 8px; white-space:nowrap;">${vals.hip_shoulder_separation != null ? `${vals.hip_shoulder_separation.toFixed(1)}°` : "—"}</td>
            <td style="padding:5px 8px; white-space:nowrap;">${vals.arm_extension != null ? `${vals.arm_extension.toFixed(1)}°` : "—"}</td>
            <td style="padding:5px 8px; white-space:nowrap;">${vals.preparation_knee_flexion != null ? `${vals.preparation_knee_flexion.toFixed(1)}°` : "—"}</td>
            <td style="padding:5px 8px; white-space:nowrap;">${vals.stance != null ? `${Math.round(vals.stance)}°` : "—"}</td>
            <td style="padding:5px 8px; white-space:nowrap;">${vals.takeback_depth != null ? vals.takeback_depth.toFixed(2) : "—"}</td>
            <td style="padding:5px 8px; white-space:nowrap;">${vals.racket_max_speed != null ? `${Math.round(vals.racket_max_speed)} px/s` : "—"}</td>
            <td style="padding:5px 8px; white-space:nowrap; display:flex; gap:4px;">
              <button type="button" class="button" style="min-height:22px; padding:0 7px; font-size:11px; white-space:nowrap;" onclick="openTrajectoryModal('${escapeHtml(sw.session_id)}', ${sw.event_id})">📈 曲线</button>
              <button type="button" class="button" style="min-height:22px; padding:0 7px; font-size:11px; white-space:nowrap;" onclick="loadSwingIntoGridSlot('${escapeHtml(sw.session_id)}', ${sw.event_id})">🎬 视频</button>
            </td>
          </tr>
        `;
      }).join("");

      const latestRev = sideData.latest_review;
      let reviewBodyHtml = "";
      if (latestRev && latestRev.review_text) {
        const vpaths = Array.isArray(latestRev.video_paths) && latestRev.video_paths.length
          ? latestRev.video_paths.map(p => escapeHtml(p.split("/").pop())).join(", ")
          : `${(latestRev.session_ids || []).length} 场训练会话`;
        const revId = latestRev.review_id ? ` · 存档ID: <code>${escapeHtml(latestRev.review_id)}</code>` : "";
        reviewBodyHtml = `
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; font-size:11px; color:var(--muted); margin-bottom:8px; padding-bottom:6px; border-bottom:1px dashed var(--line);">
            <div>
              分析范围：${sideShort} · 引擎：<code>${escapeHtml(latestRev.provider || "")}</code> · 模型：<code>${escapeHtml(latestRev.model || "")}</code>${revId}
            </div>
            <div>
              💾 数据库历史存档 · 关联视频：<strong style="color:var(--text);">${vpaths}</strong> · 存档时间：${escapeHtml(latestRev.created_at || "")}
            </div>
          </div>
          <div style="white-space:pre-wrap; line-height:1.68; font-size:13px; color:var(--text);">${escapeHtml(latestRev.review_text || "")}</div>
        `;
      } else {
        reviewBodyHtml = `
          基于该学员全部${sideShort}清洗拟合数据，模拟 15 年执教经验职业教练，输出学员看得懂的动作画像、好坏球具体拍号诊疗、球场默念口令与课后专项练习。（生成的评语将自动持久化入库，关联学员、时间及视频证据包）
        `;
      }

      body.innerHTML = `
        ${kpiHtml}

        <!-- Commonalities vs Contradictions Dual Card -->
        <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:14px;">
          <div style="flex:1; min-width:300px; padding:12px; background:rgba(74,222,128,0.03); border:1px solid rgba(74,222,128,0.22); border-radius:10px;">
            <div style="font-size:13px; font-weight:700; color:#4ade80; margin-bottom:8px;">✅ 多次${sideShort}训练共性（稳定习惯与反复毛病）</div>
            ${commonHtml}
          </div>
          <div style="flex:1; min-width:300px; padding:12px; background:rgba(251,191,36,0.03); border:1px solid rgba(251,191,36,0.22); border-radius:10px;">
            <div style="font-size:13px; font-weight:700; color:#fbbf24; margin-bottom:8px;">⚡ 跨场次前后矛盾与波动冲突（不一致处）</div>
            ${contraHtml}
          </div>
        </div>

        <!-- 1x4 Four-Video Full-Body Synchronized Strip (Single Stroke Only, Max 4 Videos) -->
        <div id="grid-2x2-section" style="padding:12px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:12px; margin-bottom:14px;">
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:10px;">
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:6px;">
              <strong style="font-size:13px; color:${accent}; margin-right:4px;">🎬 ${sideShort} 1×4 四路全身同步视频（含步伐）</strong>
              <button type="button" class="mode-pill ${globalGridPreset === 'conflict_4' ? 'active' : ''}" onclick="setGlobalGridPreset('conflict_4')">⚡ 矛盾极差四场</button>
              <button type="button" class="mode-pill ${globalGridPreset === 'timeline_4' ? 'active' : ''}" onclick="setGlobalGridPreset('timeline_4')">📈 全周期演变</button>
              <button type="button" class="mode-pill ${globalGridPreset === 'best_worst_4' ? 'active' : ''}" onclick="setGlobalGridPreset('best_worst_4')">🔥 好球 vs 弱球</button>
              <button type="button" class="mode-pill ${globalGridPreset === 'custom_2x2' ? 'active' : ''}" onclick="setGlobalGridPreset('custom_2x2')">🎛️ 自由选择</button>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
              <div style="display:inline-flex; gap:2px; background:rgba(0,0,0,0.28); padding:2px; border-radius:6px; border:1px solid rgba(255,255,255,0.08);">
                <button type="button" onclick="setGlobalCropMode('front_focus')" style="height:26px; padding:0 9px; font-size:11px; border:none; border-radius:5px; cursor:pointer; background:${compareCropMode === 'front_focus' ? 'var(--blue, #38bdf8)' : 'transparent'}; color:${compareCropMode === 'front_focus' ? '#000' : 'var(--text)'}; font-weight:${compareCropMode === 'front_focus' ? '600' : '400'};">🎯 正面全身</button>
                <button type="button" onclick="setGlobalCropMode('back_focus')" style="height:26px; padding:0 9px; font-size:11px; border:none; border-radius:5px; cursor:pointer; background:${compareCropMode === 'back_focus' ? 'var(--blue, #38bdf8)' : 'transparent'}; color:${compareCropMode === 'back_focus' ? '#000' : 'var(--text)'}; font-weight:${compareCropMode === 'back_focus' ? '600' : '400'};">🪞 镜面全身</button>
                <button type="button" onclick="setGlobalCropMode('raw')" style="height:26px; padding:0 9px; font-size:11px; border:none; border-radius:5px; cursor:pointer; background:${compareCropMode === 'raw' ? 'var(--blue, #38bdf8)' : 'transparent'}; color:${compareCropMode === 'raw' ? '#000' : 'var(--text)'}; font-weight:${compareCropMode === 'raw' ? '600' : '400'};">🖼️ 原画幅</button>
              </div>
              <button type="button" class="button" style="min-height:30px; padding:0 10px; font-size:12px;" onclick="syncGridSeekContact()">💥 四路对齐触球帧</button>
              <button type="button" class="button" style="min-height:30px; padding:0 10px; font-size:12px;" onclick="syncGridPlayClips(0.25)">🎬 0.25x 同步慢放</button>
              <button type="button" class="button" style="min-height:30px; padding:0 10px; font-size:12px;" onclick="syncGridPlayClips(0.5)">0.5x</button>
              <button type="button" class="button" style="min-height:30px; padding:0 10px; font-size:12px;" onclick="syncGridPauseClips()">⏸ 暂停</button>
              <button type="button" class="button" style="min-height:30px; padding:0 8px; font-size:12px;" onclick="syncGridStepFrames(-1)" title="四路同步后退 1 帧">⏮ -1帧</button>
              <button type="button" class="button" style="min-height:30px; padding:0 8px; font-size:12px;" onclick="syncGridStepFrames(1)" title="四路同步前进 1 帧">+1帧 ⏭</button>
            </div>
          </div>
          <div class="compare-grid-1x4">
            ${globalSlots.slice(0, 4).map((slot, idx) => renderGridSlotCardHtml(slot, idx)).join("")}
          </div>
        </div>

        <!-- Single-Side 8-Axis Radar + Full Dimension Table -->
        <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-bottom:14px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:10px; padding:12px;">
          <div style="flex:0 0 308px; margin:0 auto;">
            ${buildSingleSideRadarSvg(sideKey, sideData, accent)}
          </div>
          <div style="flex:1; min-width:340px; overflow:auto;">
            <table style="width:100%; font-size:12.5px; border-collapse:collapse; text-align:left; color:var(--text);">
              <thead>
                <tr style="border-bottom:1px solid var(--line); color:var(--muted);">
                  <th style="padding:6px 8px;">${sideShort}姿态维度</th>
                  <th style="padding:6px 8px;">全局中位 ± IQR</th>
                  <th style="padding:6px 8px;">样本区间 (Min~Max)</th>
                  <th style="padding:6px 8px;">定型度</th>
                  <th style="padding:6px 8px;">早期 → 近期变化</th>
                </tr>
              </thead>
              <tbody>
                ${metricRowsHtml}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Per-Swing 8-Axis Radar Gallery (Every Single Swing) -->
        ${buildPerSwingRadarGalleryHtml(sideKey, sideData, accent)}

        <!-- Single-Side Chronological Trend & Per-Swing Fitting Chart -->
        <div style="padding:12px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:10px; margin-bottom:14px;">
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px;">
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
              <strong style="font-size:13px; color:var(--text);">📈 ${sideShort}训练波动与逐拍轨迹拟合观测</strong>
              <span style="font-size:10.5px; padding:2px 7px; border-radius:4px; background:${isFh ? 'rgba(56,189,248,0.14)' : 'rgba(74,222,128,0.14)'}; color:${accent}; border:1px solid ${isFh ? 'rgba(56,189,248,0.32)' : 'rgba(74,222,128,0.32)'};">仅统计${sideShort} · 已隔离${isFh ? '反手' : '正手'}</span>
              ${(() => {
                const effMode = globalTrendViewMode === "auto"
                  ? ((sideData.trend_series || []).length <= 1 ? "per_swing" : "per_session")
                  : globalTrendViewMode;
                return `
                  <div class="mode-pill-group" style="padding:2px;">
                    <button type="button" class="mode-pill ${effMode === 'per_swing' ? 'active' : ''}" style="height:24px; padding:0 8px; font-size:11px;" onclick="setGlobalTrendViewMode('per_swing')">🔬 按逐拍拟合 (分场次区隔)</button>
                    <button type="button" class="mode-pill ${effMode === 'per_session' ? 'active' : ''}" style="height:24px; padding:0 8px; font-size:11px;" onclick="setGlobalTrendViewMode('per_session')">📅 按场次趋势</button>
                  </div>
                `;
              })()}
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:5px;">
              ${trendBtnsHtml}
            </div>
          </div>
          ${(() => {
            const effMode = globalTrendViewMode === "auto"
              ? ((sideData.trend_series || []).length <= 1 ? "per_swing" : "per_session")
              : globalTrendViewMode;
            return effMode === "per_swing"
              ? buildPerSwingFittedChartSvg(sideKey, sideData, globalTrendMetricKey, accent)
              : buildSingleSideTrendChartSvg(sideKey, sideData, globalTrendMetricKey, accent);
          })()}
        </div>

        <!-- Attribution + Warnings -->
        <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:14px;">
          ${attrHtml}
          ${warnsHtml}
        </div>

        <!-- Per-Session Sequence Table -->
        <details open style="padding:12px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:10px; margin-bottom:14px;">
          <summary style="cursor:pointer; color:var(--text); font-size:13px; font-weight:600;">📋 ${sideShort}各次训练逐场对比明细 (${sideData.session_count || 0} 场 · 优先展示拟合中位)</summary>
          <div style="overflow:auto; margin-top:8px;">
            <table style="width:100%; font-size:12px; border-collapse:collapse; text-align:left; color:var(--text);">
              <thead>
                <tr style="border-bottom:1px solid var(--line); color:var(--muted);">
                  <th style="padding:6px 8px; white-space:nowrap;">训练时间</th>
                  <th style="padding:6px 8px; white-space:nowrap;">视角</th>
                  <th style="padding:6px 8px; white-space:nowrap;">有效/${sideShort}拍数</th>
                  <th style="padding:6px 8px; white-space:nowrap;">转肩变化</th>
                  <th style="padding:6px 8px; white-space:nowrap;">肩髋分离</th>
                  <th style="padding:6px 8px; white-space:nowrap;">肘伸展角</th>
                  <th style="padding:6px 8px; white-space:nowrap;">准备屈膝</th>
                  <th style="padding:6px 8px; white-space:nowrap;">拍框峰值</th>
                  <th style="padding:6px 8px; white-space:nowrap;">定型度</th>
                  <th style="padding:6px 8px; white-space:nowrap;">操作</th>
                </tr>
              </thead>
              <tbody>
                ${sessionRowsHtml}
              </tbody>
            </table>
          </div>
        </details>

        <!-- Per-Swing Complete Detail Table -->
        <details style="padding:12px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:10px; margin-bottom:14px;">
          <summary style="cursor:pointer; color:var(--text); font-size:13px; font-weight:600;">🔎 ${sideShort}逐拍完整数据明细 (共 ${sideSwingsAll.length} 拍 · 点击可看拟合曲线或载入 1×4 视频)</summary>
          <div style="overflow:auto; margin-top:8px;">
            <table style="width:100%; font-size:12px; border-collapse:collapse; text-align:left; color:var(--text);">
              <thead>
                <tr style="border-bottom:1px solid var(--line); color:var(--muted);">
                  <th style="padding:6px 8px; white-space:nowrap;">拍序</th>
                  <th style="padding:6px 8px; white-space:nowrap;">质量分级</th>
                  <th style="padding:6px 8px; white-space:nowrap;">转肩变化</th>
                  <th style="padding:6px 8px; white-space:nowrap;">肩髋分离</th>
                  <th style="padding:6px 8px; white-space:nowrap;">触球肘展</th>
                  <th style="padding:6px 8px; white-space:nowrap;">准备屈膝</th>
                  <th style="padding:6px 8px; white-space:nowrap;">足部站位</th>
                  <th style="padding:6px 8px; white-space:nowrap;">镜面引拍</th>
                  <th style="padding:6px 8px; white-space:nowrap;">拟合拍速</th>
                  <th style="padding:6px 8px; white-space:nowrap;">操作</th>
                </tr>
              </thead>
              <tbody>
                ${perSwingTableRowsHtml}
              </tbody>
            </table>
          </div>
        </details>

        <!-- One-Click AI Single-Side Review (Coach Language) -->
        <div style="padding:12px 14px; background:rgba(255,255,255,0.02); border:1px solid var(--line); border-radius:10px;">
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px;">
            <strong style="font-size:13px; color:var(--text);">🎾 AI 主教练${sideShort}实战评语与动作处方（学员听得懂的人话评析）</strong>
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
              <select id="global-ai-model-select" class="compare-select" style="height:32px; font-size:12px;" onchange="onChangeGlobalAiModel(this.value)" aria-label="选择 AI 模型">
                ${GLOBAL_AI_MODEL_OPTIONS.map(opt => `<option value="${escapeHtml(opt.value)}" ${opt.value === selectedGlobalAiChoice ? "selected" : ""}>${escapeHtml(opt.label)}</option>`).join("")}
              </select>
              <button id="gen-global-ai-btn" type="button" class="button primary" style="min-height:32px; padding:0 12px; font-size:12px;" onclick="generateGlobalAiReview()">🎾 生成主教练人话评语与处方</button>
            </div>
          </div>
          <div id="global-ai-review-body" style="font-size:12px; color:var(--muted); line-height:1.6;">
            ${reviewBodyHtml}
          </div>
        </div>
      `;
    }

    async function loadGlobalProfile() {
      const body = document.getElementById("player-compare-body");
      const playerSel = document.getElementById("compare-player-select");
      const daysSel = document.getElementById("compare-days-select");
      if (!body || !playerSel) return;
      const pid = playerSel.value || "default";
      const days = daysSel ? daysSel.value : "all";
      const sideKey = activeStrokeSide();
      body.textContent = `正在聚合该学员全部${sideKey === "backhand" ? "反手" : "正手"}训练数据...`;
      try {
        const qs = new URLSearchParams({ player_id: pid, dedup: dedupEnabled ? "1" : "0" });
        if (days && days !== "all") qs.set("days", days);
        const res = await api(`/api/players/global-profile?${qs.toString()}`);
        globalProfileData = await res.json();
        trajectorySummaryBySide = { forehand: null, backhand: null };
        selectedRadarSwingKey = "";
        updateModeTabCounts();
        globalSlots = buildDefaultGlobalSlots(sideKey, globalGridPreset === "custom_2x2" ? "conflict_4" : globalGridPreset);
        renderSingleStrokeDashboard(sideKey);
        loadTrajectorySummary(sideKey);
      } catch (err) {
        body.textContent = `加载全局训练数据失败：${err.message}`;
      }
    }

    /* ============================================================
     * Single-Session A/B Comparison Mode (Preserved)
     * ============================================================ */
    function buildDualRadarSvg(sessA, sessB) {
      const axes = [
        CORE_COMPARE_METRICS[0],
        CORE_COMPARE_METRICS[1],
        CORE_COMPARE_METRICS[3],
        CORE_COMPARE_METRICS[4],
        CORE_COMPARE_METRICS[5],
        CORE_COMPARE_METRICS[7],
      ];
      const cx = 115, cy = 115, rMax = 74;
      const toPt = (idx, ratio) => {
        const angle = -Math.PI / 2 + (idx * 2 * Math.PI) / axes.length;
        const r = Math.max(0.08, Math.min(1.0, ratio)) * rMax;
        return `${(cx + r * Math.cos(angle)).toFixed(1)},${(cy + r * Math.sin(angle)).toFixed(1)}`;
      };
      const gridRings = [0.33, 0.66, 1.0].map(level => (
        `<polygon points="${axes.map((_, i) => toPt(i, level)).join(" ")}" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="1"/>`
      )).join("");
      const axisLines = axes.map((ax, i) => {
        const angle = -Math.PI / 2 + (i * 2 * Math.PI) / axes.length;
        const x2 = cx + rMax * Math.cos(angle);
        const y2 = cy + rMax * Math.sin(angle);
        const lx = cx + (rMax + 24) * Math.cos(angle);
        const ly = cy + (rMax + 16) * Math.sin(angle);
        return `
          <line x1="${cx}" y1="${cy}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
          <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" fill="var(--muted)" font-size="10" text-anchor="middle" dominant-baseline="middle">${ax.label.slice(0, 4)}</text>
        `;
      }).join("");

      const polyFor = (sess) => {
        if (!sess) return "";
        return axes.map((ax, i) => {
          const val = sess.metrics?.[ax.key]?.median ?? 0;
          return toPt(i, val / ax.refMax);
        }).join(" ");
      };
      const ptsA = polyFor(sessA);
      const ptsB = polyFor(sessB);

      return `
        <svg viewBox="0 0 230 230" width="230" height="230" style="display:block; margin:0 auto;">
          ${gridRings}
          ${axisLines}
          ${ptsA ? `<polygon points="${ptsA}" fill="rgba(56,189,248,0.14)" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="4,3"/>` : ""}
          ${ptsB ? `<polygon points="${ptsB}" fill="rgba(74,222,128,0.20)" stroke="#4ade80" stroke-width="2.2"/>` : ""}
        </svg>
        <div style="display:flex; justify-content:center; gap:14px; font-size:11px; color:var(--muted); margin-top:4px;">
          <span><span style="display:inline-block;width:10px;height:2px;background:#38bdf8;vertical-align:middle;margin-right:4px;"></span>基准 A (虚线)</span>
          <span><span style="display:inline-block;width:10px;height:2px;background:#4ade80;vertical-align:middle;margin-right:4px;"></span>当前 B (实线)</span>
        </div>
      `;
    }

    function applyCompareMediaTransform(slotId, fallbackViewMode = "single_view") {
      const mediaEl = document.getElementById(`cmp-video-${slotId}`) || document.getElementById(`cmp-img-${slotId}`);
      const badgeEl = document.getElementById(`cmp-view-badge-${slotId}`);
      const panWrap = document.getElementById(`cmp-pan-wrap-${slotId}`);
      if (!mediaEl) return;

      const w = mediaEl.videoWidth || mediaEl.naturalWidth || 0;
      const h = mediaEl.videoHeight || mediaEl.naturalHeight || 0;
      let isDual = fallbackViewMode === "dual_view";
      if (w > 0 && h > 0) {
        isDual = (w / h) <= 1.58;
      }
      slotDetectedDualView[slotId] = isDual;
      const fx = compareFocusX[slotId] ?? 76;
      const tf = computeMediaTransformCss(isDual, compareCropMode, fx, "100%");
      mediaEl.style.cssText = tf.css;
      if (badgeEl) badgeEl.textContent = tf.badge;
      if (panWrap) panWrap.hidden = !tf.showPan;
    }

    function nudgeCompareFocus(slotId, delta) {
      const cur = compareFocusX[slotId] ?? 76;
      compareFocusX[slotId] = Math.max(15, Math.min(90, cur + delta));
      const sessId = slotId === "A" ? selectedCompareSessionA : selectedCompareSessionB;
      const sess = latestCompareSessions.find(s => s.session_id === sessId);
      applyCompareMediaTransform(slotId, sess?.view_mode || "single_view");
    }

    function setCompareCropMode(mode) {
      compareCropMode = mode;
      renderCompareDashboard();
    }

    function buildSwingMediaBoxHtml(sw, slotId, viewMode = "single_view") {
      if (!sw) return "";
      const startF = sw.start_frame || 0;
      const contactF = sw.contact_frame ?? startF;
      const offsetSec = Math.max(0, (contactF - startF) / 25.0).toFixed(2);
      const mediaInner = sw.clip_url ? `
        <video id="cmp-video-${slotId}" src="${sw.clip_url}" poster="${sw.impact_freeze_url || ''}" muted controls loop playsinline
               data-contact-sec="${offsetSec}"
               onloadedmetadata="applyCompareMediaTransform('${slotId}', '${viewMode}')"
               style="width:100%; height:100%; object-fit:contain; display:block;"></video>
      ` : (sw.impact_freeze_url ? `
        <img id="cmp-img-${slotId}" src="${sw.impact_freeze_url}" alt="触球定格"
             onload="applyCompareMediaTransform('${slotId}', '${viewMode}')"
             style="width:100%; height:100%; object-fit:contain; display:block;"/>
      ` : `<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);background:rgba(255,255,255,0.02);border-radius:6px;">无视频/定格切片</div>`);

      const turnVal = sw.shoulder_turn_change != null ? `${Number(sw.shoulder_turn_change).toFixed(1)}°` : "—";
      const extVal = sw.arm_extension != null ? `${Number(sw.arm_extension).toFixed(1)}°` : "—";
      const spdRaw = sw.racket_speed_px_s ?? sw.racket_max_speed_px_s;
      const spdVal = spdRaw != null ? `${Math.round(spdRaw)} px/s` : "—";

      setTimeout(() => applyCompareMediaTransform(slotId, viewMode), 20);

      return `
        <div class="cmp-media-stage">
          ${mediaInner}
          <div style="position:absolute; top:8px; left:8px; right:8px; display:flex; justify-content:space-between; align-items:center; pointer-events:none;">
            <span id="cmp-view-badge-${slotId}" style="font-size:11px; padding:3px 8px; border-radius:4px; background:rgba(0,0,0,0.72); color:#e2e8f0; border:1px solid rgba(255,255,255,0.14);">视角加载中...</span>
            <span id="cmp-pan-wrap-${slotId}" style="pointer-events:auto;" hidden>
              <button type="button" onclick="nudgeCompareFocus('${slotId}', -8)" style="font-size:11px; padding:2px 7px; background:rgba(0,0,0,0.75); color:#fff; border:1px solid rgba(255,255,255,0.22); border-radius:4px; cursor:pointer;" title="向左平移单视角放大焦点">◀左移</button>
              <button type="button" onclick="nudgeCompareFocus('${slotId}', +8)" style="font-size:11px; padding:2px 7px; background:rgba(0,0,0,0.75); color:#fff; border:1px solid rgba(255,255,255,0.22); border-radius:4px; cursor:pointer;" title="向右平移单视角放大焦点">右移▶</button>
            </span>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-top:6px;">
          <span>触球帧: ${sw.contact_frame ?? "—"}F (${offsetSec}s)</span>
          <span>转肩 ${turnVal} · 肘角 ${extVal} · 触球拍速 ${spdVal}</span>
        </div>
      `;
    }

    function renderSwingVisualSlot(sideLabel, sess, slotId) {
      if (!sess) return `<div style="padding:10px;color:var(--muted);">未选择会话</div>`;
      const swings = sess.swings || [];
      const rep = sess.representative_event || swings[0] || null;
      if (!rep) return `<div style="padding:10px;color:var(--muted);">该会话无有效实战挥拍</div>`;

      const viewBadge = sess.view_mode === "dual_view" ? "双视角" : "单视角";
      const optionsHtml = swings.map(sw => {
        const selected = sw.event_id === rep.event_id ? "selected" : "";
        const spdVal = sw.racket_speed_px_s ?? sw.racket_max_speed_px_s;
        const spd = spdVal != null ? ` · ${Math.round(spdVal)}px/s` : "";
        return `<option value="${sw.event_id}" ${selected}>Event #${sw.event_id} (${escapeHtml(sw.stroke_type || "Swing")}${spd})</option>`;
      }).join("");

      return `
        <div style="background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px;">
            <strong style="font-size:12px; color:${slotId === 'A' ? '#38bdf8' : '#4ade80'};">${sideLabel} [${viewBadge}] · ${escapeHtml(sess.display_time)}</strong>
            <select id="cmp-swing-select-${slotId}" onchange="updateCompareSwingSlot('${slotId}')" style="width:auto; font-size:12px; height:28px; background:rgba(255,255,255,0.06); border:1px solid var(--line); border-radius:6px; color:var(--text); padding:0 8px;">
              ${optionsHtml}
            </select>
          </div>
          <div id="cmp-swing-media-${slotId}">
            ${buildSwingMediaBoxHtml(rep, slotId, sess.view_mode)}
          </div>
        </div>
      `;
    }

    function updateCompareSwingSlot(slotId) {
      const sessId = slotId === "A" ? selectedCompareSessionA : selectedCompareSessionB;
      const sess = latestCompareSessions.find(s => s.session_id === sessId);
      const sel = document.getElementById(`cmp-swing-select-${slotId}`);
      const box = document.getElementById(`cmp-swing-media-${slotId}`);
      if (!sess || !sel || !box) return;
      const eid = Number(sel.value);
      const sw = (sess.swings || []).find(x => x.event_id === eid) || sess.representative_event;
      box.innerHTML = buildSwingMediaBoxHtml(sw, slotId, sess.view_mode);
    }

    function syncPlayABClips(rate = 0.25) {
      ["A", "B"].forEach(id => {
        const v = document.getElementById(`cmp-video-${id}`);
        if (!v) return;
        const cSec = parseFloat(v.dataset.contactSec || "0") || 0;
        v.pause();
        v.playbackRate = rate;
        v.currentTime = Math.max(0, cSec - 0.5);
        v.play().catch(() => {});
      });
    }

    function syncPauseABClips() {
      ["A", "B"].forEach(id => {
        const v = document.getElementById(`cmp-video-${id}`);
        if (v) v.pause();
      });
    }

    function syncSeekABContact() {
      ["A", "B"].forEach(id => {
        const v = document.getElementById(`cmp-video-${id}`);
        if (!v) return;
        const cSec = parseFloat(v.dataset.contactSec || "0") || 0;
        v.pause();
        v.currentTime = Math.max(0, cSec);
      });
    }

    function openSessionInConsole(sid) {
      if (!sid) return;
      window.open(`/?session_id=${encodeURIComponent(sid)}`, "_blank", "noopener");
    }

    function renderCompareDashboard() {
      const body = document.getElementById("player-compare-body");
      if (!body) return;
      const sessions = latestCompareSessions;
      if (!sessions || sessions.length === 0) {
        body.innerHTML = `<div style="padding:28px 12px; text-align:center; color:var(--muted);">该学员在当前筛选条件下暂无已入库击球会话。请点击上方「📂 批量归属会话」将历史会话归属给该学员。</div>`;
        return;
      }

      if (!selectedCompareSessionB || !sessions.some(s => s.session_id === selectedCompareSessionB)) {
        selectedCompareSessionB = sessions[sessions.length - 1].session_id;
      }
      if (!selectedCompareSessionA || !sessions.some(s => s.session_id === selectedCompareSessionA)) {
        selectedCompareSessionA = sessions.length >= 2 ? sessions[sessions.length - 2].session_id : sessions[0].session_id;
      }

      const sessA = sessions.find(s => s.session_id === selectedCompareSessionA) || sessions[0];
      const sessB = sessions.find(s => s.session_id === selectedCompareSessionB) || sessions[sessions.length - 1];

      const sessOptsA = sessions.map(s => {
        const vTag = s.view_mode === "dual_view" ? "[双视角] " : "[单视角] ";
        return `<option value="${escapeHtml(s.session_id)}" ${s.session_id === sessA.session_id ? "selected" : ""}>${vTag}${escapeHtml(s.display_time)} (${s.matched_swings}拍)</option>`;
      }).join("");
      const sessOptsB = sessions.map(s => {
        const vTag = s.view_mode === "dual_view" ? "[双视角] " : "[单视角] ";
        return `<option value="${escapeHtml(s.session_id)}" ${s.session_id === sessB.session_id ? "selected" : ""}>${vTag}${escapeHtml(s.display_time)} (${s.matched_swings}拍)</option>`;
      }).join("");

      const metricRowsHtml = CORE_COMPARE_METRICS.map(cfg => {
        const mA = sessA?.metrics?.[cfg.key];
        const mB = sessB?.metrics?.[cfg.key];
        const valA = mA?.median;
        const valB = mB?.median;
        const iqrA = mA?.iqr;
        const iqrB = mB?.iqr;
        const conB = mB?.consistency_score;

        let deltaHtml = "—";
        let proofHtml = "<span style='color:var(--muted);'>样本待积累</span>";
        if (valA != null && valB != null) {
          const diff = Number((valB - valA).toFixed(2));
          const improved = cfg.higherBetter ? diff > 0 : diff < 0;
          const sign = diff > 0 ? "+" : "";
          const color = diff === 0 ? "var(--muted)" : (improved ? "#4ade80" : "#f87171");

          const pooledIqr = Math.max(((iqrA ?? 0) + (iqrB ?? 0)) / 2.0, Math.abs(valA) * 0.06, 0.01);
          const effectRatio = Number((Math.abs(diff) / pooledIqr).toFixed(2));
          let sigBadge = "";
          if (effectRatio >= 0.75) {
            sigBadge = improved
              ? `<span style="color:#4ade80;margin-left:4px;">[🔥显著进步 ${effectRatio}x]</span>`
              : `<span style="color:#f87171;margin-left:4px;">[⚠️显著退化 ${effectRatio}x]</span>`;
          } else if (effectRatio >= 0.35) {
            sigBadge = improved
              ? `<span style="color:#38bdf8;margin-left:4px;">[📈有效改善 ${effectRatio}x]</span>`
              : `<span style="color:#fbbf24;margin-left:4px;">[📉有所下滑 ${effectRatio}x]</span>`;
          } else {
            sigBadge = `<span style="color:var(--muted);margin-left:4px;">[〰️正常波动 ${effectRatio}x]</span>`;
          }
          deltaHtml = `<span style="color:${color}; font-weight:600;">${sign}${diff} ${cfg.unit}</span>${sigBadge}`;

          if (iqrA != null && iqrB != null && (mA.count >= 2 || mB.count >= 2)) {
            const iqrDiff = Number((iqrB - iqrA).toFixed(2));
            const conTag = conB != null ? ` · 定型度 ${conB}%` : "";
            if (improved && iqrDiff <= 0) {
              proofHtml = `<span style="color:#4ade80;">✅ 幅度提升 + 波动收窄 (±${iqrA}→±${iqrB}${conTag})</span>`;
            } else if (improved) {
              proofHtml = `<span style="color:#38bdf8;">↗ 幅度提升 · 波动 ±${iqrB}${conTag}</span>`;
            } else if (iqrDiff < 0) {
              proofHtml = `<span style="color:#fbbf24;">⚖️ 动作定型收窄 (±${iqrA}→±${iqrB}${conTag})</span>`;
            } else {
              proofHtml = `<span style="color:var(--muted);">P25..P75: ${mB.p25 ?? "—"}~${mB.p75 ?? "—"}${conTag}</span>`;
            }
          }
        }

        return `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:6px 8px;"><strong>${cfg.label}</strong></td>
            <td style="padding:6px 8px; color:#38bdf8;">${valA != null ? `${valA} ${cfg.unit} <small style="color:var(--muted);">±${iqrA ?? 0}</small>` : "—"}</td>
            <td style="padding:6px 8px; color:#4ade80;">${valB != null ? `${valB} ${cfg.unit} <small style="color:var(--muted);">±${iqrB ?? 0}</small>` : "—"}</td>
            <td style="padding:6px 8px;">${deltaHtml}</td>
            <td style="padding:6px 8px; font-size:12px;">${proofHtml}</td>
          </tr>
        `;
      }).join("");

      const attr = latestPersonalAttribution;
      const attributionHtml = attr && attr.available ? `
        <div style="margin-bottom:14px; padding:12px 14px; background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.22); border-radius:10px;">
          <div style="font-size:13px; font-weight:600; color:#38bdf8; margin-bottom:4px;">🔑 个人标杆分位归因 (Top 25% 好球 vs Bottom 25% 弱球 · 共 ${attr.sample_count} 拍)</div>
          <div style="font-size:13px; line-height:1.5; color:var(--text); margin-bottom:8px;">${escapeHtml(attr.insight || "")}</div>
          <div style="display:flex; flex-wrap:wrap; gap:8px;">
            ${(attr.factors || []).slice(0, 4).map(f => `
              <span style="font-size:12px; padding:4px 9px; background:rgba(255,255,255,0.05); border-radius:5px;">
                <strong>${escapeHtml(f.label)}</strong>: 好球 ${f.top_median}${f.unit} vs 弱球 ${f.bottom_median}${f.unit}
                <span style="color:#4ade80; margin-left:4px;">(效应量 ${f.effect_ratio}x)</span>
              </span>
            `).join("")}
          </div>
        </div>
      ` : "";

      const fatA = sessA?.fatigue_split || {};
      const fatB = sessB?.fatigue_split || {};
      const warnMap = new Map();
      (sessA?.coach_warnings || []).forEach(w => warnMap.set(w.message, { a: w.rate_pct, b: 0 }));
      (sessB?.coach_warnings || []).forEach(w => {
        const cur = warnMap.get(w.message) || { a: 0, b: 0 };
        cur.b = w.rate_pct;
        warnMap.set(w.message, cur);
      });
      const warnRowsHtml = Array.from(warnMap.entries()).slice(0, 5).map(([msg, ab]) => {
        const diff = Number((ab.b - ab.a).toFixed(1));
        const color = diff < 0 ? "#4ade80" : (diff > 0 ? "#f87171" : "var(--muted)");
        const sign = diff > 0 ? "+" : "";
        return `<div style="display:flex; justify-content:space-between; font-size:12px; padding:4px 0; border-bottom:1px dashed rgba(255,255,255,0.06);">
          <span>${escapeHtml(msg)}</span>
          <span><span style="color:#38bdf8;">${ab.a}%</span> → <span style="color:#4ade80;">${ab.b}%</span> <strong style="color:${color}; margin-left:4px;">(${sign}${diff}%)</strong></span>
        </div>`;
      }).join("") || `<div style="font-size:12px; color:var(--muted);">A/B 会话均未检出显著动作瑕疵告警 🎉</div>`;

      const fatigueAndWarningsHtml = `
        <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:14px;">
          <div style="flex:1; min-width:280px; padding:12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:10px;">
            <strong style="font-size:13px; display:block; margin-bottom:6px; color:var(--text);">🔋 单次训练前后半程保持度 (体能/专注衰减分析)</strong>
            <div style="font-size:12px; line-height:1.65;">
              <div><span style="color:#38bdf8;">基准 A (${sessA?.consistency_index != null ? `定型度 ${sessA.consistency_index}%` : "—"}):</span> ${escapeHtml(fatA.available ? fatA.summary : (fatA.reason || "样本不足"))}</div>
              <div><span style="color:#4ade80;">当前 B (${sessB?.consistency_index != null ? `定型度 ${sessB.consistency_index}%` : "—"}):</span> ${escapeHtml(fatB.available ? fatB.summary : (fatB.reason || "样本不足"))}</div>
            </div>
          </div>
          <div style="flex:1; min-width:280px; padding:12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:10px;">
            <strong style="font-size:13px; display:block; margin-bottom:6px; color:var(--text);">⚠️ 动作瑕疵复发率对比 (基准 A → 当前 B)</strong>
            ${warnRowsHtml}
          </div>
        </div>
      `;

      body.innerHTML = `
        <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px; background:rgba(255,255,255,0.03); border:1px solid var(--line); border-radius:10px; margin-bottom:14px;">
          <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
            <span style="color:#38bdf8; font-weight:600;">基准 A:</span>
            <select id="cmp-session-a" onchange="onChangeCompareAB()" style="width:auto; font-size:12px; height:32px; background:#0a1018; border:1px solid var(--line); border-radius:6px; color:var(--text); padding:0 8px;">${sessOptsA}</select>
            <span style="color:var(--muted);">VS</span>
            <span style="color:#4ade80; font-weight:600;">当前 B:</span>
            <select id="cmp-session-b" onchange="onChangeCompareAB()" style="width:auto; font-size:12px; height:32px; background:#0a1018; border:1px solid var(--line); border-radius:6px; color:var(--text); padding:0 8px;">${sessOptsB}</select>
          </div>
          <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
            <div style="display:inline-flex; gap:2px; background:rgba(0,0,0,0.25); padding:2px; border-radius:6px; border:1px solid rgba(255,255,255,0.08);">
              <button type="button" onclick="setCompareCropMode('front_focus')" style="height:28px; padding:0 10px; font-size:12px; border:none; border-radius:5px; cursor:pointer; background:${compareCropMode === 'front_focus' ? 'var(--blue, #38bdf8)' : 'transparent'}; color:${compareCropMode === 'front_focus' ? '#000' : 'var(--text)'}; font-weight:${compareCropMode === 'front_focus' ? '600' : '400'};">🎯 聚焦正面</button>
              <button type="button" onclick="setCompareCropMode('back_focus')" style="height:28px; padding:0 10px; font-size:12px; border:none; border-radius:5px; cursor:pointer; background:${compareCropMode === 'back_focus' ? 'var(--blue, #38bdf8)' : 'transparent'}; color:${compareCropMode === 'back_focus' ? '#000' : 'var(--text)'}; font-weight:${compareCropMode === 'back_focus' ? '600' : '400'};">🪞 聚焦镜面</button>
              <button type="button" onclick="setCompareCropMode('raw')" style="height:28px; padding:0 10px; font-size:12px; border:none; border-radius:5px; cursor:pointer; background:${compareCropMode === 'raw' ? 'var(--blue, #38bdf8)' : 'transparent'}; color:${compareCropMode === 'raw' ? '#000' : 'var(--text)'}; font-weight:${compareCropMode === 'raw' ? '600' : '400'};">🖼️ 原画幅</button>
            </div>
            <button type="button" class="button" style="min-height:32px; padding:0 10px; font-size:12px;" onclick="syncSeekABContact()">💥 对齐触球帧定格</button>
            <button type="button" class="button" style="min-height:32px; padding:0 10px; font-size:12px;" onclick="syncPlayABClips(0.25)">🎬 0.25x 触球同步慢放</button>
            <button type="button" class="button" style="min-height:32px; padding:0 10px; font-size:12px;" onclick="syncPlayABClips(0.5)">0.5x 同步</button>
            <button type="button" class="button" style="min-height:32px; padding:0 10px; font-size:12px;" onclick="syncPauseABClips()">⏸ 同步暂停</button>
          </div>
        </div>

        <div class="compare-ab-slots">
          ${renderSwingVisualSlot("基准会话 A", sessA, "A")}
          ${renderSwingVisualSlot("当前会话 B", sessB, "B")}
        </div>

        <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-bottom:14px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:12px;">
          <div style="flex:0 0 240px; margin:0 auto;">
            ${buildDualRadarSvg(sessA, sessB)}
          </div>
          <div style="flex:1; min-width:320px; overflow:auto;">
            <table style="width:100%; font-size:13px; border-collapse:collapse; text-align:left; color:var(--text);">
              <thead>
                <tr style="border-bottom:1px solid var(--line); color:var(--muted);">
                  <th style="padding:6px 8px;">核心生物力学维度</th>
                  <th style="padding:6px 8px;">基准 A (中位±IQR)</th>
                  <th style="padding:6px 8px;">当前 B (中位±IQR)</th>
                  <th style="padding:6px 8px;">中位变化 Δ</th>
                  <th style="padding:6px 8px;">数据稳定性证明</th>
                </tr>
              </thead>
              <tbody>
                ${metricRowsHtml}
              </tbody>
            </table>
          </div>
        </div>

        ${attributionHtml}
        ${fatigueAndWarningsHtml}

        <details style="margin-top:6px;">
          <summary style="cursor:pointer; color:var(--muted); font-size:12px;">查看全部 ${sessions.length} 次训练会话序列明细</summary>
          <div style="overflow:auto; margin-top:8px;">
            <table style="width:100%; font-size:12px; border-collapse:collapse; text-align:left; color:var(--text);">
              <thead>
                <tr style="border-bottom:1px solid var(--line); color:var(--muted);">
                  <th style="padding:5px 8px;">日期时间</th>
                  <th style="padding:5px 8px;">有效拍数</th>
                  <th style="padding:5px 8px;">转肩变化</th>
                  <th style="padding:5px 8px;">肩髋分离</th>
                  <th style="padding:5px 8px;">肘伸展角</th>
                  <th style="padding:5px 8px;">重心横移</th>
                  <th style="padding:5px 8px;">拍框峰值</th>
                  <th style="padding:5px 8px;">操作</th>
                </tr>
              </thead>
              <tbody>
                ${sessions.slice().reverse().map(s => `
                  <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:5px 8px;">${escapeHtml(s.display_time)}</td>
                    <td style="padding:5px 8px;">${s.matched_swings} 拍</td>
                    <td style="padding:5px 8px;">${s.metrics?.shoulder_turn_change?.median != null ? `${s.metrics.shoulder_turn_change.median}°` : "—"}</td>
                    <td style="padding:5px 8px;">${s.metrics?.hip_shoulder_separation?.median != null ? `${s.metrics.hip_shoulder_separation.median}°` : "—"}</td>
                    <td style="padding:5px 8px;">${s.metrics?.arm_extension?.median != null ? `${s.metrics.arm_extension.median}°` : "—"}</td>
                    <td style="padding:5px 8px;">${s.metrics?.weight_transfer?.median != null ? `${s.metrics.weight_transfer.median} bw` : "—"}</td>
                    <td style="padding:5px 8px;">${s.metrics?.racket_max_speed?.median != null ? `${Math.round(s.metrics.racket_max_speed.median)} px/s` : "—"}</td>
                    <td style="padding:5px 8px;"><button type="button" class="button" style="min-height:24px;padding:0 8px;font-size:11px;" onclick="openSessionInConsole('${escapeHtml(s.session_id)}')">打开会话 ↗</button></td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </details>
      `;
    }

    function onChangeCompareAB() {
      const selA = document.getElementById("cmp-session-a");
      const selB = document.getElementById("cmp-session-b");
      if (selA) selectedCompareSessionA = selA.value;
      if (selB) selectedCompareSessionB = selB.value;
      renderCompareDashboard();
    }

    async function loadPlayerComparison() {
      const body = document.getElementById("player-compare-body");
      const playerSel = document.getElementById("compare-player-select");
      const strokeSel = document.getElementById("compare-stroke-select");
      if (!body || !playerSel) return;
      const pid = playerSel.value || "default";
      const stroke = strokeSel ? strokeSel.value : "all";
      body.textContent = "正在查询数据库对比数据...";
      try {
        const res = await api(`/api/players/compare?player_id=${encodeURIComponent(pid)}&stroke_type=${encodeURIComponent(stroke)}`);
        const data = await res.json();
        latestCompareSessions = data.sessions || [];
        latestPersonalAttribution = data.personal_attribution || null;
        renderCompareDashboard();
      } catch (err) {
        body.textContent = `查询对比数据失败：${err.message}`;
      }
    }

    function loadActiveViewData() {
      return pageViewMode === "ab" ? loadPlayerComparison() : loadGlobalProfile();
    }

    /* ============================================================
     * Batch Session Assignment Modal
     * ============================================================ */
    function updateBatchCountBadge() {
      const el = document.getElementById("batch-selected-count");
      if (el) el.textContent = `已选 ${batchCheckedIds.size} 场`;
    }

    function toggleBatchRow(sid, checked) {
      if (checked) batchCheckedIds.add(sid);
      else batchCheckedIds.delete(sid);
      updateBatchCountBadge();
    }

    function getFilteredBatchSessions() {
      const onlyDefault = Boolean(document.getElementById("batch-only-default-chk")?.checked);
      return onlyDefault
        ? batchAllSessions.filter(s => !s.player_id || s.player_id === "default")
        : batchAllSessions;
    }

    function selectAllBatchRows(checked) {
      getFilteredBatchSessions().forEach(s => {
        if (checked) batchCheckedIds.add(s.session_id);
        else batchCheckedIds.delete(s.session_id);
      });
      renderBatchSessionRows();
    }

    function renderBatchSessionRows() {
      const box = document.getElementById("batch-sessions-list");
      if (!box) return;
      const rows = getFilteredBatchSessions();
      updateBatchCountBadge();
      if (!rows.length) {
        box.innerHTML = `<div style="padding:20px; text-align:center; color:var(--muted);">当前筛选下暂无会话。</div>`;
        return;
      }
      box.innerHTML = `
        <table style="width:100%; font-size:12px; border-collapse:collapse; text-align:left;">
          <thead>
            <tr style="border-bottom:1px solid var(--line); color:var(--muted);">
              <th style="padding:6px; width:34px;">选</th>
              <th style="padding:6px;">时间</th>
              <th style="padding:6px;">视频/来源</th>
              <th style="padding:6px;">拍数</th>
              <th style="padding:6px;">当前归属</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(s => {
              const chk = batchCheckedIds.has(s.session_id) ? "checked" : "";
              const viewTag = s.view_mode === "dual_view" ? "双视角" : "单视角";
              const groupBadge = (s.source_group_size > 1)
                ? `<span style="font-size:10px; padding:1px 5px; border-radius:4px; background:rgba(56,189,248,0.16); color:#38bdf8; margin-left:4px;">同视频×${s.source_group_size}</span>`
                : (s.is_duplicate
                  ? `<span style="font-size:10px; padding:1px 5px; border-radius:4px; background:rgba(251,191,36,0.16); color:#fbbf24; margin-left:4px;">同视频冗余</span>`
                  : "");
              return `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                  <td style="padding:6px;">
                    <input type="checkbox" ${chk} style="width:15px; height:15px;" onchange="toggleBatchRow('${escapeHtml(s.session_id)}', this.checked)">
                  </td>
                  <td style="padding:6px; white-space:nowrap;">${escapeHtml(s.display_time)}</td>
                  <td style="padding:6px; max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(s.session_id)}">${escapeHtml(s.source_name || s.session_id)}${groupBadge}</td>
                  <td style="padding:6px; white-space:nowrap;">${s.event_count} 拍 (${viewTag})</td>
                  <td style="padding:6px; white-space:nowrap; color:${s.player_id === 'default' ? 'var(--muted)' : '#38bdf8'};">${escapeHtml(s.player_name || '默认球员')}</td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      `;
    }

    async function openBatchAssignModal() {
      const modal = document.getElementById("batch-assign-modal");
      const box = document.getElementById("batch-sessions-list");
      const targetSel = document.getElementById("batch-target-player");
      const curPlayer = document.getElementById("compare-player-select")?.value;
      if (targetSel && curPlayer) targetSel.value = curPlayer;
      if (modal) modal.hidden = false;
      if (box) box.textContent = "正在加载全部历史会话...";
      batchCheckedIds.clear();
      try {
        const res = await api("/api/sessions");
        const list = await res.json();
        batchAllSessions = Array.isArray(list) ? list : [];
        renderBatchSessionRows();
      } catch (err) {
        if (box) box.textContent = `加载会话失败：${err.message}`;
      }
    }

    function closeBatchAssignModal() {
      const modal = document.getElementById("batch-assign-modal");
      if (modal) modal.hidden = true;
    }

    async function submitBatchAssign() {
      const ids = Array.from(batchCheckedIds);
      const targetPid = document.getElementById("batch-target-player")?.value || "default";
      if (!ids.length) {
        showToast("请先勾选至少一场历史会话", true);
        return;
      }
      const btn = document.getElementById("batch-submit-btn");
      if (btn) btn.disabled = true;
      try {
        const res = await api("/api/sessions/assign-player-batch", {
          method: "POST",
          body: JSON.stringify({ session_ids: ids, player_id: targetPid }),
        });
        const data = await res.json();
        closeBatchAssignModal();
        await loadComparePlayers(targetPid);
        const mainSel = document.getElementById("compare-player-select");
        if (mainSel) mainSel.value = targetPid;
        syncUrlParams();
        await loadActiveViewData();
        showToast(`已将 ${data.updated_count} 场会话归属给「${data.player_name}」`);
      } catch (err) {
        showToast(`批量归属失败：${err.message}`, true);
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    /* ============================================================
     * Duplicate Video Session Cleanup Modal
     * ============================================================ */
    function updateDedupCountBadge() {
      const el = document.getElementById("dedup-selected-count");
      if (el) el.textContent = `已选 ${dedupCheckedIds.size} 场冗余`;
    }

    function toggleDedupRow(sid, checked) {
      if (checked) dedupCheckedIds.add(sid);
      else dedupCheckedIds.delete(sid);
      updateDedupCountBadge();
    }

    function selectRecommendedDuplicates(checked) {
      dedupCheckedIds.clear();
      if (checked && dedupModalData && Array.isArray(dedupModalData.recommended_cleanup_ids)) {
        dedupModalData.recommended_cleanup_ids.forEach(sid => dedupCheckedIds.add(sid));
      }
      renderDedupModalGroups();
    }

    function renderDedupModalGroups() {
      const box = document.getElementById("dedup-groups-list");
      if (!box) return;
      updateDedupCountBadge();
      const groups = dedupModalData?.groups || [];
      if (!groups.length) {
        box.innerHTML = `<div style="padding:28px 16px; text-align:center; color:#4ade80;">当前范围内未发现同源重复视频记录 🎉（共 ${dedupModalData?.raw_session_count || 0} 场独立会话）</div>`;
        return;
      }
      const headerSummary = `
        <div style="margin-bottom:10px; padding:8px 10px; border-radius:8px; background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.2); font-size:12px;">
          共扫描 <strong>${dedupModalData.raw_session_count}</strong> 场分析记录，识别出 <strong>${dedupModalData.duplicate_group_count}</strong> 组同原视频重跑（合计 <strong>${dedupModalData.duplicate_session_count}</strong> 场冗余可归档，保留后剩余 <strong>${dedupModalData.unique_session_count}</strong> 场独立视频）。
        </div>
      `;
      const groupsHtml = groups.map((g, gIdx) => {
        const canon = g.canonical || {};
        const dups = g.duplicates || [];
        const matchLabel = g.match_by === "source_sha256" ? "SHA-256 内容一致" : "文件名一致";
        const shaShort = g.source_sha256 ? ` · 指纹 ${String(g.source_sha256).slice(0, 12)}` : "";
        const canonView = canon.view_mode === "dual_view" ? "双视角" : "单视角";
        const canonMatched = (canon.forehand_count || 0) + (canon.backhand_count || 0);
        return `
          <div style="margin-bottom:12px; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:rgba(255,255,255,0.015);">
            <div style="padding:8px 12px; background:rgba(255,255,255,0.03); border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; font-size:12px;">
              <div>
                <strong>#${gIdx + 1} ${escapeHtml(g.label || "未命名视频")}</strong>
                <span style="margin-left:8px; font-size:11px; padding:1px 6px; border-radius:4px; background:rgba(56,189,248,0.14); color:#38bdf8;">${matchLabel}${shaShort}</span>
              </div>
              <span style="color:var(--muted);">同视频共 ${g.total_count} 场 · 建议保留 1 场 / 清理 ${dups.length} 场</span>
            </div>
            <div style="padding:8px 12px;">
              <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; padding:6px 8px; border-radius:6px; background:rgba(74,222,128,0.07); border:1px solid rgba(74,222,128,0.22); margin-bottom:6px; font-size:12px;">
                <span>
                  <span style="font-size:10px; padding:1px 6px; border-radius:4px; background:rgba(74,222,128,0.22); color:#4ade80; font-weight:600; margin-right:6px;">保留代表</span>
                  <strong>${escapeHtml(canon.display_time || "")}</strong> · ${canonView} · ${canonMatched} 实战拍 (总${canon.event_count || 0}) · <code style="font-size:11px;">${escapeHtml(canon.source_name || canon.session_id)}</code>
                </span>
                <span style="color:var(--muted); font-size:11px;">学员: ${escapeHtml(canon.player_name || canon.player_id || "默认球员")}</span>
              </div>
              ${dups.map(d => {
                const chk = dedupCheckedIds.has(d.session_id) ? "checked" : "";
                const dView = d.view_mode === "dual_view" ? "双视角" : "单视角";
                const dMatched = (d.forehand_count || 0) + (d.backhand_count || 0);
                return `
                  <label style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; padding:5px 8px; border-bottom:1px dashed rgba(255,255,255,0.06); font-size:12px; cursor:pointer; margin:0; text-transform:none; letter-spacing:0;">
                    <span style="display:inline-flex; align-items:center; gap:7px;">
                      <input type="checkbox" ${chk} style="width:14px; height:14px;" onchange="toggleDedupRow('${escapeHtml(d.session_id)}', this.checked)">
                      <span style="font-size:10px; padding:1px 5px; border-radius:4px; background:rgba(251,191,36,0.16); color:#fbbf24;">冗余重跑</span>
                      <span>${escapeHtml(d.display_time || "")}</span>
                      <span style="color:var(--muted);">· ${dView} · ${dMatched} 实战拍 (总${d.event_count || 0}) · <code>${escapeHtml(d.source_name || d.session_id)}</code></span>
                    </span>
                    <span style="color:var(--muted); font-size:11px;">学员: ${escapeHtml(d.player_name || d.player_id || "默认球员")}</span>
                  </label>
                `;
              }).join("")}
            </div>
          </div>
        `;
      }).join("");
      box.innerHTML = headerSummary + groupsHtml;
    }

    async function loadDedupGroups() {
      const box = document.getElementById("dedup-groups-list");
      const scope = document.getElementById("dedup-scope-select")?.value || "current";
      const curPlayer = document.getElementById("compare-player-select")?.value || "";
      if (box) box.textContent = "正在扫描同源视频指纹...";
      dedupCheckedIds.clear();
      try {
        const qs = new URLSearchParams();
        if (scope === "current" && curPlayer) qs.set("player_id", curPlayer);
        const res = await api(`/api/sessions/duplicates${qs.toString() ? `?${qs.toString()}` : ""}`);
        dedupModalData = await res.json();
        (dedupModalData.recommended_cleanup_ids || []).forEach(sid => dedupCheckedIds.add(sid));
        renderDedupModalGroups();
      } catch (err) {
        if (box) box.textContent = `扫描重复视频失败：${err.message}`;
      }
    }

    function openDedupModal() {
      const modal = document.getElementById("dedup-modal");
      if (modal) modal.hidden = false;
      loadDedupGroups();
    }

    function closeDedupModal() {
      const modal = document.getElementById("dedup-modal");
      if (modal) modal.hidden = true;
    }

    async function submitDedupCleanup(mode = "archive") {
      const ids = Array.from(dedupCheckedIds);
      if (!ids.length) {
        showToast("请先勾选至少一条冗余分析记录", true);
        return;
      }
      const btn = document.getElementById("dedup-archive-btn");
      if (btn) btn.disabled = true;
      try {
        const res = await api("/api/sessions/dedup-cleanup", {
          method: "POST",
          body: JSON.stringify({ session_ids: ids, mode }),
        });
        const data = await res.json();
        closeDedupModal();
        await loadComparePlayers();
        await loadActiveViewData();
        showToast(`已安全归档 ${data.cleaned_count} 场重复记录至 control_panel_duplicates_archive`);
      } catch (err) {
        showToast(`去重清理失败：${err.message}`, true);
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    /* ============================================================
     * Per-Swing Trajectory Fitting & Repair Inspector Modal
     * ============================================================ */
    function buildFrameCurveSvg(opts) {
      const {
        title,
        unit,
        frameIds,
        lines,
        oldCf,
        newCf,
        prepWindow,
        rejectedMask,
        interpMask,
        markerSeries,
      } = opts;
      const n = frameIds.length;
      if (!n) return "";
      const allVals = [];
      lines.forEach(l => (l.values || []).forEach(v => { if (v != null && Number.isFinite(v)) allVals.push(v); }));
      if (!allVals.length) {
        return `<div style="padding:18px; text-align:center; color:var(--muted); font-size:12px; border:1px solid var(--line); border-radius:8px; margin-bottom:10px;">${escapeHtml(title)}：暂无有效逐帧数据</div>`;
      }
      let minV = Math.min(...allVals);
      let maxV = Math.max(...allVals);
      if (Math.abs(maxV - minV) < 1e-6) { minV -= 1; maxV += 1; }
      else { const pad = (maxV - minV) * 0.12; minV -= pad; maxV += pad; }

      const W = 940, H = 205, padL = 52, padR = 20, padT = 20, padB = 28;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const fMin = frameIds[0], fMax = frameIds[n - 1];
      const fSpan = Math.max(1, fMax - fMin);
      const xOfF = (f) => padL + ((f - fMin) / fSpan) * plotW;
      const yOf = (v) => padT + (1 - (v - minV) / (maxV - minV)) * plotH;

      const gridHtml = [0, 0.5, 1.0].map(t => {
        const y = padT + t * plotH;
        const val = maxV - t * (maxV - minV);
        const lbl = Math.abs(val) >= 100 ? Math.round(val) : val.toFixed(1);
        return `
          <line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
          <text x="${padL - 7}" y="${(y + 3).toFixed(1)}" fill="#8d9caf" font-size="9.5" text-anchor="end">${lbl}</text>
        `;
      }).join("");

      let prepBandHtml = "";
      if (Array.isArray(prepWindow) && prepWindow.length === 2) {
        const x1 = Math.max(padL, xOfF(prepWindow[0]));
        const x2 = Math.min(W - padR, xOfF(prepWindow[1]));
        if (x2 > x1) {
          prepBandHtml = `<rect x="${x1.toFixed(1)}" y="${padT}" width="${(x2 - x1).toFixed(1)}" height="${plotH}" fill="rgba(74,222,128,0.10)"/>`;
        }
      }

      const pathsHtml = lines.map(l => {
        const segs = [];
        let cur = [];
        (l.values || []).forEach((v, idx) => {
          if (v != null && Number.isFinite(v)) {
            cur.push(`${xOfF(frameIds[idx]).toFixed(1)},${yOf(v).toFixed(1)}`);
          } else if (cur.length) {
            segs.push(cur);
            cur = [];
          }
        });
        if (cur.length) segs.push(cur);
        return segs.map(pts => (
          pts.length >= 2
            ? `<polyline points="${pts.join(" ")}" fill="none" stroke="${l.color}" stroke-width="${l.width || 1.8}" ${l.dashed ? 'stroke-dasharray="4,3"' : ""} opacity="${l.opacity ?? 1}" stroke-linejoin="round"/>`
            : `<circle cx="${pts[0].split(",")[0]}" cy="${pts[0].split(",")[1]}" r="2.5" fill="${l.color}"/>`
        )).join("");
      }).join("");

      let markersHtml = "";
      if (Array.isArray(rejectedMask) && Array.isArray(markerSeries?.raw)) {
        rejectedMask.forEach((isRej, idx) => {
          if (!isRej) return;
          const v = markerSeries.raw[idx] ?? maxV * 0.95;
          const cx = xOfF(frameIds[idx]).toFixed(1);
          const cy = yOf(v).toFixed(1);
          markersHtml += `
            <g>
              <line x1="${Number(cx) - 4}" y1="${Number(cy) - 4}" x2="${Number(cx) + 4}" y2="${Number(cy) + 4}" stroke="#f87171" stroke-width="1.8"/>
              <line x1="${Number(cx) + 4}" y1="${Number(cy) - 4}" x2="${Number(cx) - 4}" y2="${Number(cy) + 4}" stroke="#f87171" stroke-width="1.8"/>
            </g>
          `;
        });
      }
      if (Array.isArray(interpMask) && Array.isArray(markerSeries?.smooth)) {
        interpMask.forEach((isInt, idx) => {
          if (!isInt) return;
          const v = markerSeries.smooth[idx];
          if (v == null || !Number.isFinite(v)) return;
          markersHtml += `<circle cx="${xOfF(frameIds[idx]).toFixed(1)}" cy="${yOf(v).toFixed(1)}" r="3.6" fill="#4ade80" stroke="#0a1018" stroke-width="1"/>`;
        });
      }

      const oldLine = oldCf != null ? `<line x1="${xOfF(oldCf).toFixed(1)}" y1="${padT}" x2="${xOfF(oldCf).toFixed(1)}" y2="${padT + plotH}" stroke="#fbbf24" stroke-width="1.4" stroke-dasharray="4,3"/>` : "";
      const newLine = newCf != null ? `<line x1="${xOfF(newCf).toFixed(1)}" y1="${padT}" x2="${xOfF(newCf).toFixed(1)}" y2="${padT + plotH}" stroke="#e879f9" stroke-width="1.8"/>` : "";

      const legendHtml = lines.map(l => (
        `<span style="display:inline-flex; align-items:center; gap:4px;"><span style="display:inline-block; width:12px; height:2px; background:${l.color}; ${l.dashed ? 'border-top:2px dashed ' + l.color + '; background:transparent;' : ''}"></span>${escapeHtml(l.label)}</span>`
      )).join("");

      return `
        <div style="margin-bottom:12px; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:rgba(255,255,255,0.015);">
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; font-size:12px;">
            <strong style="color:var(--text);">${escapeHtml(title)} <span style="color:var(--muted); font-weight:400;">(${escapeHtml(unit)})</span></strong>
            <div style="display:flex; flex-wrap:wrap; gap:10px; font-size:11px; color:var(--muted);">
              ${legendHtml}
              ${oldCf != null ? `<span style="color:#fbbf24;">┆ 原触球 ${oldCf}F</span>` : ""}
              ${newCf != null ? `<span style="color:#e879f9;">│ 校准触球 ${newCf}F</span>` : ""}
            </div>
          </div>
          <svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; display:block;">
            ${prepBandHtml}
            ${gridHtml}
            ${pathsHtml}
            ${oldLine}
            ${newLine}
            ${markersHtml}
            <text x="${padL}" y="${H - 7}" fill="#8d9caf" font-size="9.5" text-anchor="start">帧 ${fMin}</text>
            <text x="${W / 2}" y="${H - 7}" fill="#8d9caf" font-size="9.5" text-anchor="middle">帧 ${Math.round((fMin + fMax) / 2)}</text>
            <text x="${W - padR}" y="${H - 7}" fill="#8d9caf" font-size="9.5" text-anchor="end">帧 ${fMax}</text>
          </svg>
        </div>
      `;
    }

    async function openTrajectoryModal(sessionId, eventId) {
      const modal = document.getElementById("trajectory-modal");
      const titleEl = document.getElementById("traj-modal-title");
      const subEl = document.getElementById("traj-modal-subtitle");
      const body = document.getElementById("traj-modal-body");
      if (!modal || !body) return;

      const sidePool = (globalProfileData?.[activeStrokeSide()]?.swings) || [];
      activeTrajectorySwingList = sidePool.map(s => ({ session_id: s.session_id, event_id: s.event_id }));
      activeTrajectoryIndex = activeTrajectorySwingList.findIndex(
        x => x.session_id === sessionId && Number(x.event_id) === Number(eventId)
      );
      const prevBtn = document.getElementById("traj-prev-btn");
      const nextBtn = document.getElementById("traj-next-btn");
      if (prevBtn) prevBtn.disabled = activeTrajectoryIndex <= 0;
      if (nextBtn) nextBtn.disabled = activeTrajectoryIndex < 0 || activeTrajectoryIndex >= activeTrajectorySwingList.length - 1;

      modal.hidden = false;
      if (titleEl) titleEl.textContent = `📈 Event #${eventId} 逐帧轨迹拟合与修复观测`;
      body.textContent = "正在读取逐帧骨架与球拍轨迹并执行 PCHIP + Savitzky-Golay 拟合...";

      try {
        const qs = new URLSearchParams({ session_id: sessionId, event_id: String(eventId) });
        const res = await api(`/api/session/swing-trajectory?${qs.toString()}`);
        const d = await res.json();
        if (!d.available || !d.series) {
          body.innerHTML = `<div style="padding:24px; text-align:center; color:var(--muted);">${escapeHtml(d.reason || "暂无逐帧轨迹数据")}</div>`;
          return;
        }
        const tierBadge = d.tier === "A"
          ? `<span style="padding:2px 7px; border-radius:4px; background:rgba(74,222,128,0.18); color:#4ade80; font-size:11px; font-weight:600;">Tier A 高置信 (权重 1.0)</span>`
          : (d.tier === "B"
            ? `<span style="padding:2px 7px; border-radius:4px; background:rgba(251,191,36,0.18); color:#fbbf24; font-size:11px; font-weight:600;">Tier B 插值降权 (权重 0.5)</span>`
            : `<span style="padding:2px 7px; border-radius:4px; background:rgba(248,113,113,0.18); color:#f87171; font-size:11px; font-weight:600;">Tier C 误切残缺 (权重 0)</span>`);

        if (subEl) {
          subEl.innerHTML = `${escapeHtml(d.display_time || d.session_id)} · ${escapeHtml(d.stroke_type || "")} · 门控剔除 ${d.rejected_count} 帧 · PCHIP 插值 ${d.interpolated_count} 帧 · 触球校准偏移 ${d.shift_frames >= 0 ? "+" : ""}${d.shift_frames} 帧`;
        }

        const sm = d.summary || {};
        const fmtPair = (entry, unit) => {
          const r1 = entry?.raw != null ? `${entry.raw}${unit}` : "—";
          const r2 = entry?.refined != null ? `${entry.refined}${unit}` : "—";
          return `<span style="color:var(--muted);">${r1}</span> → <strong style="color:#38bdf8;">${r2}</strong>`;
        };
        const summaryBar = `
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; padding:9px 12px; margin-bottom:12px; border-radius:9px; background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.22); font-size:12px;">
            <div style="display:flex; flex-wrap:wrap; gap:14px; align-items:center;">
              ${tierBadge}
              <span>触球肘角: ${fmtPair(sm.arm_extension, "°")}</span>
              <span>准备屈膝: ${fmtPair(sm.preparation_knee_flexion, "°")}</span>
              <span>转肩幅度: ${fmtPair(sm.shoulder_turn_change, "°")}</span>
              <span>球拍峰值: ${fmtPair(sm.racket_max_speed, " px/s")}</span>
            </div>
            <span style="font-size:11px; color:var(--muted);">红×=门控剔除误检 · 绿点=PCHIP插值 · 绿区=固定蓄力窗</span>
          </div>
        `;

        const s = d.series;
        const chartSpeed = buildFrameCurveSvg({
          title: "1. 球拍速度逐帧拟合曲线（Hampel+手腕门控 → PCHIP插值 → SG平滑）",
          unit: "px/s",
          frameIds: s.frame_ids,
          lines: [
            { label: "原始速度 (截断6500)", values: s.raw_racket_speed, color: "#64748b", dashed: true, width: 1.2, opacity: 0.85 },
            { label: "拟合平滑速度", values: s.smooth_racket_speed, color: "#38bdf8", width: 2.3 },
          ],
          oldCf: d.old_contact_frame,
          newCf: d.refined_contact_frame,
          prepWindow: null,
          rejectedMask: s.rejected_mask,
          interpMask: s.interp_mask,
          markerSeries: { raw: s.raw_racket_speed, smooth: s.smooth_racket_speed },
        });

        const chartJoints = buildFrameCurveSvg({
          title: "2. 持拍侧肘伸展角 & 准备期屈膝角逐帧拟合曲线（绿区为固定触球前蓄力窗）",
          unit: "°",
          frameIds: s.frame_ids,
          lines: [
            { label: "原始肘角", values: s.raw_arm, color: "#94a3b8", dashed: true, width: 1.1, opacity: 0.65 },
            { label: "平滑肘角", values: s.smooth_arm, color: "#38bdf8", width: 2.1 },
            { label: "原始屈膝", values: s.raw_knee, color: "#86efac", dashed: true, width: 1.1, opacity: 0.55 },
            { label: "平滑屈膝", values: s.smooth_knee, color: "#4ade80", width: 2.1 },
          ],
          oldCf: d.old_contact_frame,
          newCf: d.refined_contact_frame,
          prepWindow: d.prep_window_frames,
        });

        const chartShoulder = buildFrameCurveSvg({
          title: "3. 肩部连线转角逐帧拟合曲线",
          unit: "°",
          frameIds: s.frame_ids,
          lines: [
            { label: "原始肩角", values: s.raw_shoulder, color: "#94a3b8", dashed: true, width: 1.1, opacity: 0.65 },
            { label: "平滑肩角", values: s.smooth_shoulder, color: "#c084fc", width: 2.1 },
          ],
          oldCf: d.old_contact_frame,
          newCf: d.refined_contact_frame,
          prepWindow: d.prep_window_frames,
        });

        body.innerHTML = summaryBar + chartSpeed + chartJoints + chartShoulder;
      } catch (err) {
        body.innerHTML = `<div style="padding:24px; text-align:center; color:#f87171;">加载单拍拟合曲线失败：${escapeHtml(err.message)}</div>`;
      }
    }

    function closeTrajectoryModal() {
      const modal = document.getElementById("trajectory-modal");
      if (modal) modal.hidden = true;
    }

    function stepTrajectoryModal(delta) {
      const nextIdx = activeTrajectoryIndex + delta;
      if (nextIdx < 0 || nextIdx >= activeTrajectorySwingList.length) return;
      const target = activeTrajectorySwingList[nextIdx];
      openTrajectoryModal(target.session_id, target.event_id);
    }

    async function initPlayerComparePage() {
      const params = new URLSearchParams(window.location.search);
      const initialPlayer = params.get("player_id") || "";
      const initialStroke = params.get("stroke_type") || "Forehand";
      const initialMode = params.get("mode") || "forehand";
      const initialDays = params.get("days") || "all";
      const initialDedup = params.get("dedup");
      dedupEnabled = initialDedup !== "0" && initialDedup !== "false";

      const dedupChk = document.getElementById("compare-dedup-toggle");
      if (dedupChk) dedupChk.checked = dedupEnabled;

      const strokeSel = document.getElementById("compare-stroke-select");
      if (strokeSel && ["all", "Forehand", "Backhand"].includes(initialStroke)) {
        strokeSel.value = initialStroke;
      }
      const daysSel = document.getElementById("compare-days-select");
      if (daysSel && ["all", "30", "7"].includes(initialDays)) {
        daysSel.value = initialDays;
      }

      try {
        const cfgRes = await api("/api/config");
        const cfg = await cfgRes.json();
        compareToken = cfg.token || "";
      } catch (_) {}

      try {
        await loadComparePlayers(initialPlayer);
      } catch (err) {
        showToast(`加载学员列表失败：${err.message}`, true);
      }

      const fhBtn = document.getElementById("mode-forehand-btn");
      const bhBtn = document.getElementById("mode-backhand-btn");
      const abBtn = document.getElementById("mode-ab-btn");
      if (fhBtn) fhBtn.addEventListener("click", () => setPageViewMode("forehand"));
      if (bhBtn) bhBtn.addEventListener("click", () => setPageViewMode("backhand"));
      if (abBtn) abBtn.addEventListener("click", () => setPageViewMode("ab"));

      const playerSel = document.getElementById("compare-player-select");
      if (playerSel && !playerSel.dataset.bound) {
        playerSel.dataset.bound = "true";
        playerSel.addEventListener("change", () => {
          syncUrlParams();
          loadActiveViewData();
        });
      }
      if (daysSel && !daysSel.dataset.bound) {
        daysSel.dataset.bound = "true";
        daysSel.addEventListener("change", () => {
          syncUrlParams();
          loadGlobalProfile();
        });
      }
      if (strokeSel && !strokeSel.dataset.bound) {
        strokeSel.dataset.bound = "true";
        strokeSel.addEventListener("change", () => {
          syncUrlParams();
          loadPlayerComparison();
        });
      }
      const addBtn = document.getElementById("add-player-btn");
      if (addBtn && !addBtn.dataset.bound) {
        addBtn.dataset.bound = "true";
        addBtn.addEventListener("click", createNewComparePlayer);
      }
      const batchBtn = document.getElementById("batch-assign-btn");
      if (batchBtn && !batchBtn.dataset.bound) {
        batchBtn.dataset.bound = "true";
        batchBtn.addEventListener("click", openBatchAssignModal);
      }
      const dedupBtn = document.getElementById("dedup-manage-btn");
      if (dedupBtn && !dedupBtn.dataset.bound) {
        dedupBtn.dataset.bound = "true";
        dedupBtn.addEventListener("click", openDedupModal);
      }
      if (dedupChk && !dedupChk.dataset.bound) {
        dedupChk.dataset.bound = "true";
        dedupChk.addEventListener("change", () => {
          dedupEnabled = Boolean(dedupChk.checked);
          syncUrlParams();
          loadGlobalProfile();
        });
      }
      const refreshBtn = document.getElementById("refresh-compare-btn");
      if (refreshBtn && !refreshBtn.dataset.bound) {
        refreshBtn.dataset.bound = "true";
        refreshBtn.addEventListener("click", async () => {
          await loadComparePlayers();
          await loadActiveViewData();
        });
      }

      const resolvedMode = initialMode === "ab" ? "ab" : (initialMode === "backhand" ? "backhand" : "forehand");
      setPageViewMode(resolvedMode);
    }

    if (typeof window !== "undefined") {
      window.setGlobalGridPreset = setGlobalGridPreset;
      window.jumpGridToConflictSessions = jumpGridToConflictSessions;
      window.setGlobalCropMode = setGlobalCropMode;
      window.applyGridMediaTransform = applyGridMediaTransform;
      window.nudgeGridFocus = nudgeGridFocus;
      window.onChangeGridSlotSession = onChangeGridSlotSession;
      window.onChangeGridSlotSwing = onChangeGridSlotSwing;
      window.syncGridPlayClips = syncGridPlayClips;
      window.syncGridPauseClips = syncGridPauseClips;
      window.syncGridSeekContact = syncGridSeekContact;
      window.syncGridStepFrames = syncGridStepFrames;
      window.setGlobalTrendMetric = setGlobalTrendMetric;
      window.onChangeGlobalAiModel = onChangeGlobalAiModel;
      window.generateGlobalAiReview = generateGlobalAiReview;
      window.onChangeCompareAB = onChangeCompareAB;
      window.setCompareCropMode = setCompareCropMode;
      window.syncSeekABContact = syncSeekABContact;
      window.syncPlayABClips = syncPlayABClips;
      window.syncPauseABClips = syncPauseABClips;
      window.updateCompareSwingSlot = updateCompareSwingSlot;
      window.nudgeCompareFocus = nudgeCompareFocus;
      window.applyCompareMediaTransform = applyCompareMediaTransform;
      window.openSessionInConsole = openSessionInConsole;
      window.openBatchAssignModal = openBatchAssignModal;
      window.closeBatchAssignModal = closeBatchAssignModal;
      window.selectAllBatchRows = selectAllBatchRows;
      window.toggleBatchRow = toggleBatchRow;
      window.renderBatchSessionRows = renderBatchSessionRows;
      window.submitBatchAssign = submitBatchAssign;
      window.openDedupModal = openDedupModal;
      window.closeDedupModal = closeDedupModal;
      window.loadDedupGroups = loadDedupGroups;
      window.toggleDedupRow = toggleDedupRow;
      window.selectRecommendedDuplicates = selectRecommendedDuplicates;
      window.submitDedupCleanup = submitDedupCleanup;
      window.setGlobalTrendViewMode = setGlobalTrendViewMode;
      window.selectRadarSwing = selectRadarSwing;
      window.stepRadarSwing = stepRadarSwing;
      window.loadSwingIntoGridSlot = loadSwingIntoGridSlot;
      window.openTrajectoryModal = openTrajectoryModal;
      window.closeTrajectoryModal = closeTrajectoryModal;
      window.stepTrajectoryModal = stepTrajectoryModal;

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initPlayerComparePage);
      } else {
        initPlayerComparePage();
      }
    }
