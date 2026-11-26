    const ids = [
      "stream_id", "mapped_stream_id", "custom_stream_source", "username", "password", "crop_margin", "output_fps",
      "roi_enabled", "show_roi_boundary", "show_roi_fill", "show_roi_points",
      "analysis_interval", "settle_frames", "max_suggestions", "min_confidence",
      "realtime_swing_events", "realtime_coach", "coach_tts", "coach_tts_playback", "deepseek_coach",
      "realtime_frame_output", "evidence_bundle", "display_origin_x",
      "display_origin_y", "save_video", "hdmi_output", "algo2_dual_view",
      "output_dir", "session_name", "analysis_mode", "pose_model"
    ];
    const checkboxIds = new Set([
      "roi_enabled", "show_roi_boundary", "show_roi_fill", "show_roi_points",
      "realtime_swing_events", "realtime_coach", "coach_tts", "coach_tts_playback", "deepseek_coach",
      "realtime_frame_output", "evidence_bundle", "save_video", "hdmi_output", "algo2_dual_view"
    ]);
    const CUSTOM_STREAM_ID = "custom";
    const SIDEBAR_COLLAPSED_KEY = "tennis_ai_sidebar_collapsed_v1";
    const state = { token: "", streams: [], status: "stopped", previewUrl: null };
    const $ = id => document.getElementById(id);

    function setSidebarCollapsed(collapsed, { persist = true } = {}) {
      const isCollapsed = Boolean(collapsed);
      const layout = $("workspace-layout");
      const sidebar = $("settings-sidebar");
      const toggleBtn = $("toggle-sidebar-btn");
      if (layout) layout.classList.toggle("sidebar-collapsed", isCollapsed);
      if (sidebar) sidebar.setAttribute("aria-hidden", isCollapsed ? "true" : "false");
      if (toggleBtn) {
        toggleBtn.textContent = isCollapsed ? "显示侧边栏" : "隐藏侧边栏";
        toggleBtn.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
      }
      if (persist) {
        try {
          localStorage.setItem(SIDEBAR_COLLAPSED_KEY, isCollapsed ? "1" : "0");
        } catch {}
      }
      if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
      }
    }

    const SESSION_SUFFIX_RE = /(?:_\d{8}_\d{6}(?:_[a-z0-9]{2,4})?|_\d{8}T\d{6}Z_[a-z0-9]{4,8})$/i;
    let lastGeneratedSessionName = "";
    let lastSubmittedSessionName = "";

    function stripSessionSuffix(raw) {
      const base = String(raw || "").trim().replace(SESSION_SUFFIX_RE, "").replace(/_+$/g, "");
      return base || "live_session";
    }

    function buildSessionSuffix(date = new Date()) {
      const pad = n => String(n).padStart(2, "0");
      return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
    }

    function ensureUniqueSessionName(raw, { forceRefresh = false } = {}) {
      const cleaned = String(raw || "").trim();
      if (!forceRefresh && cleaned && SESSION_SUFFIX_RE.test(cleaned) && cleaned !== lastSubmittedSessionName) {
        return cleaned.slice(0, 64);
      }
      const base = stripSessionSuffix(cleaned);
      const suffix = buildSessionSuffix();
      const maxBaseLen = Math.max(1, 64 - suffix.length - 1);
      let candidate = `${base.slice(0, maxBaseLen)}_${suffix}`;
      if (candidate === lastGeneratedSessionName || candidate === lastSubmittedSessionName || candidate === cleaned) {
        const tie = Math.random().toString(36).slice(2, 5);
        const tieMaxBase = Math.max(1, 64 - suffix.length - tie.length - 2);
        candidate = `${base.slice(0, tieMaxBase)}_${suffix}_${tie}`;
      }
      lastGeneratedSessionName = candidate;
      return candidate;
    }

    function showToast(message, error = false) {
      const toast = $("toast");
      toast.textContent = message;
      toast.className = "toast visible" + (error ? " error" : "");
      clearTimeout(showToast.timer);
      showToast.timer = setTimeout(() => toast.className = "toast", 3600);
    }

    async function api(path, options = {}) {
      const headers = { ...(options.headers || {}) };
      if (options.method === "POST") {
        headers["Content-Type"] = headers["Content-Type"] || "application/json";
        headers["X-Control-Token"] = state.token;
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

    function formPayload() {
      const payload = {};
      if ($("stream_id").value === "local_video") payload.video_id = state.videoId || "";
      for (const id of ids) {
        const element = $(id);
        if (!element) continue;
        payload[id] = checkboxIds.has(id) ? element.checked : element.value;
      }
      if ($("stream_id").value !== "local_video") {
        payload.mapped_stream_id = "";
      }
      return payload;
    }

    function savePreset() {
      const payload = formPayload();
      delete payload.password;
      delete payload.username;
      delete payload.custom_stream_source;
      delete payload.video_id;
      delete payload.output_dir;
      if (payload.stream_id === "local_video") delete payload.stream_id;
      if (payload.stream_id === CUSTOM_STREAM_ID) delete payload.stream_id;
      if (payload.stream_id !== "local_video") delete payload.mapped_stream_id;
      localStorage.setItem("tennis-control-preset-v1", JSON.stringify(payload));
    }

    function applyValues(values) {
      for (const [id, value] of Object.entries(values || {})) {
        const element = $(id);
        if (!element || value === undefined || value === null) continue;
        if (checkboxIds.has(id)) element.checked = Boolean(value);
        else element.value = value;
      }
    }

    function selectedStream() {
      const inputId = $("stream_id").value;
      const cameraId = inputId === "local_video" ? $("mapped_stream_id")?.value : inputId;
      return state.streams.find(item => item.stream_id === cameraId);
    }

    function customPublicSource() {
      const value = $("custom_stream_source").value.trim();
      if (!value) return "输入地址后可刷新预览或启动分析";
      try {
        const parsed = new URL(value);
        parsed.username = "";
        parsed.password = "";
        parsed.search = "";
        parsed.hash = "";
        return parsed.toString();
      } catch (_) {
        return value.split(/[?#]/, 1)[0];
      }
    }

    function syncEngineDisplay() {
      const modeVal = $("analysis_mode")?.value || "standard";
      const poseVal = $("pose_model")?.value || "yolo26n-pose.mlpackage";
      const isVideo = $("stream_id")?.value === "local_video";

      const modeHint = $("analysis-mode-hint");
      if (modeHint) {
        if (modeVal === "standard") {
          modeHint.textContent = isVideo
            ? "本地视频将自动全速离线处理：启用 640 七点球拍并行叠加（共享源帧），事件与 Coach 由 960 主链路稳健输出。"
            : "实时码流将自动连续不丢帧处理：启用 640 七点球拍并行叠加（共享源帧），事件与 Coach 由 960 主链路稳健输出。";
        } else {
          modeHint.textContent = isVideo
            ? "本地视频将自动全速离线处理并启用多进程 ANE 检测池：仅运行 960 主链路（不加载七点球拍），吞吐更高。"
            : "实时码流将自动连续不丢帧处理：仅运行 960 主链路（不加载七点球拍），25FPS 实时延迟最低。";
        }
      }

      const poseHint = $("pose-model-hint");
      if (poseHint) {
        poseHint.textContent = poseVal.includes("yolo26m")
          ? "高精度 yolo26m-pose：关节稳定性更高，适合离线分析或低并发场景。"
          : "25FPS 实时码流推荐 yolo26n-pose（单帧推理省约 10ms，余量更充足）。";
      }

      const summaryEl = $("active-models-summary");
      if (summaryEl) {
        const detLabel = state.detectorModel?.label || "exp004-960";
        const poseLabel = poseVal.includes("yolo26m") ? "yolo26m-pose" : "yolo26n-pose";
        const racketLabel = modeVal === "standard" ? "七点 640（并行增强）" : "960 主检测（未启七点）";
        summaryEl.textContent = `球检测：${detLabel} · 姿态：${poseLabel} · 球拍：${racketLabel}`;
      }
    }

    function syncConditionalFields() {
      const streamValue = $("stream_id")?.value || "";
      const video = streamValue === "local_video";
      const custom = streamValue === CUSTOM_STREAM_ID;
      const configured = custom ? state.defaultCredentialsConfigured :
        (state.cameraCredentialsConfigured || {})[streamValue];

      const credDetails = $("credentials-details");
      if (credDetails) {
        credDetails.hidden = video;
        if (custom && !configured) credDetails.open = true;
      }

      const ttsPlaybackField = $("tts-playback-field");
      if (ttsPlaybackField) {
        ttsPlaybackField.hidden = !Boolean($("coach_tts")?.checked);
      }

      const hdmiOriginFields = $("hdmi-origin-fields");
      if (hdmiOriginFields) {
        hdmiOriginFields.hidden = !Boolean($("hdmi_output")?.checked);
      }
    }

    function updateStreamMeta() {
      syncEngineDisplay();
      syncConditionalFields();
      const video = $("stream_id").value === "local_video";
      $("local-video-field").hidden = !video;
      $("username").disabled = video;
      $("password").disabled = video;
      const custom = $("stream_id").value === CUSTOM_STREAM_ID;
      const configured = custom ? state.defaultCredentialsConfigured :
        (state.cameraCredentialsConfigured || {})[$("stream_id").value];
      $("username").placeholder = configured ? "当前通道已配置，留空自动使用" : "留空使用本机配置；可临时覆盖";
      $("password").placeholder = configured ? "当前通道已配置，无需重复输入" : "不在浏览器中保存";
      $("custom-stream-field").hidden = !custom;
      $("custom_stream_source").required = custom;
      if (video) {
        const mappedId = $("mapped_stream_id") ? $("mapped_stream_id").value : "";
        const mappedStream = mappedId ? state.streams.find(item => item.stream_id === mappedId) : null;
        $("roi_enabled").disabled = !mappedStream;
        if (mappedStream) {
          $("stream-meta").innerHTML = `<strong>本地视频</strong><span>映射机位：${escapeHtml(mappedStream.label)} (${state.videoName || "已选择文件"})</span>`;
          $("point-list").innerHTML = (mappedStream.points || []).map(
            (point, index) => `<span>P${index + 1}: [${point[0]}, ${point[1]}]</span>`
          ).join("") || "<span>已绑定机位，但该机位未配置有效ROI</span>";
        } else {
          $("stream-meta").textContent = "本地视频 · " + (state.videoName || "请先选择视频文件");
          $("point-list").textContent = "全画面离线完整分析 · 未绑定机位 ROI";
        }
        updateRoiBoundaryEditor(mappedStream);
        savePreset();
        return;
      }
      if ($("mapped_stream_id")) {
        $("mapped_stream_id").value = "";
      }
      $("roi_enabled").disabled = false;
      if (custom) {
        $("stream-meta").innerHTML = `<strong>自定义码流</strong><span>${escapeHtml(customPublicSource())}</span>`;
        $("point-list").innerHTML = "<span>ROI 将按码流地址自动匹配</span>";
        savePreset();
        return;
      }
      const stream = selectedStream();
      const editor = $("roi-boundary-editor");
      if (editor) editor.hidden = video || custom;
      if (!stream) return;
      $("stream-meta").innerHTML = `<strong>${escapeHtml(stream.label)}</strong><span>${escapeHtml(stream.source)}</span>`;
      const hasMirror = Boolean(stream.mirror_view && stream.mirror_view.polygon && stream.mirror_view.polygon.length >= 3);
      $("point-list").innerHTML = (stream.points || []).map(
        (point, index) => `<span>P${index + 1}: [${point[0]}, ${point[1]}]</span>`
      ).join("") + (hasMirror ? `<span style="color: #ff6e73; border-color: rgba(255, 110, 115, 0.4);">🛡️ 镜面排除已激活 (阻断镜中虚像)</span>` : "");

      updateRoiBoundaryEditor(stream);
      savePreset();
    }

    function updateRoiBoundaryEditor(stream) {
      const editor = $("roi-boundary-editor");
      if (editor) editor.hidden = !stream;
      if (!stream) {
        for (const id of ["roi_p1", "roi_p2", "roi_p3", "roi_p4", "mirror_m1", "mirror_m2", "mirror_m3", "mirror_m4"]) {
          if ($(id)) $(id).value = "";
        }
        if ($("roi-mirror-tag")) $("roi-mirror-tag").textContent = "未绑定机位";
        roiEditorState.courtPoints = [];
        roiEditorState.mirrorPoints = [];
        if (roiEditorState.active) drawRoiCanvas();
        return;
      }
      const hasMirror = Boolean(stream.mirror_view?.polygon?.length >= 3);
      const pts = (typeof sortPointsTLTRBRBL === "function" && stream.points && stream.points.length === 4)
        ? sortPointsTLTRBRBL(stream.points) : (stream.points || []);
      if ($("roi_p1")) $("roi_p1").value = pts[0] ? `${pts[0][0]}, ${pts[0][1]}` : "";
      if ($("roi_p2")) $("roi_p2").value = pts[1] ? `${pts[1][0]}, ${pts[1][1]}` : "";
      if ($("roi_p3")) $("roi_p3").value = pts[2] ? `${pts[2][0]}, ${pts[2][1]}` : "";
      if ($("roi_p4")) $("roi_p4").value = pts[3] ? `${pts[3][0]}, ${pts[3][1]}` : "";

      const mPoly = stream.mirror_view?.polygon || [];
      const [sw, sh] = (typeof getSourceDimensions === "function") ? getSourceDimensions() : [2560, 1440];
      const mSrcRaw = mPoly.map(p => [Math.round(p[0] * sw), Math.round(p[1] * sh)]);
      const mSrcPts = (typeof sortPointsTLTRBRBL === "function" && mSrcRaw.length === 4)
        ? sortPointsTLTRBRBL(mSrcRaw) : mSrcRaw;

      if ($("mirror_m1")) $("mirror_m1").value = mSrcPts[0] ? `${mSrcPts[0][0]}, ${mSrcPts[0][1]}` : "";
      if ($("mirror_m2")) $("mirror_m2").value = mSrcPts[1] ? `${mSrcPts[1][0]}, ${mSrcPts[1][1]}` : "";
      if ($("mirror_m3")) $("mirror_m3").value = mSrcPts[2] ? `${mSrcPts[2][0]}, ${mSrcPts[2][1]}` : "";
      if ($("mirror_m4")) $("mirror_m4").value = mSrcPts[3] ? `${mSrcPts[3][0]}, ${mSrcPts[3][1]}` : "";

      if ($("roi_mirror_exclusion") && stream.mirror_view) {
        $("roi_mirror_exclusion").checked = stream.mirror_view.exclusion_enabled !== false;
      }
      if ($("roi-mirror-tag")) $("roi-mirror-tag").textContent = hasMirror ? "🛡️ 镜面排除已启用" : "未配置镜面";

      if (typeof roiEditorState !== "undefined" && roiEditorState.active && typeof initRoiEditorFromStream === "function") {
        initRoiEditorFromStream(stream);
        syncCanvasGeometry();
      }
    }

    function escapeHtml(value) {
      const div = document.createElement("div");
      div.textContent = String(value ?? "");
      return div.innerHTML;
    }

    function applyPreviewBlob(blob) {
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = URL.createObjectURL(blob);
      const image = $("preview-image");
      image.src = state.previewUrl;
      image.hidden = false;
      $("preview-empty").hidden = true;
      image.onload = () => {
        if (typeof roiEditorState !== "undefined" && roiEditorState.active) {
          syncCanvasGeometry();
        }
      };
    }

    async function refreshPreview(manual = true, clean = null) {
      const settingsForm = $("settings-form");
      if (typeof settingsForm.checkValidity === "function" && !settingsForm.checkValidity() && typeof setSidebarCollapsed === "function") {
        setSidebarCollapsed(false, { persist: false });
      }
      if (!settingsForm.reportValidity()) return false;
      const requestId = state.previewRequestId = (state.previewRequestId || 0) + 1;
      const payload = formPayload();
      const shouldClean = (clean !== null) ? Boolean(clean) : Boolean(typeof roiEditorState !== "undefined" && roiEditorState.active);
      if (shouldClean) {
        payload.clean = true;
      }
      $("preview-loading").classList.add("visible");
      $("preview-button").disabled = true;
      try {
        const response = await api("/api/preview", {
          method: "POST",
          body: JSON.stringify(payload)
        });
        const blob = await response.blob();
        if (requestId !== state.previewRequestId) return false;
        applyPreviewBlob(blob);
        // Wait for the source image to replace the last analysis composite
        // before allowing the editor's original-pixel canvas to become visible.
        const image = $("preview-image");
        if (image.decode) await image.decode();
        if (requestId !== state.previewRequestId) return false;
        if (manual) showToast(shouldClean ? "画面已加载 (纯净无标注模式)" : "ROI 预览已更新");
        return true;
      } catch (error) {
        showToast(`预览失败：${error.message}`, true);
        return false;
      } finally {
        if (requestId === state.previewRequestId) {
          $("preview-loading").classList.remove("visible");
          $("preview-button").disabled = false;
        }
      }
    }

    async function refreshLivePreview() {
      const editing = () => typeof roiEditorState !== "undefined"
        && (roiEditorState.active || roiEditorState.loading);
      if (editing()) return;
      if (refreshLivePreview.pending) return;
      const requestId = state.previewRequestId || 0;
      refreshLivePreview.pending = true;
      try {
        const response = await fetch(
          `/api/live-preview?t=${Date.now()}`,
          { cache: "no-store" }
        );
        if (response.ok) {
          const blob = await response.blob();
          // A response dispatched before editor entry can arrive after the
          // source frame. Its dimensions describe the SBS output, not the ROI.
          if (!editing() && requestId === (state.previewRequestId || 0)) {
            applyPreviewBlob(blob);
          }
        }
      } finally {
        refreshLivePreview.pending = false;
      }
    }

    async function startAnalysis() {
      if ($("stream_id").value === "local_video" && !state.videoId) {
        showToast("请先选择视频并等待上传完成", true);
        return;
      }
      const settingsForm = $("settings-form");
      if (typeof settingsForm.checkValidity === "function" && !settingsForm.checkValidity() && typeof setSidebarCollapsed === "function") {
        setSidebarCollapsed(false, { persist: false });
      }
      if (!settingsForm.reportValidity()) return;
      const sessionInput = $("session_name");
      if (sessionInput) {
        sessionInput.value = ensureUniqueSessionName(sessionInput.value, { forceRefresh: false });
      }
      savePreset();
      $("start-button").disabled = true;
      try {
        const startPayload = formPayload();
        const playerSel = document.getElementById("session-player-select");
        if (playerSel && playerSel.value) {
          startPayload.player_id = playerSel.value;
        }
        const response = await api("/api/start", {
          method: "POST",
          body: JSON.stringify(startPayload)
        });
        updateStatus(await response.json());
        if (sessionInput) {
          lastSubmittedSessionName = startPayload.session_name || sessionInput.value;
          sessionInput.value = ensureUniqueSessionName(sessionInput.value, { forceRefresh: true });
          savePreset();
        }
        showToast("分析进程已启动");
      } catch (error) {
        showToast(`启动失败：${error.message}`, true);
      } finally {
        await pollStatus();
      }
    }

    async function stopAnalysis() {
      $("stop-button").disabled = true;
      try {
        const response = await api("/api/stop", {
          method: "POST",
          body: "{}"
        });
        updateStatus(await response.json());
        showToast("已发送安全停止信号");
      } catch (error) {
        showToast(`停止失败：${error.message}`, true);
      }
    }

    function stateLabel(value) {
      return {
        stopped: "已停止",
        running: "运行中",
        stopping: "正在停止",
        failed: "运行失败"
      }[value] || value;
    }

    function updateStatus(status) {
      state.status = status.state;
      const modelStatus = $("running-racket-model");
      if (modelStatus && status.racket_model) {
        const model = status.racket_model;
        modelStatus.textContent = model.mode === "legacy"
          ? "最近启动：轻量极速版（仅 960 主链路）"
          : `最近启动：标准增强版（七点球拍 ${model.imgsz}×${model.imgsz} / ${model.compute_units}）`;
      }
      const pill = $("status-pill");
      pill.textContent = stateLabel(status.state);
      pill.className = `status-pill ${status.state}`;
      const running = status.state === "running";
      const stopping = status.state === "stopping";
      $("start-button").disabled = running || stopping;
      $("stop-button").disabled = !running;
      $("pid-value").textContent = status.pid ?? "—";
      $("elapsed-value").textContent = `${Math.round(status.elapsed_seconds || 0)}s`;
      $("return-value").textContent = status.returncode ?? "—";
      $("process-description").textContent = running
        ? "实时处理与事件分析正在运行"
        : stopping ? "等待 Pipeline 安全释放资源" : "等待启动";
      const logs = (status.logs || []).join("\n") || "控制台日志将在这里显示。";
      const log = $("log-output");
      const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 42;
      log.textContent = logs;
      if (nearBottom) log.scrollTop = log.scrollHeight;
      const report = $("report-link");
      if (status.artifacts?.report_url) {
        report.href = status.artifacts.report_url;
        report.hidden = false;
      } else {
        report.hidden = true;
      }
      const evidence = $("evidence-link");
      if (status.artifacts?.evidence_manifest_url) {
        evidence.href = status.artifacts.evidence_manifest_url;
        evidence.hidden = false;
      } else {
        evidence.hidden = true;
      }
      if (running) {
        refreshLivePreview();
      }
    }


    function handleLogsData(data) {
      const log = $("log-output");
      if (!log || !data) return;
      const newLines = data.logs || [];
      if (!newLines.length) return;
      const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
      if (log.textContent === "控制台日志将在这里显示。" || (data.total && data.total === newLines.length)) {
        log.textContent = newLines.join("\n");
      } else {
        log.textContent += (log.textContent ? "\n" : "") + newLines.join("\n");
      }
      if (nearBottom) log.scrollTop = log.scrollHeight;
    }

    let evtSource = null;
    let sseFallbackTimer = null;

    function setSSEStatus(type, text) {
      const badge = $("sse-status-badge");
      if (!badge) return;
      badge.className = `sse-badge ${type}`;
      badge.textContent = text;
    }

    function initSSE() {
      if (evtSource) {
        try { evtSource.close(); } catch (_) {}
        evtSource = null;
      }
      setSSEStatus("connecting", "🟡 连接中…");

      try {
        evtSource = new EventSource("/api/events");
        evtSource.onopen = () => {
          setSSEStatus("live", "🟢 实时流 (SSE)");
          if (sseFallbackTimer) {
            clearInterval(sseFallbackTimer);
            sseFallbackTimer = null;
          }
        };

        evtSource.addEventListener("status", (e) => {
          try {
            const data = JSON.parse(e.data);
            updateStatus(data);
          } catch (err) {
            console.error("SSE status error:", err);
          }
        });

        evtSource.addEventListener("swings", (e) => {
          try {
            const data = JSON.parse(e.data);
            handleSwingsData(data);
          } catch (err) {
            console.error("SSE swings error:", err);
          }
        });

        evtSource.addEventListener("logs", (e) => {
          try {
            const data = JSON.parse(e.data);
            handleLogsData(data);
          } catch (err) {
            console.error("SSE logs error:", err);
          }
        });

        evtSource.onerror = () => {
          setSSEStatus("fallback", "⏳ 轮询回退 (Polling)");
          if (!sseFallbackTimer) {
            sseFallbackTimer = setInterval(pollStatus, 2000);
          }
        };
      } catch (err) {
        setSSEStatus("fallback", "⏳ 轮询回退 (Polling)");
        if (!sseFallbackTimer) {
          sseFallbackTimer = setInterval(pollStatus, 2000);
        }
      }
    }

    async function pollStatus() {
      try {
        const response = await api("/api/status");
        updateStatus(await response.json());
        await pollSwings();
      } catch (error) {
        showToast(`状态连接失败：${error.message}`, true);
      }
    }

    async function initialize() {
      try {
        const response = await api("/api/config");
        const config = await response.json();
        state.token = config.token;
        state.defaultCredentialsConfigured = config.credentials_configured;
        state.cameraCredentialsConfigured = config.camera_credentials_configured || {};
        state.streams = config.streams || [];
        $("stream_id").innerHTML = state.streams.map(
          stream => `<option value="${escapeHtml(stream.stream_id)}">${escapeHtml(stream.label)}</option>`
        ).join("") + `<option value="${CUSTOM_STREAM_ID}">自定义码流</option><option value="local_video">本地视频文件</option>`;
        if ($("mapped_stream_id")) {
          $("mapped_stream_id").innerHTML = `<option value="">不使用机位绑定 (全画幅分析)</option>` + state.streams.map(
            stream => `<option value="${escapeHtml(stream.stream_id)}">${escapeHtml(stream.label)} (绑定ROI与镜面)</option>`
          ).join("");
          $("mapped_stream_id").addEventListener("change", () => {
            updateStreamMeta();
            savePreset();
          });
        }
        state.detectorModel = config.detector_model || { label: "exp004-960" };
        applyValues(config.defaults);
        const model640Available = Boolean(config.racket_model_profiles?.["640_shadow"]?.available);
        const modeSelect = $("analysis_mode");
        if (modeSelect) {
          const stdOpt = modeSelect.querySelector('option[value="standard"]');
          if (stdOpt) stdOpt.disabled = !model640Available;
          if (!model640Available) modeSelect.value = "lite";
        }
        const poseSelect = $("pose_model");
        if (poseSelect && config.pose_models) {
          for (const key of ["yolo26n-pose.mlpackage", "yolo26m-pose.mlpackage"]) {
            const opt = poseSelect.querySelector(`option[value="${key}"]`);
            if (opt && config.pose_models[key] && config.pose_models[key].available === false) {
              opt.disabled = true;
            }
          }
        }
        try {
          const saved = JSON.parse(localStorage.getItem("tennis-control-preset-v1") || "null");
          if (saved) {
            if (saved.stream_id === CUSTOM_STREAM_ID) delete saved.stream_id;
            if (saved.stream_id !== "local_video") delete saved.mapped_stream_id;
            delete saved.output_dir;
            if (!localStorage.getItem("tennis-panel-v2")) {
              saved.analysis_mode = saved.racket_pose_mode === "legacy"
                ? "lite"
                : (config.defaults?.analysis_mode || "standard");
              if (!saved.pose_model || saved.pose_model === "configured") {
                saved.pose_model = config.defaults?.pose_model || "yolo26n-pose.mlpackage";
              }
            }
            for (const staleKey of [
              "racket_pose_mode", "racket_model_profile", "processing_mode",
              "inference_schedule", "racket_ipc", "live_mode", "inference_workers",
            ]) {
              delete saved[staleKey];
            }
            if (modeSelect && "analysis_mode" in saved) {
              const opt = saved.analysis_mode ? modeSelect.querySelector(`option[value="${saved.analysis_mode}"]`) : null;
              if (!opt || opt.disabled) delete saved.analysis_mode;
            }
            if (poseSelect && "pose_model" in saved) {
              const opt = saved.pose_model ? poseSelect.querySelector(`option[value="${saved.pose_model}"]`) : null;
              if (!opt || opt.disabled) delete saved.pose_model;
            }
            applyValues(saved);
          }
          localStorage.setItem("tennis-panel-v2", "1");
          localStorage.setItem("tennis-continuous-default-v1", "1");
        } catch (_) {}
        if (modeSelect && !modeSelect.value) {
          modeSelect.value = config.defaults?.analysis_mode || (model640Available ? "standard" : "lite");
        }
        if (poseSelect && !poseSelect.value) {
          poseSelect.value = config.defaults?.pose_model || "yolo26n-pose.mlpackage";
        }
        const sessionInput = $("session_name");
        if (sessionInput) {
          sessionInput.value = ensureUniqueSessionName(sessionInput.value, { forceRefresh: true });
        }
        updateStreamMeta();
        const filterValidBtn = $("filter-valid-btn");
        const filterAllBtn = $("filter-all-btn");
        if (filterValidBtn && filterAllBtn) {
          filterValidBtn.addEventListener("click", () => {
            if (filterOnlyValidSwings) return;
            filterOnlyValidSwings = true;
            filterValidBtn.classList.add("active");
            filterAllBtn.classList.remove("active");
            renderSwingsFeed(latestRawEvents);
          });
          filterAllBtn.addEventListener("click", () => {
            if (!filterOnlyValidSwings) return;
            filterOnlyValidSwings = false;
            filterAllBtn.classList.add("active");
            filterValidBtn.classList.remove("active");
            renderSwingsFeed(latestRawEvents);
          });
        }
        const refreshSwingsBtn = $("refresh-swings-btn");
        if (refreshSwingsBtn) {
          refreshSwingsBtn.addEventListener("click", () => {
            if (typeof loadSessionList === "function") loadSessionList();
            pollSwings(true);
          });
        }
        if (typeof loadSessionList === "function") await loadSessionList();
        await pollStatus();
        await pollSwings();
        initSSE();
      } catch (error) {
        showToast(`页面初始化失败：${error.message}`, true);
      }
    }

    $("local_video_file").addEventListener("change", async () => {
      const file = $("local_video_file").files[0];
      const generation = (state.uploadGeneration || 0) + 1;
      state.uploadGeneration = generation;
      state.videoId = "";
      state.videoName = "";
      updateStreamMeta();
      if (!file) return;
      $("video-upload-status").textContent = "正在上传到本机，请稍候…";
      try {
        if (!file.size || file.size > 2 * 1024 ** 3) throw new Error("请选择不超过 2 GB 的非空视频");
        const response = await api("/api/video/upload", {
          method: "POST", body: file,
          headers: { "Content-Type": "application/octet-stream", "X-Video-Suffix": "." + file.name.split(".").pop().toLowerCase() }
        });
        const result = await response.json();
        if (generation !== state.uploadGeneration) return;
        state.videoId = result.video_id;
        state.videoName = file.name;
        $("video-upload-status").textContent = `${file.name} 已就绪，可刷新预览或启动分析`;
        updateStreamMeta();
      } catch (error) {
        if (generation === state.uploadGeneration) $("video-upload-status").textContent = `上传失败：${error.message}`;
      }
    });

    function initControlPanel() {
      const pb = $("preview-button");
      if (pb) pb.addEventListener("click", () => refreshPreview(true));
      const ogc = $('open-ground-calibration');
      if (ogc) ogc.addEventListener('click', () => {
        const payload = formPayload(), query = new URLSearchParams();
        for (const key of ['stream_id','mapped_stream_id','video_id']) if (payload[key]) query.set(key, payload[key]);
        window.open('/ground-calibration?' + query, '_blank', 'noopener');
      });
      const sb = $("start-button");
      if (sb) sb.addEventListener("click", startAnalysis);
      const stp = $("stop-button");
      if (stp) stp.addEventListener("click", stopAnalysis);
      const sid = $("stream_id");
      if (sid) sid.addEventListener("change", updateStreamMeta);
      const analysisMode = $("analysis_mode");
      if (analysisMode) analysisMode.addEventListener("change", updateStreamMeta);
      const poseModel = $("pose_model");
      if (poseModel) poseModel.addEventListener("change", updateStreamMeta);
      const coachTts = $("coach_tts");
      if (coachTts) coachTts.addEventListener("change", syncConditionalFields);
      const hdmiOutput = $("hdmi_output");
      if (hdmiOutput) hdmiOutput.addEventListener("change", syncConditionalFields);
      const css = $("custom_stream_source");
      if (css) css.addEventListener("input", updateStreamMeta);
      const sf = $("settings-form");
      if (sf) {
        sf.addEventListener("change", savePreset);
        sf.addEventListener("submit", event => event.preventDefault());
      }
      const toggleSidebarBtn = $("toggle-sidebar-btn");
      if (toggleSidebarBtn) {
        let initialCollapsed = false;
        try {
          initialCollapsed = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
        } catch {}
        setSidebarCollapsed(initialCollapsed, { persist: false });
        toggleSidebarBtn.addEventListener("click", () => {
          const layout = $("workspace-layout");
          const currentlyCollapsed = Boolean(layout && layout.classList.contains("sidebar-collapsed"));
          setSidebarCollapsed(!currentlyCollapsed);
        });
      }
      const refreshSessionNameBtn = $("refresh-session-name-btn");
      if (refreshSessionNameBtn) {
        refreshSessionNameBtn.addEventListener("click", () => {
          const sessionInput = $("session_name");
          if (!sessionInput) return;
          sessionInput.value = ensureUniqueSessionName(sessionInput.value, { forceRefresh: true });
          savePreset();
        });
      }
      initialize();
    }

    if (typeof document !== "undefined") {
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initControlPanel);
      } else {
        initControlPanel();
      }
    }

