    // ==========================================
    // 📐 画面交互描点与边界编辑模块 (Interactive ROI Point Drawing & Editing)
    // ==========================================
    const roiEditorState = {
      active: false,
      target: "court", // "court" (球场ROI) | "mirror" (镜面排除区)
      mode: "drag", // "drag" (拖拽已有顶点) | "click" (重新顺序点击4点)
      courtPoints: [], // [[sourceX, sourceY], ...] 原视频像素坐标
      mirrorPoints: [], // [[sourceX, sourceY], ...] 原视频像素坐标
      clickStep: 0,
      hoverIdx: -1,
      draggingIdx: -1,
      mousePos: null,
    };

    function getSourceDimensions() {
      const stream = selectedStream();
      let sw = (stream && stream.frame_size && stream.frame_size[0]) ? stream.frame_size[0] : 0;
      let sh = (stream && stream.frame_size && stream.frame_size[1]) ? stream.frame_size[1] : 0;
      const canvas = $("roi-canvas");
      const img = $("preview-image");
      if (!sw || !sh) {
        sw = img && img.naturalWidth ? img.naturalWidth : (canvas && canvas.width ? canvas.width : 2560);
        sh = img && img.naturalHeight ? img.naturalHeight : (canvas && canvas.height ? canvas.height : 1440);
      }
      return [sw, sh];
    }

    // Keep editor points in original source pixels. Preview decoding and canvas
    // layout can change independently; neither may change the stored geometry.
    function sourceToPreview(pt) {
      if (!pt || pt.length < 2) return null;
      return [Math.round(pt[0]), Math.round(pt[1])];
    }

    function previewToSource(pt) {
      if (!pt || pt.length < 2) return null;
      return [Math.round(pt[0]), Math.round(pt[1])];
    }

    function normToPreview(pt) {
      if (!pt || pt.length < 2) return null;
      const [sw, sh] = getSourceDimensions();
      return [Math.round(pt[0] * sw), Math.round(pt[1] * sh)];
    }

    function previewToNorm(pt) {
      if (!pt || pt.length < 2) return null;
      const [sw, sh] = getSourceDimensions();
      return [
        Number(Math.max(0, Math.min(1, pt[0] / sw)).toFixed(4)),
        Number(Math.max(0, Math.min(1, pt[1] / sh)).toFixed(4)),
      ];
    }

    function initRoiEditorFromStream(stream) {
      if (!stream) {
        roiEditorState.courtPoints = [];
        roiEditorState.mirrorPoints = [];
        return;
      }
      if (stream.points && stream.points.length >= 3) {
        const sorted = (stream.points.length === 4) ? sortPointsTLTRBRBL(stream.points) : stream.points;
        roiEditorState.courtPoints = sorted.map(sourceToPreview);
      } else {
        roiEditorState.courtPoints = [];
      }
      const mPoly = stream.mirror_view?.polygon;
      if (mPoly && mPoly.length >= 3) {
        const [sw, sh] = getSourceDimensions();
        const mSrcRaw = mPoly.map(p => [Math.round(p[0] * sw), Math.round(p[1] * sh)]);
        const sorted = (mSrcRaw.length === 4) ? sortPointsTLTRBRBL(mSrcRaw) : mSrcRaw;
        roiEditorState.mirrorPoints = sorted.map(sourceToPreview);
      } else {
        roiEditorState.mirrorPoints = [];
      }
    }

    function getImageRenderedRect(img) {
      if (!img || !img.naturalWidth || !img.naturalHeight) return null;
      const cw = img.clientWidth;
      const ch = img.clientHeight;
      const nw = img.naturalWidth;
      const nh = img.naturalHeight;
      if (!cw || !ch) return null;

      const imgRatio = nw / nh;
      const clientRatio = cw / ch;

      let rw, rh, rx, ry;
      if (clientRatio > imgRatio) {
        rh = ch;
        rw = ch * imgRatio;
        rx = (cw - rw) / 2;
        ry = 0;
      } else {
        rw = cw;
        rh = cw / imgRatio;
        rx = 0;
        ry = (ch - rh) / 2;
      }
      const rect = img.getBoundingClientRect();
      return {
        left: rect.left + rx,
        top: rect.top + ry,
        width: rw,
        height: rh,
        naturalWidth: nw,
        naturalHeight: nh,
      };
    }

    function syncCanvasGeometry() {
      const img = $("preview-image");
      const canvas = $("roi-canvas");
      const stage = $("preview-stage-container");
      if (!img || !canvas || !stage || img.hidden || !img.naturalWidth) return;

      const rect = getImageRenderedRect(img);
      if (!rect) return;

      const stageRect = stage.getBoundingClientRect();
      canvas.style.position = "absolute";
      canvas.style.left = `${rect.left - stageRect.left}px`;
      canvas.style.top = `${rect.top - stageRect.top}px`;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      const [sw, sh] = getSourceDimensions();
      canvas.width = sw;
      canvas.height = sh;

      const stream = selectedStream();
      if ((!roiEditorState.courtPoints.length || !roiEditorState.mirrorPoints.length) && stream) {
        initRoiEditorFromStream(stream);
      }

      drawRoiCanvas();
    }

    function parsePointString(str) {
      if (!str) return null;
      const parts = str.split(/[,，\s]+/).filter(Boolean);
      if (parts.length >= 2) {
        const x = parseFloat(parts[0]);
        const y = parseFloat(parts[1]);
        if (!isNaN(x) && !isNaN(y)) return [x, y];
      }
      return null;
    }

    function parseSidebarPoints(target = "court") {
      const isCourt = target === "court";
      const ids = isCourt ? ["roi_p1", "roi_p2", "roi_p3", "roi_p4"] : ["mirror_m1", "mirror_m2", "mirror_m3", "mirror_m4"];
      const raw = ids.map(id => parsePointString($(id)?.value)).filter(Boolean);
      if (isCourt) {
        return raw.map(p => [Math.round(p[0]), Math.round(p[1])]);
      } else {
        const [sw, sh] = getSourceDimensions();
        return raw.map(p => {
          if (p[0] <= 1.0 && p[1] <= 1.0) {
            return [Math.round(p[0] * sw), Math.round(p[1] * sh)];
          }
          return [Math.round(p[0]), Math.round(p[1])];
        });
      }
    }

    function updateRoiInputs(target = "court", previewPoints = null) {
      const isCourt = target === "court";
      const ids = isCourt ? ["roi_p1", "roi_p2", "roi_p3", "roi_p4"] : ["mirror_m1", "mirror_m2", "mirror_m3", "mirror_m4"];
      const pts = previewPoints || (isCourt ? roiEditorState.courtPoints : roiEditorState.mirrorPoints);
      if (!pts) return;

      ids.forEach((id, idx) => {
        const el = $(id);
        if (el && pts[idx]) {
          const srcPt = previewToSource(pts[idx]);
          el.value = `${srcPt[0]}, ${srcPt[1]}`;
          if (!isCourt) {
            const normPt = previewToNorm(pts[idx]);
            el.title = `源分辨率像素: [${srcPt[0]}, ${srcPt[1]}] · 归一化: (${normPt[0]}, ${normPt[1]})`;
          }
        }
      });
      savePreset();
    }

    function sortPointsTLTRBRBL(points) {
      if (!points || points.length !== 4) return points;
      const pts = points.map(p => [Number(p[0]), Number(p[1])]);
      const byY = [...pts].sort((a, b) => a[1] - b[1]);
      const topTwo = byY.slice(0, 2).sort((a, b) => a[0] - b[0]);
      const bottomTwo = byY.slice(2).sort((a, b) => b[0] - a[0]); // bottom-right then bottom-left
      return [topTwo[0], topTwo[1], bottomTwo[0], bottomTwo[1]];
    }

    function updateRoiEditorStatus(text) {
      const el = $("roi-editor-status");
      if (el) el.textContent = text;
    }

    function setEditTarget(target) {
      roiEditorState.target = target;
      roiEditorState.mode = "drag";
      roiEditorState.hoverIdx = -1;
      roiEditorState.draggingIdx = -1;
      roiEditorState.clickStep = 0;

      document.querySelectorAll(".roi-target-btn").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-target") === target);
      });

      const isCourt = target === "court";
      const badge = $("roi-mode-badge");
      if (badge) {
        badge.textContent = isCourt ? "🎾 球场 ROI 编辑 (P1~P4)" : "🪞 镜面排除区编辑 (M1~M4)";
        badge.style.borderColor = isCourt ? "var(--cyan)" : "#ff6e73";
        badge.style.color = isCourt ? "var(--cyan)" : "#ff6e73";
        badge.style.background = isCourt ? "rgba(0, 240, 255, 0.18)" : "rgba(255, 90, 95, 0.18)";
      }

      const courtBox = $("sidebar-court-box");
      const mirrorBox = $("sidebar-mirror-box");
      if (courtBox) courtBox.style.display = isCourt ? "grid" : "none";
      if (mirrorBox) mirrorBox.style.display = !isCourt ? "grid" : "none";

      const pts = isCourt ? roiEditorState.courtPoints : roiEditorState.mirrorPoints;
      if (!pts || pts.length < 4) {
        updateRoiEditorStatus(isCourt ? "尚未设置完整 4 点球场 ROI，请点击【🔄 重新描点】绘制" : "尚未设置镜面排除区，请点击【🔄 重新描点】在画面中绘制镜面 4 角");
      } else {
        updateRoiEditorStatus(isCourt ? "正在编辑球场 ROI (P1~P4)：拖拽黄色手柄微调，或点击重新描点" : "正在编辑镜面排除区 (M1~M4)：拖拽红色手柄微调，或点击重新描点");
      }

      drawRoiCanvas();
    }

    function drawRoiCanvas() {
      const canvas = $("roi-canvas");
      if (!canvas || !roiEditorState.active) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const cw = canvas.width;
      const ch = canvas.height;
      ctx.clearRect(0, 0, cw, ch);

      const isCourt = roiEditorState.target === "court";
      const activePts = isCourt ? roiEditorState.courtPoints : roiEditorState.mirrorPoints;
      const secondaryPts = isCourt ? roiEditorState.mirrorPoints : roiEditorState.courtPoints;
      const isClickMode = roiEditorState.mode === "click";
      const activeLabels = isCourt ? ["P1 (左上)", "P2 (右上)", "P3 (右下)", "P4 (左下)"] : ["M1 (左上)", "M2 (右上)", "M3 (右下)", "M4 (左下)"];

      // 1. 绘制非活跃目标（半透明背景轮廓）
      if (secondaryPts && secondaryPts.length >= 3) {
        ctx.save();
        ctx.beginPath();
        secondaryPts.forEach((p, idx) => {
          if (idx === 0) ctx.moveTo(p[0], p[1]);
          else ctx.lineTo(p[0], p[1]);
        });
        ctx.closePath();
        ctx.fillStyle = isCourt ? "rgba(255, 60, 60, 0.14)" : "rgba(0, 220, 255, 0.12)";
        ctx.fill();
        ctx.strokeStyle = isCourt ? "rgba(255, 90, 95, 0.75)" : "rgba(0, 240, 255, 0.65)";
        ctx.lineWidth = Math.max(2, Math.round(cw / 750));
        ctx.setLineDash([6, 5]);
        ctx.stroke();

        ctx.fillStyle = isCourt ? "#ff6e73" : "#00f0ff";
        ctx.font = `bold ${Math.max(12, Math.round(cw / 125))}px sans-serif`;
        const firstP = secondaryPts[0];
        const secTag = isCourt ? "🛡️ 镜面排除区 (M1~M4)" : "🎾 球场 ROI (P1~P4)";
        ctx.fillText(secTag, firstP[0] + 10, firstP[1] + 24);
        ctx.restore();
      }

      // 2. 绘制活跃目标多边形
      const themeColor = isCourt ? "#00f0ff" : "#ff5757";
      const fillColor = isCourt ? "rgba(0, 220, 255, 0.20)" : "rgba(255, 60, 60, 0.22)";

      if (activePts.length >= 2) {
        ctx.save();
        ctx.beginPath();
        activePts.forEach((p, idx) => {
          if (idx === 0) ctx.moveTo(p[0], p[1]);
          else ctx.lineTo(p[0], p[1]);
        });
        if (!isClickMode && activePts.length >= 3) {
          ctx.closePath();
          ctx.fillStyle = fillColor;
          ctx.fill();
        }
        ctx.strokeStyle = themeColor;
        ctx.lineWidth = Math.max(3, Math.round(cw / 550));
        ctx.stroke();
        ctx.restore();
      }

      // 3. 点击模式下的橡皮筋动态虚线
      if (isClickMode && roiEditorState.mousePos && roiEditorState.clickStep > 0 && roiEditorState.clickStep < 4) {
        const prev = activePts[roiEditorState.clickStep - 1];
        if (prev) {
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(prev[0], prev[1]);
          ctx.lineTo(roiEditorState.mousePos[0], roiEditorState.mousePos[1]);
          ctx.strokeStyle = isCourt ? "rgba(245, 208, 76, 0.9)" : "rgba(255, 130, 135, 0.9)";
          ctx.lineWidth = Math.max(2, Math.round(cw / 700));
          ctx.setLineDash([8, 6]);
          ctx.stroke();
          ctx.restore();
        }
      }

      // 4. 点击模式下的十字准星提示
      if (isClickMode && roiEditorState.mousePos && roiEditorState.clickStep < 4) {
        const mx = roiEditorState.mousePos[0];
        const my = roiEditorState.mousePos[1];
        const crossColor = isCourt ? "#f5d04c" : "#ff6e73";
        ctx.save();
        ctx.beginPath();
        ctx.arc(mx, my, Math.max(12, Math.round(cw / 150)), 0, Math.PI * 2);
        ctx.strokeStyle = crossColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = crossColor;
        ctx.font = `bold ${Math.max(12, Math.round(cw / 120))}px sans-serif`;
        ctx.fillText(`+ 点击放置 ${activeLabels[roiEditorState.clickStep]}`, mx + 16, my - 8);
        ctx.restore();
      }

      // 5. 绘制各个顶点手柄与源物理坐标胶囊标签
      const baseR = Math.max(14, Math.round(cw / 130));
      const fontSize = Math.max(12, Math.round(cw / 120));

      activePts.forEach((pt, i) => {
        if (!pt) return;
        const isHover = roiEditorState.hoverIdx === i;
        const isDrag = roiEditorState.draggingIdx === i;
        const r = (isHover || isDrag) ? baseR * 1.35 : baseR;
        const handleColor = isCourt
          ? (isDrag ? "#00f0ff" : (isHover ? "#f5d04c" : "#ffcc00"))
          : (isDrag ? "#ff333a" : (isHover ? "#ff9ea2" : "#ff6e73"));

        // 外层高亮辉光环
        if (isHover || isDrag) {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], r + 6, 0, Math.PI * 2);
          ctx.strokeStyle = isDrag ? themeColor : (isCourt ? "#f5d04c" : "#ff9ea2");
          ctx.lineWidth = 4;
          ctx.stroke();
        }

        // 手柄圆点
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
        ctx.fillStyle = handleColor;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = Math.max(2, Math.round(r * 0.22));
        ctx.stroke();

        // 圆内标号 1..4
        ctx.fillStyle = "#000000";
        ctx.font = `bold ${Math.round(r * 0.95)}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`${i + 1}`, pt[0], pt[1]);

        // 顶点坐标胶囊背景（显示原始物理源像素坐标！）
        const srcPt = previewToSource(pt);
        const text = `${activeLabels[i] || (isCourt ? `P${i+1}` : `M${i+1}`)} [${srcPt[0]}, ${srcPt[1]}]`;
        ctx.font = `bold ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        const textW = ctx.measureText(text).width;
        const pillH = fontSize + 8;
        const pillW = textW + 14;
        let pillX = pt[0] + r + 8;
        let pillY = pt[1] - pillH / 2;

        if (pillX + pillW > cw - 8) pillX = pt[0] - r - 8 - pillW;
        if (pillY < 8) pillY = 8;
        if (pillY + pillH > ch - 8) pillY = ch - 8 - pillH;

        ctx.fillStyle = "rgba(10, 16, 26, 0.92)";
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(pillX, pillY, pillW, pillH, 4);
        else ctx.rect(pillX, pillY, pillW, pillH);
        ctx.fill();
        ctx.strokeStyle = isDrag ? themeColor : (isHover ? (isCourt ? "#f5d04c" : "#ff9ea2") : themeColor);
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = isDrag ? themeColor : (isHover ? (isCourt ? "#f5d04c" : "#ff9ea2") : "#ffffff");
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, pillX + 7, pillY + pillH / 2);
      });
    }

    async function toggleRoiEditor(forceActive = null) {
      const nextActive = forceActive !== null ? Boolean(forceActive) : !roiEditorState.active;
      const btnHeader = $("btn-header-roi-edit");
      const btnSide = $("btn-toggle-roi-editor");
      const canvas = $("roi-canvas");
      const toolbar = $("roi-editor-toolbar");

      if (nextActive) {
        showToast("正在获取纯净无标注画面以进行高精度描点…");
        // Always request clean preview frame without baked annotations!
        await refreshPreview(false, true);

        roiEditorState.active = true;
        const stream = selectedStream();
        initRoiEditorFromStream(stream);

        if (roiEditorState.courtPoints.length < 3) {
          const sideCourt = parseSidebarPoints("court");
          if (sideCourt.length >= 3) {
            roiEditorState.courtPoints = sideCourt.map(sourceToPreview);
          }
        }
        if (roiEditorState.mirrorPoints.length < 3) {
          const sideMirror = parseSidebarPoints("mirror");
          if (sideMirror.length >= 3) {
            roiEditorState.mirrorPoints = sideMirror.map(sourceToPreview);
          }
        }

        canvas.hidden = false;
        canvas.style.pointerEvents = "auto";
        toolbar.hidden = false;
        if (btnHeader) {
          btnHeader.textContent = "✕ 退出描点";
          btnHeader.style.background = "rgba(255, 110, 115, 0.15)";
          btnHeader.style.borderColor = "#ff6e73";
          btnHeader.style.color = "#ff6e73";
        }
        if (btnSide) {
          btnSide.textContent = "✕ 退出描点编辑";
          btnSide.style.background = "rgba(255, 110, 115, 0.15)";
          btnSide.style.borderColor = "#ff6e73";
          btnSide.style.color = "#ff6e73";
        }
        syncCanvasGeometry();
        setEditTarget(roiEditorState.target || "court");
      } else {
        roiEditorState.active = false;
        roiEditorState.mode = "drag";
        roiEditorState.draggingIdx = -1;
        roiEditorState.hoverIdx = -1;
        canvas.hidden = true;
        canvas.style.pointerEvents = "none";
        toolbar.hidden = true;
        if (btnHeader) {
          btnHeader.textContent = "📐 描点编辑";
          btnHeader.style.background = "rgba(0, 240, 255, 0.12)";
          btnHeader.style.borderColor = "var(--cyan)";
          btnHeader.style.color = "var(--cyan)";
        }
        if (btnSide) {
          btnSide.textContent = "📐 画面交互描点编辑";
          btnSide.style.background = "rgba(0, 240, 255, 0.1)";
          btnSide.style.borderColor = "var(--cyan)";
          btnSide.style.color = "var(--cyan)";
        }
        await refreshPreview(false, false);
      }
    }

    function startRepointRoi() {
      const isCourt = roiEditorState.target === "court";
      roiEditorState.mode = "click";
      roiEditorState.clickStep = 0;
      if (isCourt) {
        roiEditorState.courtPoints = [];
      } else {
        roiEditorState.mirrorPoints = [];
      }
      roiEditorState.hoverIdx = -1;
      roiEditorState.draggingIdx = -1;
      const targetName = isCourt ? "球场 ROI" : "镜面排除区";
      updateRoiEditorStatus(`请在画面中点击【${targetName}】第 1 个点：左上角`);
      const badge = $("roi-mode-badge");
      if (badge) badge.textContent = `🎯 顺序点击 4 点 (${targetName})`;
      drawRoiCanvas();
    }

    function autoSortRoiPoints() {
      const isCourt = roiEditorState.target === "court";
      const pts = isCourt ? roiEditorState.courtPoints : roiEditorState.mirrorPoints;
      if (!pts || pts.length !== 4) {
        showToast("需要 4 个顶点才能进行四角自动排序", true);
        return;
      }
      const sorted = sortPointsTLTRBRBL(pts);
      if (isCourt) {
        roiEditorState.courtPoints = sorted;
        updateRoiInputs("court", sorted);
      } else {
        roiEditorState.mirrorPoints = sorted;
        updateRoiInputs("mirror", sorted);
      }
      drawRoiCanvas();
      showToast("已自动对齐四角：左上→右上→右下→左下");
    }

    function getCanvasPixelCoords(e) {
      const canvas = $("roi-canvas");
      if (!canvas) return [0, 0];
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return [0, 0];
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const rawX = Math.round(Math.max(0, Math.min(canvas.width - 1, (e.clientX - rect.left) * scaleX)));
      const rawY = Math.round(Math.max(0, Math.min(canvas.height - 1, (e.clientY - rect.top) * scaleY)));
      return [rawX, rawY];
    }

    const roiCanvasEl = $("roi-canvas");
    if (roiCanvasEl) {
      roiCanvasEl.addEventListener("mousedown", (e) => {
        if (!roiEditorState.active) return;
        const [rawX, rawY] = getCanvasPixelCoords(e);
        const isCourt = roiEditorState.target === "court";
        const labels = isCourt ? ["P1 (左上角)", "P2 (右上角)", "P3 (右下角)", "P4 (左下角)"] : ["M1 (左上角)", "M2 (右上角)", "M3 (右下角)", "M4 (左下角)"];
        let pts = isCourt ? roiEditorState.courtPoints : roiEditorState.mirrorPoints;

        if (roiEditorState.mode === "click") {
          pts[roiEditorState.clickStep] = [rawX, rawY];
          roiEditorState.clickStep++;

          if (roiEditorState.clickStep >= 4) {
            const sorted = sortPointsTLTRBRBL(pts);
            if (isCourt) {
              roiEditorState.courtPoints = sorted;
              updateRoiInputs("court", sorted);
            } else {
              roiEditorState.mirrorPoints = sorted;
              updateRoiInputs("mirror", sorted);
            }
            roiEditorState.mode = "drag";
            setEditTarget(roiEditorState.target);
            updateRoiEditorStatus("✅ 4点绘制完成！已自动按四角对齐，可拖拽手柄微调，或点击保存");
            showToast("✅ 4点描线完成，已自动对齐四角并填充坐标！");
          } else {
            if (isCourt) {
              roiEditorState.courtPoints = pts;
              updateRoiInputs("court", pts);
            } else {
              roiEditorState.mirrorPoints = pts;
              updateRoiInputs("mirror", pts);
            }
            updateRoiEditorStatus(`已放置 ${roiEditorState.clickStep}/4 点，请点击第 ${roiEditorState.clickStep + 1} 个点：${labels[roiEditorState.clickStep]}`);
          }
          drawRoiCanvas();
          return;
        }

        if (roiEditorState.mode === "drag") {
          const hitRadius = Math.max(28, Math.round(roiCanvasEl.width / 65));
          let hit = -1;
          pts.forEach((pt, idx) => {
            if (Math.hypot(pt[0] - rawX, pt[1] - rawY) <= hitRadius) hit = idx;
          });
          if (hit !== -1) {
            roiEditorState.draggingIdx = hit;
            roiCanvasEl.style.cursor = "grabbing";
            const srcPt = previewToSource([rawX, rawY]);
            updateRoiEditorStatus(`正在拖拽调整 ${labels[hit]}：[${srcPt[0]}, ${srcPt[1]}]`);
            drawRoiCanvas();
          }
        }
      });

      window.addEventListener("mousemove", (e) => {
        if (!roiEditorState.active) return;
        const [rawX, rawY] = getCanvasPixelCoords(e);
        roiEditorState.mousePos = [rawX, rawY];
        const isCourt = roiEditorState.target === "court";
        const labels = isCourt ? ["P1 (左上角)", "P2 (右上角)", "P3 (右下角)", "P4 (左下角)"] : ["M1 (左上角)", "M2 (右上角)", "M3 (右下角)", "M4 (左下角)"];
        let pts = isCourt ? roiEditorState.courtPoints : roiEditorState.mirrorPoints;

        if (roiEditorState.mode === "drag") {
          if (roiEditorState.draggingIdx !== -1) {
            pts[roiEditorState.draggingIdx] = [rawX, rawY];
            if (isCourt) {
              roiEditorState.courtPoints = pts;
              updateRoiInputs("court", pts);
            } else {
              roiEditorState.mirrorPoints = pts;
              updateRoiInputs("mirror", pts);
            }
            const srcPt = previewToSource([rawX, rawY]);
            updateRoiEditorStatus(`正在拖拽调整 ${labels[roiEditorState.draggingIdx]}：[${srcPt[0]}, ${srcPt[1]}]`);
            drawRoiCanvas();
          } else {
            const hitRadius = Math.max(28, Math.round(roiCanvasEl.width / 65));
            let hit = -1;
            pts.forEach((pt, idx) => {
              if (Math.hypot(pt[0] - rawX, pt[1] - rawY) <= hitRadius) hit = idx;
            });
            if (roiEditorState.hoverIdx !== hit) {
              roiEditorState.hoverIdx = hit;
              roiCanvasEl.style.cursor = hit !== -1 ? "grab" : "default";
              if (hit !== -1) {
                const srcPt = previewToSource(pts[hit]);
                updateRoiEditorStatus(`指向 ${labels[hit]} [${srcPt[0]}, ${srcPt[1]}] · 按住鼠标左键可拖动微调`);
              } else {
                updateRoiEditorStatus(isCourt ? "拖拽黄色角点手柄微调，或点击重新描点" : "拖拽红色手柄微调镜面边界，或点击重新描点");
              }
              drawRoiCanvas();
            }
          }
        } else if (roiEditorState.mode === "click") {
          roiCanvasEl.style.cursor = "crosshair";
          drawRoiCanvas();
        }
      });

      window.addEventListener("mouseup", () => {
        if (!roiEditorState.active) return;
        if (roiEditorState.draggingIdx !== -1) {
          roiEditorState.draggingIdx = -1;
          roiCanvasEl.style.cursor = roiEditorState.hoverIdx !== -1 ? "grab" : "default";
          const isCourt = roiEditorState.target === "court";
          updateRoiInputs(isCourt ? "court" : "mirror");
          updateRoiEditorStatus("拖拽角点手柄微调，或点击保存边界");
          drawRoiCanvas();
        }
      });
    }

    // 手动输入框同步输入
    ["roi_p1", "roi_p2", "roi_p3", "roi_p4"].forEach((id) => {
      const el = $(id);
      if (el) {
        el.addEventListener("input", () => {
          if (roiEditorState.active && roiEditorState.mode === "drag" && roiEditorState.target === "court") {
            const pts = parseSidebarPoints("court");
            if (pts.length > 0) {
              roiEditorState.courtPoints = pts.map(sourceToPreview);
              drawRoiCanvas();
            }
          }
        });
      }
    });

    ["mirror_m1", "mirror_m2", "mirror_m3", "mirror_m4"].forEach((id) => {
      const el = $(id);
      if (el) {
        el.addEventListener("input", () => {
          if (roiEditorState.active && roiEditorState.mode === "drag" && roiEditorState.target === "mirror") {
            const pts = parseSidebarPoints("mirror");
            if (pts.length > 0) {
              roiEditorState.mirrorPoints = pts.map(sourceToPreview);
              drawRoiCanvas();
            }
          }
        });
      }
    });

    // 目标切换按钮监听
    document.querySelectorAll(".roi-target-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-target") || "court";
        setEditTarget(target);
      });
    });

    window.addEventListener("resize", () => {
      if (roiEditorState.active) syncCanvasGeometry();
    });

    $("btn-toggle-roi-editor")?.addEventListener("click", () => toggleRoiEditor());
    $("btn-header-roi-edit")?.addEventListener("click", () => toggleRoiEditor());
    $("tool-btn-close")?.addEventListener("click", () => toggleRoiEditor(false));
    $("btn-repoint-roi")?.addEventListener("click", async () => {
      await toggleRoiEditor(true);
      startRepointRoi();
    });
    $("tool-btn-repoint")?.addEventListener("click", () => startRepointRoi());
    $("btn-auto-sort-roi")?.addEventListener("click", autoSortRoiPoints);
    $("tool-btn-sort")?.addEventListener("click", autoSortRoiPoints);
    $("tool-btn-save")?.addEventListener("click", saveRoiBoundary);

    async function saveRoiBoundary() {
      const stream = selectedStream();
      if (!stream || !stream.stream_id) {
        showToast("请先选择机位", true);
        return;
      }

      // 提取球场 ROI 物理源坐标
      let courtSourcePoints = [];
      if (roiEditorState.courtPoints && roiEditorState.courtPoints.length >= 3) {
        courtSourcePoints = roiEditorState.courtPoints.map(previewToSource);
      } else {
        courtSourcePoints = parseSidebarPoints("court");
      }

      if (courtSourcePoints.length < 3) {
        showToast("球场 ROI 边界至少需要输入 3 个有效顶点坐标 (格式: X, Y)", true);
        return;
      }
      if (courtSourcePoints.length === 4) {
        courtSourcePoints = sortPointsTLTRBRBL(courtSourcePoints);
      }

      // 提取镜面排除区归一化坐标 [0.0 ~ 1.0]
      let mirrorNormPoints = [];
      if (roiEditorState.mirrorPoints && roiEditorState.mirrorPoints.length >= 3) {
        mirrorNormPoints = roiEditorState.mirrorPoints.map(previewToNorm);
      } else {
        const sideMirror = parseSidebarPoints("mirror");
        const [sw, sh] = getSourceDimensions();
        if (sideMirror.length >= 3) {
          mirrorNormPoints = sideMirror.map(p => [
            Number(Math.max(0, Math.min(1, p[0] / sw)).toFixed(4)),
            Number(Math.max(0, Math.min(1, p[1] / sh)).toFixed(4)),
          ]);
        }
      }

      if (mirrorNormPoints.length === 4) {
        mirrorNormPoints = sortPointsTLTRBRBL(mirrorNormPoints);
      }

      const mirrorExclusion = $("roi_mirror_exclusion") ? $("roi_mirror_exclusion").checked : true;
      const btn = $("save-roi-boundary-btn");
      const toolBtn = $("tool-btn-save");
      if (btn) btn.disabled = true;
      if (toolBtn) toolBtn.disabled = true;

      try {
        const payload = {
          stream_id: stream.stream_id,
          points: courtSourcePoints,
          mirror_exclusion: mirrorExclusion,
        };
        if (mirrorNormPoints.length >= 3) {
          payload.mirror_polygon = mirrorNormPoints;
        }

        const response = await api("/api/roi/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (result.success) {
          showToast(`✅ 机位 ${stream.stream_id} ROI 与镜面排除区保存成功！`);
          stream.points = result.points;
          if (result.mirror_view) {
            stream.mirror_view = result.mirror_view;
          }
          roiEditorState.courtPoints = stream.points.map(sourceToPreview);
          if (stream.mirror_view && stream.mirror_view.polygon) {
            roiEditorState.mirrorPoints = stream.mirror_view.polygon.map(normToPreview);
          }
          updateStreamMeta();
          if (roiEditorState.active) {
            drawRoiCanvas();
          } else {
            refreshPreview(false, false);
          }
        } else {
          showToast(`保存失败: ${result.message || result.error || "未知错误"}`, true);
        }
      } catch (error) {
        showToast(`保存失败: ${error.message}`, true);
      } finally {
        if (btn) btn.disabled = false;
        if (toolBtn) toolBtn.disabled = false;
      }
    }

    if ($("save-roi-boundary-btn")) {
      $("save-roi-boundary-btn").addEventListener("click", saveRoiBoundary);
    }
