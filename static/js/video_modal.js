    var activeVideoPlayers = typeof activeVideoPlayers !== "undefined" ? activeVideoPlayers : new Set();
    var lastEventsJson = typeof lastEventsJson !== "undefined" ? lastEventsJson : "";
    var filterOnlyValidSwings = typeof filterOnlyValidSwings !== "undefined" ? filterOnlyValidSwings : true;
    var latestRawEvents = typeof latestRawEvents !== "undefined" ? latestRawEvents : [];

    function toggleClipPlayer(eventId, clipUrl) {
      const container = document.getElementById(`clip-player-${eventId}`);
      if (!container) return;
      if (container.hidden) {
        container.hidden = false;
        const video = container.querySelector(".clip-video-el");
        if (video && !video.src) {
          video.src = clipUrl;
        }
        activeVideoPlayers.add(eventId);
        if (video) video.play().catch(() => {});
      } else {
        container.hidden = true;
        const video = container.querySelector(".clip-video-el");
        if (video) video.pause();
        activeVideoPlayers.delete(eventId);
      }
    }

    function setSpeed(button, eventId, rate) {
      const container = document.getElementById(`clip-player-${eventId}`);
      if (!container) return;
      const video = container.querySelector(".clip-video-el");
      if (video) video.playbackRate = rate;
      const bar = container.querySelector(".player-controls-bar");
      if (bar) {
        bar.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
        button.classList.add("active");
      }
    }

    function stepFrame(eventId, deltaSec) {
      const container = document.getElementById(`clip-player-${eventId}`);
      if (!container) return;
      const video = container.querySelector(".clip-video-el");
      if (video) {
        video.pause();
        video.currentTime = Math.max(0, video.currentTime + deltaSec);
      }
    }

    function showFreezeModal(imgUrl) {
      const modal = document.getElementById("freeze-modal");
      const img = document.getElementById("freeze-modal-img");
      if (modal && img) {
        img.src = imgUrl;
        modal.hidden = false;
      }
    }

    function openClipModal(eventId, clipUrl, strokeText, score) {
      const modal = document.getElementById("clip-modal");
      const video = document.getElementById("modal-clip-video");
      const title = document.getElementById("modal-clip-title");
      if (modal && video) {
        if (title) title.textContent = `Event #${eventId} · ${strokeText} (质量评分: ${score}分)`;
        const metrics = document.getElementById("modal-event-metrics");
        if (metrics) {
          const event = latestRawEvents.find(e => e.event_id === eventId) || {};
          metrics.innerHTML = (event.clip_telemetry_event_id !== eventId ? `<p style="color:var(--muted);font-size:12px">历史切片未绑定当前事件遥测，视频卡片可能保留上一拍数值。复核请查看下方当前事件指标。</p>` : "") + renderEventMeasurements(event);
        }
        video.src = clipUrl;
        video.playbackRate = 0.5;
        modal.hidden = false;
        video.play().catch(() => {});
      }
    }

    function closeClipModal() {
      const modal = document.getElementById("clip-modal");
      const video = document.getElementById("modal-clip-video");
      if (modal) {
        modal.hidden = true;
        if (video) {
          video.pause();
          video.src = "";
        }
      }
    }

    function setModalSpeed(rate) {
      const video = document.getElementById("modal-clip-video");
      if (video) video.playbackRate = rate;
      document.querySelectorAll(".btn-modal-speed").forEach(b => {
        b.classList.toggle("active", parseFloat(b.dataset.speed) === rate);
      });
    }

    function stepModalFrame(deltaSec) {
      const video = document.getElementById("modal-clip-video");
      if (video) {
        video.pause();
        video.currentTime = Math.max(0, video.currentTime + deltaSec);
      }
    }

    window.addEventListener("keydown", (e) => {
      const clipModal = document.getElementById("clip-modal");
      if (clipModal && !clipModal.hidden) {
        if (e.key === "Escape") {
          closeClipModal();
        } else if (e.key === " ") {
          e.preventDefault();
          const video = document.getElementById("modal-clip-video");
          if (video) video.paused ? video.play() : video.pause();
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          stepModalFrame(e.shiftKey ? -0.2 : -0.04);
        } else if (e.key === "ArrowRight") {
          e.preventDefault();
          stepModalFrame(e.shiftKey ? 0.2 : 0.04);
        } else if (e.key === "1") {
          setModalSpeed(1.0);
        } else if (e.key === "2") {
          setModalSpeed(0.5);
        } else if (e.key === "3") {
          setModalSpeed(0.25);
        }
      }
    });
