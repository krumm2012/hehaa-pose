# Design QA — 自定义码流

- Source visual truth: `/Users/krum5539/Desktop/截屏2026-08-01 07.49.40.png`
- Implementation full screenshot: `/private/tmp/tennis-custom-stream-ui.png`
- Implementation baseline crop: `/private/tmp/tennis-camera-baseline.png`
- Implementation custom-state crop: `/private/tmp/tennis-camera-custom.png`
- Browser: Codex in-app Browser, `http://127.0.0.1:8765/`
- Viewport: default desktop viewport; full-page capture is 1265 × 1800 px
- Source pixels: 1008 × 424 px; source CSS size and density are unavailable
- Baseline crop: 411 × 191 px at device scale 1
- Custom crop: 411 × 261 px at device scale 1
- State: stopped pipeline; compared both Court 01 baseline and selected custom stream
- Density normalization: proportional component comparison because the supplied source is a cropped, higher-density capture with unknown CSS dimensions

**Findings**

- No actionable P0/P1/P2 differences. The existing camera state preserves the source hierarchy, two-column credential layout, colors, borders, radii, typography, and spacing at the current responsive width.
- The custom stream field uses the same full-width field treatment as the existing stream selector and adds only the vertical space required by the new state.
- Fonts and typography: existing system-font stack, weights, label hierarchy, and line heights are unchanged.
- Spacing and layout rhythm: existing field grid and gaps are unchanged; the conditional field follows the same 10 px grid gap.
- Colors and visual tokens: existing panel, border, text, muted, focus, and input tokens are reused without new colors.
- Image quality and assets: this form contains no raster assets, logos, illustrations, or custom icons requiring reproduction.
- Copy and content: “自定义码流” and “自定义码流地址” are concise and consistent with the existing Chinese control labels.

**Interaction verification**

- Selected “自定义码流” and confirmed the address field becomes visible and required.
- Entered an RTSP URL with a fake query token and confirmed the visible stream summary omits the query.
- Switched back to Court 01 and confirmed the custom field hides, loses `required`, and all four configured ROI points return.
- Reloaded the page and confirmed the custom address and custom selection are not persisted.
- Browser console errors/warnings: none.
- Backend regression suite validates supported schemes, rejects embedded credentials, injects credential fields, preserves URL connection options, and reuses a matching ROI profile.

**Comparison history**

- First comparison: no actionable P0/P1/P2 differences; no visual correction loop was required.

**Implementation Checklist**

- [x] Preserve configured camera choices and existing behavior.
- [x] Add a conditional custom stream URL field.
- [x] Keep custom URL and credentials out of browser presets.
- [x] Validate and sanitize custom stream URLs on the backend.
- [x] Preserve preview, ROI matching, and start-analysis behavior.
- [x] Verify desktop rendering, interaction states, tests, and console output.

**Follow-up Polish**

- None required for this scoped change.

final result: passed
