# Tennis Analyzer

This context describes the visual evidence used to identify a tennis player's active ball and interpret a swing from a single-camera video.

## Tracking and Recognition

**主球 (Active Ball)**:
The one ball instance selected as the player's current rally ball in a frame sequence. A candidate with trajectory support takes priority over static-ball suppression.
_Avoid_: highest-confidence ball, any detected ball

**静止锚点 (Static Anchor)**:
A persistent image-plane location associated with a non-active ball across consecutive frames.
_Avoid_: active-ball history, ball trajectory

**轨迹支持 (Trajectory Support)**:
Evidence that a ball candidate is continuous with the prior Active Ball position or motion.
_Avoid_: confidence boost, proximity guess

**重捕获窗口 (Reacquisition Window)**:
The maximum consecutive-frame interval during which missing Active Ball observations retain Trajectory Support before a new Active Ball must be selected. The current window is eight frames.
_Avoid_: permanent track hold, unlimited history

**启动球拍方案 (Startup Racket Profile)**: A per-session choice fixed before model loading.
`legacy` uses the unified detector's racket boxes; `exclusive` uses only current
seven-point racket observations and their explicitly derived recovery. The
exclusive profile also excludes legacy boxes from Active Ball support, while
retaining the shared detector for ball inference. See ADR 0005.

## Human Coach Review

**有效事件 (Valid Event)**: An observed swing/hit accepted for event matching by the reviewer. Unknown stroke type does not invalidate an accepted event.

**触球候选帧 (Candidate Contact Frame)**: A source-frame anchor used to inspect a possible contact; it is not human confirmation of impact.

**人工触球确认 (Human Contact Confirmation)**: An explicit reviewer decision recorded separately from candidate frame alignment. Missing legacy decisions remain unreviewed.

**人工教练意见 (Human Coach Review)**: A named reviewer’s observation, evidence and judgment on the system advice, separate from recomputed system rules and optional five-dimension practice ratings. See `docs/HUMAN_COACH_REVIEW_V3.md`.
