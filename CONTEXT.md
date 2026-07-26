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
