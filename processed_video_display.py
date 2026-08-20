"""Fullscreen display adapter for processed video frames."""

from __future__ import annotations

from typing import Optional, Tuple


class ProcessedVideoDisplay:
    """Show one processed frame stream and report user-requested shutdown."""

    def __init__(
        self,
        cv2_module,
        origin: Tuple[int, int] = (0, 0),
        fullscreen: bool = True,
        window_name: str = "Tennis AI - HDMI Output",
    ):
        self.cv2 = cv2_module
        self.origin = (int(origin[0]), int(origin[1]))
        self.fullscreen = bool(fullscreen)
        self.window_name = str(window_name)
        self._initialized = False

    def show(self, frame) -> bool:
        """Display a frame; return True when ESC or q requests shutdown."""
        if not self._initialized:
            self._open(frame)
        else:
            self.cv2.imshow(self.window_name, frame)
        key = self.cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"), ord("Q"))

    def close(self) -> None:
        if not self._initialized:
            return
        try:
            self.cv2.destroyWindow(self.window_name)
        finally:
            self._initialized = False

    def _open(self, first_frame) -> None:
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        # Cocoa assigns fullscreen windows to the display that currently owns
        # the realized window. Draw and pump events before moving; otherwise
        # macOS binds the window to the main Retina display.
        self.cv2.imshow(self.window_name, first_frame)
        self.cv2.waitKey(100)
        self.cv2.moveWindow(
            self.window_name,
            self.origin[0],
            self.origin[1],
        )
        self.cv2.waitKey(100)
        if self.fullscreen:
            self.cv2.setWindowProperty(
                self.window_name,
                self.cv2.WND_PROP_FULLSCREEN,
                self.cv2.WINDOW_FULLSCREEN,
            )
        self._initialized = True
