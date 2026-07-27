import unittest

from processed_video_display import ProcessedVideoDisplay


class FakeCv2:
    WINDOW_NORMAL = 0
    WND_PROP_FULLSCREEN = 1
    WINDOW_FULLSCREEN = 2

    def __init__(self, key=0):
        self.key = key
        self.calls = []

    def namedWindow(self, name, mode):
        self.calls.append(("namedWindow", name, mode))

    def moveWindow(self, name, x, y):
        self.calls.append(("moveWindow", name, x, y))

    def setWindowProperty(self, name, prop, value):
        self.calls.append(("setWindowProperty", name, prop, value))

    def imshow(self, name, frame):
        self.calls.append(("imshow", name, frame))

    def waitKey(self, delay):
        self.calls.append(("waitKey", delay))
        return self.key

    def destroyWindow(self, name):
        self.calls.append(("destroyWindow", name))


class ProcessedVideoDisplayTests(unittest.TestCase):
    def test_opens_fullscreen_at_requested_origin_and_reuses_window(self):
        cv2 = FakeCv2()
        display = ProcessedVideoDisplay(cv2, origin=(1512, 0))

        self.assertFalse(display.show("frame-1"))
        self.assertFalse(display.show("frame-2"))
        display.close()

        self.assertEqual(
            [call[0] for call in cv2.calls].count("namedWindow"),
            1,
        )
        call_names = [call[0] for call in cv2.calls]
        first_show = call_names.index("imshow")
        first_wait = call_names.index("waitKey")
        move = call_names.index("moveWindow")
        fullscreen = call_names.index("setWindowProperty")
        self.assertLess(first_show, first_wait)
        self.assertLess(first_wait, move)
        self.assertLess(move, fullscreen)
        self.assertIn(
            ("moveWindow", "Tennis AI - HDMI Output", 1512, 0),
            cv2.calls,
        )
        self.assertIn(
            (
                "setWindowProperty",
                "Tennis AI - HDMI Output",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            ),
            cv2.calls,
        )
        self.assertEqual(
            [call for call in cv2.calls if call[0] == "imshow"],
            [
                ("imshow", "Tennis AI - HDMI Output", "frame-1"),
                ("imshow", "Tennis AI - HDMI Output", "frame-2"),
            ],
        )
        self.assertEqual(
            cv2.calls[-1],
            ("destroyWindow", "Tennis AI - HDMI Output"),
        )

    def test_escape_requests_pipeline_shutdown(self):
        display = ProcessedVideoDisplay(FakeCv2(key=27))

        self.assertTrue(display.show("frame"))

    def test_q_requests_pipeline_shutdown(self):
        display = ProcessedVideoDisplay(FakeCv2(key=ord("q")))

        self.assertTrue(display.show("frame"))


if __name__ == "__main__":
    unittest.main()
