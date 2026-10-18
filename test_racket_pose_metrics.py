import unittest
from scripts.stress_racket_pose_pipeline import frame_metrics

class StreamMetricTests(unittest.TestCase):
    def rows(self,count=1001):
        return [{'frame_id':i,'timing':{'output_submitted_at_unix_ns':i*40_000_000,'capture_to_output_submit_ms':45.}} for i in range(count)]
    def test_clock_step_is_reported_without_crashing_or_inventing_latency(self):
        rows=self.rows();rows[500]['timing']['capture_to_output_submit_ms']=None
        result=frame_metrics(rows,0)
        self.assertEqual(result['invalid_timing_frames'],1)
        self.assertEqual(result['stable_p95_ms'],45)
        self.assertTrue(result['realtime_passed'])
    def test_many_invalid_timestamps_cannot_pass(self):
        rows=self.rows()
        for row in rows[:10]:row['timing']['capture_to_output_submit_ms']=None
        self.assertFalse(frame_metrics(rows,0)['realtime_passed'])
    def test_known_reader_frame_gaps_are_counted(self):
        rows=self.rows()
        for row in rows:row['frame_id']*=2
        result=frame_metrics(rows,0)
        self.assertGreater(result['reader_frame_drop_rate'],.49)
        self.assertFalse(result['realtime_passed'])

if __name__=='__main__':unittest.main()
