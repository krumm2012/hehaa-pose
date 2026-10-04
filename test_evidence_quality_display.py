"""Exercise real report renderers at the misleading-percent presentation seam."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from realtime_swing_pipeline import RealtimeSwingOutputManager
from swing_report_builder import render_report_html


def event():
    return {'event_id':1,'stroke_type':'Forehand','confidence':.93,
        'start_frame':0,'contact_frame':2,'peak_frame':2,'end_frame':4,
        'coach_advices':[{'message':'请复核','category':'review','confidence':0}],
        'biomechanics':{'metrics':{'arm_extension':{'value':130,'unit':'deg_2d',
            'confidence':.92,'coach_eligible':True}},'extended_biomechanics':{}}}


def live_html():
    with TemporaryDirectory() as tmp:
        manager=RealtimeSwingOutputManager(str(Path(tmp)/'events.json'),str(Path(tmp)/'report.html'),
            str(Path(tmp)/'clips'),25,(100,100),5)
        try:return manager._render_live_html({'summary':{},'events':[event()]})
        finally:manager.close()


class EvidenceQualityDisplayTests(unittest.TestCase):
    def test_live_metric_percent_has_explicit_uncalibrated_meaning(self):
        report=live_html()
        self.assertTrue('证据参考 92%' in report,'missing explicit metric evidence label')
        self.assertTrue('未经准确率校准' in report,'missing calibration limitation')
        self.assertFalse('<small>92%</small>' in report,'bare percent remains')

    def test_review_prompt_does_not_show_a_zero_percent_rating(self):
        report=live_html()
        self.assertTrue('复核提示' in report,'missing review prompt status')
        self.assertFalse('<small>0%</small>' in report,'review prompt still looks like zero rating')

    def test_standalone_advice_percent_is_also_disclosed(self):
        e=event();e['advice_list']=[{'message':'核对画面','confidence':.82,'category':'capture'}]
        with TemporaryDirectory() as tmp:
            report=render_report_html({'paths':{},'events':[e],'summary':{}},str(Path(tmp)/'report.html'))
        self.assertTrue('证据参考 82%' in report,'missing standalone advice evidence label')
        self.assertTrue('未经准确率校准' in report,'missing standalone calibration limitation')


if __name__=='__main__':unittest.main()
