"""Report inputs must not attach another event's Coach by coercing identity."""
import json
import tempfile
import unittest
from pathlib import Path

from swing_report_builder import build_report_payload, write_report_html
from realtime_swing_pipeline import RealtimeSwingOutputManager


def report_case(model_id=1, coach_rows=None, contact=10):
    model = {'events': [{'event_id': model_id, 'stroke_type': 'Forehand',
                         'start_frame': 0, 'contact_frame': contact, 'end_frame': 20}]}
    coach = {'events': coach_rows if coach_rows is not None else [
        {'event_id': 1, 'frames': {'contact': 10},
         'coach_advice': {'code': 'reference-one', 'message': 'event one only'}}]}
    return model, coach


def build_case(model, coach):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames, events, coaches = (root / name for name in ('frames.json', 'events.json', 'coach.json'))
        frames.write_text(json.dumps({'video_info': {}, 'frames': []}))
        events.write_text(json.dumps(model))
        coaches.write_text(json.dumps(coach))
        return build_report_payload(str(frames), str(events), str(coaches))


class ReportIdentityContractTests(unittest.TestCase):
    def test_invalid_model_or_coach_identity_is_rejected_at_report_input(self):
        for bad in (1.9, True, '1', -1, 2**53, None):
            for role in ('model', 'coach'):
                with self.subTest(role=role, identity=bad):
                    model, coach = report_case()
                    (model if role == 'model' else coach)['events'][0]['event_id'] = bad
                    with self.assertRaises(ValueError):
                        build_case(model, coach)

    def test_missing_or_duplicate_model_identity_is_rejected(self):
        model, coach = report_case()
        del model['events'][0]['event_id']
        with self.assertRaises(ValueError):
            build_case(model, coach)
        model, coach = report_case()
        model['events'].append(dict(model['events'][0]))
        with self.assertRaises(ValueError):
            build_case(model, coach)

    def test_fractional_model_identity_cannot_attach_coach_for_integer_event(self):
        with self.assertRaises(ValueError):
            build_case(*report_case(model_id=1.9))

    def test_duplicate_coach_identity_is_not_last_row_wins(self):
        model, coach = report_case()
        coach['events'].append({'event_id': 1, 'coach_advice': {'message': 'different event'}})
        with self.assertRaises(ValueError):
            build_case(model, coach)

    def test_explicit_missing_contact_is_not_borrowed_from_coach(self):
        result = build_case(*report_case(contact=None))
        self.assertIsNone(result['events'][0]['contact_frame'])

    def test_absent_contact_is_not_borrowed_from_another_document(self):
        model, coach = report_case()
        del model['events'][0]['contact_frame']
        self.assertIsNone(build_case(model, coach)['events'][0]['contact_frame'])

    def test_zero_identity_and_zero_contact_are_preserved(self):
        model, coach = report_case(model_id=0, contact=0)
        coach['events'][0].update(event_id=0, frames={'contact':0})
        result = build_case(model, coach)['events'][0]
        self.assertEqual(result['event_id'], 0)
        self.assertEqual(result['contact_frame'], 0)
        self.assertEqual(result['coach']['event_id'], 0)

    def test_declared_frame_aliases_are_validated_even_when_shadowed(self):
        for role in ('model', 'coach'):
            with self.subTest(role=role):
                model, coach = report_case()
                (model if role == 'model' else coach)['events'][0]['frames'] = {'contact':True}
                with self.assertRaises(ValueError):
                    build_case(model, coach)

    def test_legacy_event_frame_alias_preserves_null_and_zero(self):
        for contact in (None, 0):
            with self.subTest(contact=contact):
                model, coach = report_case()
                del model['events'][0]['contact_frame']
                model['events'][0]['frames'] = {'contact':contact}
                self.assertEqual(build_case(model, coach)['events'][0]['contact_frame'],contact)

    def test_input_documents_are_not_mutated(self):
        model, coach = report_case(contact=None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame_path,event_path,coach_path=(root/name for name in ('frames.json','events.json','coach.json'))
            frame_path.write_text(json.dumps({'video_info':{},'frames':[]}))
            event_path.write_text(json.dumps(model))
            coach_path.write_text(json.dumps(coach))
            original={p:p.read_bytes() for p in (frame_path,event_path,coach_path)}
            build_report_payload(str(frame_path),str(event_path),str(coach_path))
            self.assertEqual({p:p.read_bytes() for p in original},original)

    def test_invalid_payload_cannot_replace_existing_standalone_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'report.html'
            output.write_text('previous report')
            with self.assertRaises(ValueError):
                write_report_html({'paths':{},'events':[{'event_id':1.9}]},str(output))
            self.assertEqual(output.read_text(),'previous report')

    def test_live_renderer_rejects_invalid_or_duplicate_model_identity(self):
        manager = RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
        manager.roi_metadata = {}
        manager.preview_path = None
        manager.output_html = Path('/tmp/report_identity.html')
        manager.output_json = Path('/tmp/events.json')
        for events in ([{'event_id':1.9}], [{'event_id':1},{'event_id':1}]):
            with self.subTest(events=events),self.assertRaises(ValueError):
                manager._render_live_html({'events':events,'summary':{}})


if __name__ == '__main__':
    unittest.main()
