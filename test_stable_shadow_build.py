import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.validate_stable_shadow import verify_frozen


class StableShadowBuildTests(unittest.TestCase):
    def test_identity_rejects_changed_source_model_commit_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'runtime.py';source.write_text('original')
            receipt=Path(tmp)/'frozen.json'
            original=dict(files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
                          models={'model':'model-hash'},source_commit='abc')
            def write():
                frozen=dict(original)
                frozen['build_id']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
                receipt.write_text(json.dumps(frozen))
            write()
            with patch('scripts.validate_stable_shadow.model_digest',return_value='model-hash'), \
                 patch('scripts.validate_stable_shadow.subprocess.check_output',return_value='abc\n'), \
                 patch('scripts.validate_stable_shadow.subprocess.run'):
                self.assertEqual(verify_frozen(receipt)['source_commit'],'abc')
                source.write_text('changed')
                with self.assertRaisesRegex(ValueError,'source/config/input'):verify_frozen(receipt)
                source.write_text('original')
                with patch('scripts.validate_stable_shadow.model_digest',return_value='other'):
                    with self.assertRaisesRegex(ValueError,'model'):verify_frozen(receipt)
                with patch('scripts.validate_stable_shadow.subprocess.check_output',return_value='other\n'):
                    with self.assertRaisesRegex(ValueError,'commit'):verify_frozen(receipt)
                frozen=json.loads(receipt.read_text());frozen['source_commit']='forged'
                receipt.write_text(json.dumps(frozen))
                with self.assertRaisesRegex(ValueError,'digest'):verify_frozen(receipt)


if __name__=='__main__':unittest.main()
