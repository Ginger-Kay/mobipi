"""Actual route audits require native substeps and exclude sealed-test data."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dr_v04_formal_audit import audit

class FormalAuditGuards(unittest.TestCase):
 def test_sealed_test_rejected_without_reading_video(self):
  freeze=dict(status='DR-v0.4_complete_preoutcome_freeze',primary=[dict(group_id='x',split='sealed_test',route_order=['E','D','A'])])
  with self.assertRaisesRegex(ValueError,'sealed test'):
   audit(freeze,'x','E',Path('/nonexistent/x/E/attempt'))
 def test_out_of_freeze_rejected(self):
  with self.assertRaisesRegex(ValueError,'unfrozen group'):
   audit(dict(status='not_frozen',primary=[]),'x','E',Path('/nonexistent'))

if __name__=='__main__':unittest.main()
