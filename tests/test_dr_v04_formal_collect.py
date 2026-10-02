"""Guard against consuming sealed-test or unregistered prospective groups."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dr_v04_formal_collect import verify_dispatch
from reference_executor import run_route


class FormalCollectorGuards(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.commit=subprocess.check_output(['git','-C',str(self.root),'rev-parse','HEAD'],text=True).strip()

    @patch('dr_v04_formal_collect.subprocess.check_output')
    def test_sealed_test_rejected_before_planner_or_output(self, fake):
        def git(command, *args, **kwargs):
            if command[-2:]==['rev-parse','HEAD']:
                return self.commit
            if command[-2:]==['status','--porcelain']:
                return b''
            raise AssertionError(f'unexpected planner or route access {command!r}')
        fake.side_effect=git
        freeze=dict(status='DR-v0.4_complete_preoutcome_freeze',formal_route_outcomes=0,
                    formal_execution_code_commit=self.commit,
                    primary=[dict(group_id='x',split='sealed_test')])
        with self.assertRaisesRegex(ValueError,'sealed test'):
            verify_dispatch(freeze,'x',self.root,Path('/nonexistent'))

    def test_bad_scopes_rejected_before_recording(self):
        with self.assertRaisesRegex(ValueError,'unrecognized execution scope'):
            run_route(None,'E',[],120,execution_scope='formal_without_freeze')

    def test_incomplete_freeze_rejected(self):
        with self.assertRaisesRegex(ValueError,'formal freeze not complete'):
            verify_dispatch(dict(status='study_spec_prepared'), 'x',self.root,Path('/nonexistent'))


if __name__=='__main__':unittest.main()
