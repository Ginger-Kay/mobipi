from pathlib import Path
import ast,json
import xml.etree.ElementTree as ET
import pytest

@pytest.mark.parametrize('scene,source',[
('MW-O-01','source-20261004T162000Z-MW-O-01'),
('DR-O-01','source-20261004T162000Z-DR-O-01')])
def test_actual_saved_target_handle(scene,source):
    root=Path(__file__).resolve().parents[4]/'scenes'/scene/'draft-v1'/source
    binding=json.loads((root/'target-binding.json').read_text())
    tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/check_human_recording.py').read_text())
    expression=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='handle' for t in n.targets))
    expected=eval(compile(ast.Expression(expression),'<handle>','eval'),{},dict(target=binding['fixture_name'],binding=binding))
    model=ET.parse(root/'model.xml').getroot()
    assert len([g for g in model.iter('geom') if g.get('name')==expected])==1
