"""Creative control boundaries rather than snapshots of incidental wording."""
import pytest
from emilia_studio.compiler import OllamaPromptCompiler
from emilia_studio.config import load_config
from emilia_studio.workflows import EmiliaWorkflowBuilder


def test_portrait_removes_lower_body_competition():
    compiler = OllamaPromptCompiler(load_config())
    portrait = compiler.build_emilia_ultra_prompt('soft moonlight', 'portrait')
    full = compiler.build_emilia_ultra_prompt('soft moonlight', 'full_body')
    assert 'boots' not in portrait and 'stockings' not in portrait
    assert 'boots' in full
    assert 'cropped feet' not in compiler.build_emilia_ultra_negative(framing='portrait')


def test_cloak_does_not_negate_requested_clothing():
    compiler = OllamaPromptCompiler(load_config())
    positive = compiler.build_emilia_ultra_prompt('standing', outfit='cloak')
    negative = set(compiler.build_emilia_ultra_negative(outfit='cloak').split(', '))
    assert 'hooded cloak' in positive
    assert not {'hood', 'hooded cloak', 'white cloak'} & negative


def test_companion_removes_solo_constraint():
    compiler = OllamaPromptCompiler(load_config())
    prompt = compiler.build_emilia_ultra_prompt('garden', companion='puck')
    assert 'solo' not in set(prompt.split(', '))
    assert 'Puck' in prompt


@pytest.mark.parametrize('options', [{'framing':'invalid'}, {'camera_angle':'invalid'}, {'reference_mode':'invalid'}, {'hires_bridge':'invalid'}, {'checkpoint_name':'../outside.safetensors'}, {'hires_denoise':float('nan')}, {'ipadapter_start_at':0.9, 'ipadapter_end_at':0.1}])
def test_invalid_generation_settings_rejected_before_execution(options):
    with pytest.raises(ValueError):
        EmiliaWorkflowBuilder(load_config()).build_ultra_native(**options)

def test_outfit_mask_softens_internal_edges():
    import sys
    cfg = load_config()
    if not (cfg.app_dir / 'comfy_extras' / 'nodes_mask.py').is_file():
        pytest.skip('ComfyUI runtime unavailable')
    sys.path.insert(0, str(cfg.app_dir))
    try:
        from comfy_extras.nodes_mask import SolidMask, FeatherMask, MaskComposite
        graph = EmiliaWorkflowBuilder(cfg).build_ultra_native()
        if '201' not in graph:
            pytest.skip('Outfit adapter unavailable')
        values = {}
        classes = {'SolidMask': SolidMask, 'FeatherMask': FeatherMask, 'MaskComposite': MaskComposite}
        def execute(node_id):
            if node_id not in values:
                node = graph[node_id]
                inputs = {key: execute(value[0])[value[1]] if isinstance(value, list) else value for key, value in node['inputs'].items()}
                values[node_id] = classes[node['class_type']].execute(**inputs)
            return values[node_id]
        mask = execute(graph['44']['inputs']['attn_mask'][0])[0]
        rect = graph['202']['inputs']
        x, y = rect['x'], rect['y']
        assert float(mask[0, y, x]) < 0.01
        assert float(mask[0, y + 100, x + 100]) > 0.9
        assert float(mask[0, y - 1, x - 1]) == 0
    finally:
        sys.path.pop(0)
