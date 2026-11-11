"""Consumer-visible CLI command and persistence checks."""
import json

from emilia_studio.cli import main


def test_configuration_persists_across_working_directories(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / 'settings'
    monkeypatch.setenv('EMILIA_CONFIG_DIR', str(config_dir))
    monkeypatch.delenv('EMILIA_CONFIG_TOML', raising=False)
    root = tmp_path / 'workspace'
    assert main(['config', 'init', '--root-dir', str(root)]) == 0
    assert main(['config', 'set', 'quality_port', '8205']) == 0
    monkeypatch.chdir(tmp_path)
    assert main(['config', 'show']) == 0
    assert '8205' in capsys.readouterr().out
    assert main(['config', 'init']) == 1
    assert main(['config', 'set', 'quality_port', '0']) == 1


def test_gallery_invalid_index_and_empty_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('EMILIA_OUTPUT_DIR', str(tmp_path))
    assert main(['gallery', '--open', '1']) == 1
    assert 'outside' in capsys.readouterr().err
    assert main(['gallery', '--limit', '0']) == 1


def test_machine_workflows_output_is_json(capsys):
    assert main(['workflows']) == 0
    report = json.loads(capsys.readouterr().out)
    assert 'emilia_ipadapter_highres_production.api.json' in report['canonical_workflows']


def test_unknown_download_key_fails_without_network(capsys):
    assert main(['pull', 'unknown-model']) == 1
    assert 'Unknown model' in capsys.readouterr().err


def test_noninteractive_entry_does_not_consume_input(monkeypatch):
    import io
    stream = io.StringIO('presets\n')
    monkeypatch.setattr('sys.stdin', stream)
    assert main([]) == 0
    assert stream.tell() == 0


def test_outfits_command_lists_and_prints_json(capsys):
    assert main(['outfits']) == 0
    table_out = capsys.readouterr().out
    for key in ('royal', 'cloak', 'frozen-bond', 'sanctuary', 'nightgown'):
        assert key in table_out
    assert main(['outfits', 'cloak', '--framing', 'portrait']) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec['outfit'] == 'cloak'
    assert spec['framing'] == 'portrait'
    assert spec['use_skirt_ipa'] is False
    assert 'white hooded cloak' in spec['positive_prompt']
    assert 'hooded cloak' not in spec['negative_prompt']


def test_compare_and_export_commands_work_offline(tmp_path, monkeypatch, capsys):
    from PIL import Image
    out_dir = tmp_path / 'output'
    out_dir.mkdir()
    monkeypatch.setenv('EMILIA_OUTPUT_DIR', str(out_dir))
    img_a = out_dir / 'a.png'
    img_b = out_dir / 'b.png'
    Image.new('RGB', (128, 192), (180, 140, 210)).save(img_a)
    Image.new('RGB', (128, 192), (180, 140, 210)).save(img_b)
    rec_a = img_a.with_suffix('.run.json')
    rec_a.write_text(json.dumps({'seed': 101, 'profile': 'ultra', 'final': str(img_a)}), encoding='utf-8')
    assert main(['compare', str(img_a), str(img_b), '--json']) == 0
    cmp_data = json.loads(capsys.readouterr().out)
    assert cmp_data['pixel_identical'] is True
    assert cmp_data['image_a']['seed'] == 101
    assert main(['compare', str(img_a), str(img_b)]) == 0
    assert 'YES' in capsys.readouterr().out
    bundle_dir = tmp_path / 'exported_bundle'
    assert main(['export', str(bundle_dir), '--image', str(rec_a)]) == 0
    exp_data = json.loads(capsys.readouterr().out)
    assert (bundle_dir / 'a.png').is_file()
    assert (bundle_dir / 'a.run.json').is_file()
    assert (bundle_dir / 'export_manifest.json').is_file()
    assert exp_data['destination'] == str(bundle_dir)


def test_clean_and_benchmark_commands_work_offline(tmp_path, monkeypatch, capsys):
    temp_dir = tmp_path / 'temp'
    stage_dir = temp_dir / 'ultra_deadbeef1234'
    stage_dir.mkdir(parents=True)
    (stage_dir / 'server.log').write_text('log bytes', encoding='utf-8')
    keep_dir = temp_dir / 'other_folder'
    keep_dir.mkdir(parents=True)
    monkeypatch.setenv('EMILIA_TEMP_DIR', str(temp_dir))
    monkeypatch.setenv('EMILIA_ROOT_DIR', str(tmp_path))
    assert main(['clean', '--older-than-hours', '0', '--dry-run']) == 0
    dry_res = json.loads(capsys.readouterr().out)
    assert dry_res['removed_count'] == 1
    assert stage_dir.is_dir()
    assert main(['clean', '--older-than-hours', '0']) == 0
    clean_res = json.loads(capsys.readouterr().out)
    assert clean_res['removed_count'] == 1
    assert not stage_dir.exists()
    assert keep_dir.is_dir()
    assert main(['benchmark', '--limit', '5']) == 0
    assert 'Benchmark Summary' in capsys.readouterr().out


def test_render_rejects_creative_flags_on_quality_profile_and_routes_ultra(monkeypatch, capsys):
    assert main(['render', '--profile', 'quality', '--outfit', 'cloak']) == 1
    assert 'require --profile ultra' in capsys.readouterr().err
    captured_kwargs = {}

    async def fake_pipeline(self, **kwargs):
        captured_kwargs.update(kwargs)
        return {}

    monkeypatch.setattr('emilia_studio.engine.EmiliaRenderEngine.render_quality_pipeline', fake_pipeline)
    assert main(['render', '--profile', 'ultra', '--preset', 'royal-portrait', '--outfit', 'cloak', '--angle', 'side', '--puck', '--magic', 'ice']) == 0
    assert captured_kwargs['profile'] == 'ultra'
    assert captured_kwargs['native_options']['framing'] == 'portrait'
    assert captured_kwargs['native_options']['camera_angle'] == 'side'
    assert captured_kwargs['native_options']['skirt_ipadapter_weight'] == 0.0
    assert 'Puck' in captured_kwargs['positive_prompt']
    assert 'ice crystals' in captured_kwargs['positive_prompt']


def test_non_canonical_outfit_decouples_base_and_finish_lora_and_batch_command_routes(monkeypatch, capsys):
    captured_render = {}

    async def fake_render(self, **kwargs):
        captured_render.update(kwargs)
        return {}

    monkeypatch.setattr('emilia_studio.engine.EmiliaRenderEngine.render_quality_pipeline', fake_render)
    assert main(['render', '--profile', 'ultra', '--outfit', 'frozen-bond', '--angle', 'three-quarter']) == 0
    assert captured_render['native_options']['lora_strength_base'] == 0.0
    assert captured_render['native_options']['reference_mode'] == 'paired'
    assert captured_render['native_options']['skirt_ipadapter_weight'] == 0.0
    assert captured_render['finish_options']['lora_strength'] == 0.28
    assert captured_render['ipadapter_weight'] == 0.28
    assert 'corset' in captured_render['negative_prompt']

    captured_batch = {}

    async def fake_batch(self, **kwargs):
        captured_batch.update(kwargs)
        return [
            {
                'requested_seed': s,
                'seed': s,
                'status': 'completed',
                'generation': {'seconds': 10.0},
                'finishing': {'seconds': 5.0},
                'final': f'out_{s}.png',
                'selected_native': f'nat_{s}.png',
            }
            for s in kwargs['seeds']
        ]

    monkeypatch.setattr('emilia_studio.engine.EmiliaRenderEngine.render_batch_pipeline', fake_batch)
    assert main(['batch', '--seeds', '20261113', '20261114', '--profile', 'ultra', '--outfit', 'nightgown']) == 0
    assert captured_batch['seeds'] == [20261113, 20261114]
    assert captured_batch['native_options']['lora_strength_base'] == 0.0
    assert captured_batch['finish_options']['lora_strength'] == 0.28
    assert 'Emilia Batch Render Summary' in capsys.readouterr().out


def test_experimental_models_pull_list_and_preset_default_seed_and_tuning_flags(monkeypatch, capsys):
    assert main(['pull', '--list']) == 0
    out = capsys.readouterr().out
    assert 'animesharp' in out
    assert 'illustrious-xl-1.1' in out

    captured = {}

    async def fake_render(self, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr('emilia_studio.engine.EmiliaRenderEngine.render_quality_pipeline', fake_render)
    assert main([
        'render', '--profile', 'ultra', '--preset', 'moonlit-balcony',
        '--garment-region', 'brooch_and_collar', '--hires-bridge', 'anime',
        '--secondary-upscaler', '4x-AnimeSharp.safetensors', '--upscale-blend', '0.35',
    ]) == 0
    assert captured['seed'] == 20261104
    assert captured['native_options']['hires_bridge'] == 'anime'
    assert captured['finish_options']['garment_region'] == 'brooch_and_collar'
    assert captured['finish_options']['secondary_upscaler_name'] == '4x-AnimeSharp.safetensors'
    assert captured['finish_options']['upscale_blend_strength'] == 0.35


def test_fast_flag_preset_routes_to_quality_and_ultra(monkeypatch, capsys):
    assert main(['pull', '--list']) == 0
    assert 'sdxl-vae-fp16-fix' in capsys.readouterr().out

    captured = {}

    async def fake_render(self, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr('emilia_studio.engine.EmiliaRenderEngine.render_quality_pipeline', fake_render)
    assert main(['render', '--profile', 'quality', '--fast']) == 0
    assert captured['native_steps'] == 32
    assert captured['hand_steps'] == 16
    assert captured['fp16_upscaler'] is True
    assert captured['fp16_vae'] is True
    assert captured['fast_tiled_vae'] is True
