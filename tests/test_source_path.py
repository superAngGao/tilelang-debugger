import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tilelang_debugger import cli, capture, access
from tilelang_debugger.source import load
from tilelang_debugger.instrument import prepare, Unsupported
from tilelang_debugger.access_contracts import prepare as access_prepare

ROOT = Path(__file__).resolve().parents[1]


class SourcePathTests(unittest.TestCase):
    def test_config_relative_to_its_directory_and_cli_relative_to_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_dir = root / 'configs'; config_dir.mkdir()
            kernel = root / 'chosen source.py'; kernel.write_text('chosen')
            (config_dir / 'kernel.py').write_text('wrong')
            config = config_dir / 'monitor.json'
            config.write_text(json.dumps(dict(source='../chosen source.py', points=[])))
            text, effective, metadata = load(config)
            self.assertEqual(text, 'chosen')
            self.assertEqual(effective['source'], str(kernel.resolve()))
            self.assertEqual(metadata['original_config']['source'], '../chosen source.py')
            config.write_text(json.dumps(dict(source='missing.py', points=[])))
            with patch('tilelang_debugger.source.Path.cwd', return_value=root):
                text, _, metadata = load(config, 'chosen source.py')
            self.assertEqual(text, 'chosen')
            self.assertEqual(metadata['selected_by'], 'cli')
            self.assertEqual(load(config, str(kernel))[0], 'chosen')
            with self.assertRaises(FileNotFoundError):
                load(config)
            with self.assertRaises(FileNotFoundError):
                load(config, str(root / 'missing.py'))

    def test_empty_path_and_config_shape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            for data in ([], {}, {'source': ''}, {'source': ' '}, {'source': 42}):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load(path)

    def test_source_filename_is_not_a_contract_key_but_content_still_is(self):
        directory = ROOT / 'examples/gelu'
        source = (directory / 'kernel.py').read_text(encoding='utf-8')
        driver = (directory / 'run.py').read_bytes()
        for filename, preparer in [('monitor.json', lambda s,c: prepare(s,c,driver)),
                                   ('access.json', lambda s,c: access_prepare(s,driver,c))]:
            config = json.loads((directory / filename).read_text())
            baseline = preparer(source, config)
            config['source'] = '/some other folder/custom_gelu.py'
            self.assertEqual(preparer(source, config), baseline)
            with self.assertRaises(Unsupported):
                preparer(source + '\n# changed', config)

    def test_cli_forwards_source_in_both_commands(self):
        for command, flag, module in [('run', '--monitor', capture), ('trace', '--access', access)]:
            with patch('sys.argv', ['tilelang-debugger', command, 'driver.py', flag, 'cfg.json', '--output', 'out', '--source', 'other name.py'] + (['--engine', 'reviewed'] if command == 'run' else [])), \
                 patch.object(module, 'run', return_value={'numerical_status': 'passed'}) as run:
                cli.main()
            self.assertEqual(run.call_args.kwargs, dict(source_path='other name.py'))

    def test_runs_read_selected_content_without_driver_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'driver').mkdir()
            driver = root / 'driver/run.py'; driver.write_text('driver')
            source = root / 'chosen.py'; source.write_text('exact selected content')
            for module, point_key in ((capture,'points'), (access,'accesses')):
                config = root / (point_key + '.json')
                config.write_text(json.dumps(dict(source='chosen.py', **{point_key: []})))
                with patch.object(module.sys, 'platform', 'linux'), patch.object(module, 'prepare', side_effect=Unsupported('stop at contract')) as prepare_mock:
                    with self.assertRaisesRegex(Unsupported, 'stop at contract'):
                        module.run(driver, config, root/'unused')
                self.assertEqual(prepare_mock.call_args.args[0], 'exact selected content')


if __name__ == '__main__':
    unittest.main()
