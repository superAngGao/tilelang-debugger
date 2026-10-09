"""Import boundaries and one shared session across old and new module paths."""
import subprocess
import sys
import unittest


class ComponentTests(unittest.TestCase):
    def test_offline_protocols_do_not_load_emitters_or_gpu_runtime(self):
        script = '''
import sys
from tilelang_debugger.records import value
from tilelang_debugger.sample_records import parse
assert value(9007199254740993, 'int64') == 9007199254740993
assert not any(n == 'tilelang' or n.startswith('tilelang.') or
               n == 'tvm' or n.startswith('tvm.') or n == 'torch' or
               n.startswith('tilelang_debugger.emitters') or
               n.startswith('tilelang_debugger.runtime') for n in sys.modules)
'''
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_compatibility_emitters_share_and_restore_session(self):
        from tilelang_debugger import monitor, sample_monitor, capture_state
        from tilelang_debugger.emitters import samples, tiles
        self.assertIs(monitor.capture, tiles.capture)
        self.assertIs(sample_monitor.observe_value, samples.observe_value)
        self.assertIs(tiles.state, capture_state)
        with monitor.session([dict(id="p", schema=2)]) as state:
            self.assertIs(sample_monitor.active("p"), state["p"])
            state["p"]["root_built"] = True
            capture_state._buffers["p"] = "sentinel"
            self.assertEqual(monitor.buffer_bindings(), {"p": "sentinel"})
        self.assertIsNone(capture_state._active)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with monitor.session([dict(id="p")]):
                raise RuntimeError("deliberate")
        self.assertIsNone(capture_state._active)
        self.assertIsNone(capture_state._buffers)


if __name__ == "__main__":
    unittest.main()
