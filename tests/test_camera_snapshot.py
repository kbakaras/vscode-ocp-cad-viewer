import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ocp_vscode.standalone import Viewer


class CameraSnapshotTest(unittest.TestCase):
    def test_explicit_reset_sends_full_scene_but_keep_uses_delta(self):
        previous = dict(revision=1, tree=dict(id="/Model", parts=[]), meshes={})
        for reset, expected in (("reset", "scene_full"), ("RESET", "scene_full"),
                                ("center", "scene_full"), ("keep", "scene_delta")):
            with self.subTest(reset=reset):
                viewer = Viewer.__new__(Viewer)
                viewer.scene_lock = threading.Lock()
                viewer.backend = SimpleNamespace(model=None)
                viewer.status = {}
                viewer.javascript_client = Mock()
                viewer.browser_revision = 1
                snapshot = dict(previous, type="scene_full", revision=2,
                                config=dict(reset_camera=reset, ortho=False), metrics={})
                viewer.publish_scene(snapshot, previous, {})
                message = json.loads(viewer.javascript_client.send.call_args.args[0])
                self.assertEqual(message["type"], expected)
                self.assertEqual(viewer.browser_revision, 2)
                if expected == "scene_full":
                    self.assertFalse(message["config"]["ortho"])
