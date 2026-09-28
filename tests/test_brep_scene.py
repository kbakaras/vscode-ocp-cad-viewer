import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from build123d import Box, Pos
from ocp_tessellate.convert import to_ocpgroup

from ocp_vscode.brep import encode_scene, export_brep, source_ids
from ocp_vscode.scene import SceneWorker, build_snapshot, difference, flatten


def packet(width=10, x=0, color="#aabbcc", deviation=0.1):
    shape = Pos(x, 0, 0) * Box(width, 20, 30)
    group, instances = to_ocpgroup({"Panel": shape}, names=["Model"], colors=[color])
    return encode_scene(group, instances, dict(deviation=deviation))


class ExactSceneTest(unittest.TestCase):
    def test_export_needs_neither_connection_nor_client_tessellation(self):
        with tempfile.TemporaryDirectory() as tmp, patch("ocp_tessellate.convert.tessellate", side_effect=AssertionError):
            p = export_brep(Path(tmp) / "scene.json", {"Panel": Box(10, 20, 30)}, names=["Model"])
            self.assertEqual(json.loads((Path(tmp) / "scene.json").read_text()), p)
            self.assertNotIn("vertices", p["geometries"])

    def test_repeat_and_translation_use_same_geometry(self):
        a, _ = build_snapshot(packet(), 1)
        b, _ = build_snapshot(packet(), 2)
        d = difference(a, b)
        self.assertEqual(b["metrics"]["cache_misses"], 0)
        self.assertFalse(d["meshes"] or d["add"] or d["remove"] or d["update"])
        c, index = build_snapshot(packet(x=200), 3)
        d = difference(b, c)
        self.assertFalse(d["meshes"])
        self.assertEqual(len(d["update"]), 1)
        self.assertAlmostEqual(index["/Model/Panel"].Location().Transformation().TranslationPart().X(), 195)
        self.assertIn("/Model/Panel/faces/faces_0", index)

    def test_color_is_property_only_and_tolerance_changes_mesh(self):
        a, _ = build_snapshot(packet(), 1)
        b, _ = build_snapshot(packet(color="#ff0000"), 2)
        self.assertFalse(difference(a, b)["meshes"])
        c, _ = build_snapshot(packet(deviation=0.2), 3)
        self.assertTrue(difference(b, c)["meshes"])

    def test_size_and_add_remove_differences(self):
        a, _ = build_snapshot(packet(), 1)
        b, _ = build_snapshot(packet(width=11), 2)
        self.assertEqual(len(difference(a, b)["meshes"]), 1)
        group, instances = to_ocpgroup({"Group": {"One": Box(1, 2, 3)}}, names=["Model"])
        c, _ = build_snapshot(encode_scene(group, instances, {}), 3)
        d = difference(b, c)
        self.assertEqual(d["remove"], ["/Model/Panel"])
        self.assertEqual([n["id"] for n in d["add"]], ["/Model/Group", "/Model/Group/One"])

    def test_checksum_and_duplicate_ids_rejected(self):
        p = packet()
        key = next(iter(p["geometries"]))
        p["geometries"][key] = "AAAA"
        with self.assertRaisesRegex(ValueError, "checksum"):
            build_snapshot(p, 1)
        group, instances = to_ocpgroup({"A": Box(1, 2, 3), "B": Box(1, 2, 3)}, names=["Model"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            encode_scene(group, instances, {}, {"/Model/A": "same", "/Model/B": "same"})

    def test_identity_survives_display_rename(self):
        class Group(dict):
            hash_id = "fixed-id"
        snapshots = []
        for label in ("Bolt 1", "Bolt 2"):
            objects = [{label: Group(Body=Box(1, 2, 3))}]
            group, instances = to_ocpgroup(*objects, names=["Model"])
            p = encode_scene(group, instances, {}, source_ids(objects, ["Model"], group.name))
            snapshots.append(build_snapshot(p, len(snapshots) + 1)[0])
        d = difference(*snapshots)
        self.assertFalse(d["add"] or d["remove"] or d["meshes"])
        self.assertEqual(d["update"][0]["id"], "/Model/fixed-id")

    def test_uv_policy_does_not_modify_other_occurrences(self):
        group, instances = to_ocpgroup({"A": Box(10, 20, 30), "B": Box(10, 20, 30)}, names=["Model"])
        p = encode_scene(group, instances, {})
        a, _ = build_snapshot(p, 1)
        p["tree"]["parts"][0]["material"] = "wood"
        b, _ = build_snapshot(p, 2)
        self.assertEqual(len(b["meshes"]), 2)
        self.assertEqual(a["tree"]["parts"][1]["shape"], b["tree"]["parts"][1]["shape"])
        self.assertEqual(len(difference(a, b)["meshes"]), 1)
        p["tree"]["parts"][1]["material"] = "wood"
        p["tree"]["parts"][1]["normalize_uvs"] = False
        c, _ = build_snapshot(p, 3)
        self.assertEqual(len(c["meshes"]), 2)
        self.assertNotEqual(c["tree"]["parts"][0]["shape"], c["tree"]["parts"][1]["shape"])

    def test_measurements_follow_parent_placement(self):
        p = packet(x=10)
        p["tree"]["loc"] = [[50, 0, 0], [0, 0, 0, 1]]
        _, index = build_snapshot(p, 1)
        self.assertAlmostEqual(index["/Model/Panel"].Location().Transformation().TranslationPart().X(), 55)
        self.assertAlmostEqual(index["/Model/Panel/faces/faces_0"].Location().Transformation().TranslationPart().X(), 55)

    def test_old_browser_cannot_override_connection_or_acknowledge_new_scene(self):
        from ocp_vscode.standalone import Viewer

        class Socket:
            def __init__(self, *messages):
                self.messages = iter((*messages, None))

            def receive(self):
                return next(self.messages)

        viewer = Viewer({})
        current = Socket()
        viewer.javascript_client = current
        viewer.scene = {"revision": 2}
        old = Socket('U:{"command":"scene_applied","text":{"revision":1}}')
        viewer.handle_message(old)
        self.assertIs(viewer.javascript_client, current)
        self.assertNotIn("scene_metrics", viewer.status)
        current.messages = iter(('U:{"command":"scene_applied","text":{"revision":1}}', None))
        viewer.handle_message(current)
        self.assertNotIn("scene_metrics", viewer.status)

    def test_worker_coalesces_pending_and_retains_good_scene_on_error(self):
        entered, release, published, failed = (threading.Event() for _ in range(4))
        calls, results = [], []

        def build(p, revision):
            calls.append(revision)
            if revision == 1:
                entered.set()
                release.wait(5)
            if p.get("bad"):
                raise ValueError("bad")
            return {"revision": revision}, {}

        def publish(snapshot, previous, index):
            results.append(snapshot)
            published.set()

        with patch("ocp_vscode.scene.build_snapshot", side_effect=build):
            worker = SceneWorker(publish, lambda *args: failed.set())
            worker.submit({"protocol": 1})
            self.assertTrue(entered.wait(5))
            worker.submit({"protocol": 1})
            worker.submit({"protocol": 1})
            release.set()
            self.assertTrue(published.wait(5))
            self.assertEqual(calls, [1, 3])
            self.assertEqual(results, [{"revision": 3}])
            worker.submit({"protocol": 1, "bad": True})
            self.assertTrue(failed.wait(5))
            self.assertEqual(worker.completed["revision"], 3)


if __name__ == "__main__":
    unittest.main()
