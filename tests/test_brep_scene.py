import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from build123d import Box, Pos
from ocp_tessellate.convert import to_ocpgroup

from ocp_vscode.brep import apply_metadata, encode_scene, export_brep, source_metadata
from ocp_vscode.scene import SceneWorker, build_snapshot, difference, flatten


def packet(width=10, x=0, color="#aabbcc", deviation=0.1):
    shape = Pos(x, 0, 0) * Box(width, 20, 30)
    group, instances = to_ocpgroup({"Panel": shape}, names=["Model"], colors=[color])
    return encode_scene(group, instances, dict(deviation=deviation))


class ExactSceneTest(unittest.TestCase):
    def test_layers_and_dependencies_survive_snapshot_and_property_delta(self):
        class Group(dict):
            def __init__(self, *args, **attrs):
                super().__init__(*args)
                self.__dict__.update(attrs)

        panel = Group({"Body": Box(10, 20, 30)}, hash_id="panel-id", selection_root=True)
        labels = Group({"Text": Box(1, 2, 1)}, layer=("labels", "Надписи"), layer_visible=False,
                       visible_if_any=("panel-id",))
        panel["Name"] = labels
        bolt = Group({"Body": Box(2, 2, 10)}, visible_if_any=("panel-id", "outside-id"))
        model = {"Panel": panel, "Hardware": Group({"Bolt": bolt}, layer=("hardware", "Фурнитура"))}
        group, instances = to_ocpgroup(model, names=["Model"])
        p = encode_scene(group, instances, {}, source_metadata([model], ["Model"], group.name))
        a, _ = build_snapshot(p, 1)
        flat = flatten(a["tree"])
        self.assertEqual(a["tree"]["layers"], {"labels": "Надписи", "hardware": "Фурнитура"})
        self.assertEqual(a["tree"]["layer_defaults"], {"labels": False, "hardware": True})
        self.assertEqual(flat["/Model/panel-id/Name"]["visible_if_any"], ["/Model/panel-id"])
        self.assertEqual(flat["/Model/Hardware/Bolt"]["visible_if_any"], ["/Model/panel-id"])
        self.assertEqual(flat["/Model/panel-id/Name"]["layer"], "labels")
        self.assertTrue(flat["/Model/panel-id"]["selection_root"])
        del p["tree"]["parts"][0]["selection_root"]
        p["tree"]["parts"][0]["parts"][1]["visible_if_any"] = []
        b, _ = build_snapshot(p, 2)
        delta = difference(a, b)
        self.assertFalse(delta["meshes"] or delta["add"] or delta["remove"])
        updates = {node["id"]: node for node in delta["update"]}
        self.assertEqual(updates["/Model/panel-id/Name"]["visible_if_any"], [])
        self.assertNotIn("selection_root", updates["/Model/panel-id"])
        self.assertEqual(b["metrics"]["cache_misses"], 0)

    def test_metadata_also_applies_to_client_tessellated_tree(self):
        tree = {"id": "/Model", "name": "Model", "parts": [
            {"id": "/Model/P", "name": "P"},
            {"id": "/Model/N", "name": "N"},
        ]}
        apply_metadata(tree, {"/Model/P": {"hash_id": "p", "selection_root": True},
                              "/Model/N": {"layer": ("labels", "Надписи"), "visible_if_any": ("p",),
                                           "visibility_owner": "p", "tree_leaf": True}})
        self.assertEqual(tree["parts"][1]["visible_if_any"], ["/Model/P"])
        self.assertEqual(tree["layers"], {"labels": "Надписи"})
        self.assertEqual(tree["parts"][1]["visibility_owner"], "/Model/P")
        self.assertTrue(tree["parts"][1]["tree_leaf"])
        self.assertTrue(tree["parts"][0]["selection_root"])

    def test_hardware_tree_metadata_survives_snapshots_and_property_deltas(self):
        class Group(dict):
            def __init__(self, *args, **attrs):
                super().__init__(*args)
                self.__dict__.update(attrs)

        owner = Group({"Panel": Group({"Body": Box(10, 20, 30)}, hash_id="panel")}, hash_id="module")
        bolt = Group({"Body": Box(2, 2, 10)}, hash_id="bolt",
                     visibility_owner="module", visible_if_any=("panel",))
        model = {"Product": {"Module": owner},
                 "Hardware": Group({"Bolts": Group({"Bolt": bolt}, tree_leaf=True)},
                                   layer=("hardware", "Фурнитура"))}
        group, instances = to_ocpgroup(model, names=["Model"])
        p = encode_scene(group, instances, {}, source_metadata([model], ["Model"], group.name))
        a, _ = build_snapshot(p, 1)
        flat = flatten(a["tree"])
        kind, element = "/Model/Hardware/Bolts", "/Model/Hardware/Bolts/bolt"
        self.assertTrue(flat[kind]["tree_leaf"])
        self.assertEqual(flat[element]["visibility_owner"], "/Model/Product/module")
        self.assertEqual(flat[element]["visible_if_any"], ["/Model/Product/module/panel"])
        raw_kind = p["tree"]["parts"][1]["parts"][0]
        del raw_kind["tree_leaf"]
        del raw_kind["parts"][0]["visibility_owner"]
        b, _ = build_snapshot(p, 2)
        delta = difference(a, b)
        self.assertFalse(delta["meshes"] or delta["add"] or delta["remove"])
        self.assertEqual({node["id"] for node in delta["update"]}, {kind, element})
        self.assertEqual(b["metrics"]["cache_misses"], 0)
        self.assertNotIn("tree_leaf", flatten(b["tree"])[kind])
        self.assertNotIn("visibility_owner", flatten(b["tree"])[element])

    def test_unknown_visibility_owner_is_rejected(self):
        tree = {"id": "/Model", "name": "Model", "parts": []}
        with self.assertRaisesRegex(ValueError, "Unknown visibility owner"):
            apply_metadata(tree, {"/Model": {"visibility_owner": "absent"}})

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
            encode_scene(group, instances, {}, {"/Model/A": {"hash_id": "same"}, "/Model/B": {"hash_id": "same"}})

    def test_identity_survives_display_rename(self):
        class Group(dict):
            hash_id = "fixed-id"
        snapshots = []
        for label in ("Bolt 1", "Bolt 2"):
            objects = [{label: Group(Body=Box(1, 2, 3))}]
            group, instances = to_ocpgroup(*objects, names=["Model"])
            p = encode_scene(group, instances, {}, source_metadata(objects, ["Model"], group.name))
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
