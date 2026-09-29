"""Standalone exact-scene worker and revisioned mesh snapshots."""

import hashlib
import json
import threading
import time
from importlib.metadata import version

from ocp_tessellate.convert import tessellate_group
from ocp_tessellate.ocp_utils import (
    downcast, get_edges, get_faces, get_vertices, identity_location, make_compound,
)
from ocp_tessellate.utils import numpy_to_buffer_json

from .brep import decode_scene, json_default
from .standalone_defaults import DEFAULTS


def flatten(tree):
    result = {}

    def walk(node):
        result[node["id"]] = {k: v for k, v in node.items() if k not in ("parts", "bb")}
        if "parts" in node:
            result[node["id"]]["parts"] = []
            for child in node["parts"]:
                walk(child)
    walk(tree)
    return result


def difference(previous, current):
    old, new = flatten(previous["tree"]), flatten(current["tree"])
    removed = set(old) - set(new)
    # Removing a parent removes its entire subtree in the browser.
    remove = sorted(p for p in removed if p.rsplit("/", 1)[0] not in removed)
    add = [new[p] for p in sorted(set(new) - set(old), key=lambda p: p.count("/"))]
    update = [new[p] for p in new.keys() & old.keys() if new[p] != old[p]]
    meshes = {k: v for k, v in current["meshes"].items() if k not in previous["meshes"]}
    metrics = {**current["metrics"], "mesh_count": len(meshes)}
    return dict(type="scene_delta", base_revision=previous["revision"],
                revision=current["revision"], remove=remove, add=add, update=update,
                meshes=meshes, mesh_ids=list(current["meshes"]),
                bb=current["tree"].get("bb"), config={}, metrics=metrics)


def measurement_index(mapping):
    """Build off-thread and swap only together with the displayed revision."""
    result = {}

    def walk(node):
        if "parts" in node:
            for child in node["parts"]:
                walk(child)
            return
        shape = node["shape"]
        compound = shape["obj"] if isinstance(shape, dict) else make_compound(shape)
        loc = node["loc"] or identity_location()
        result[node["id"]] = downcast(compound.Moved(loc))
        for label, objects in (("faces", get_faces(compound)),
                               ("edges", get_edges(compound)),
                               ("vertices", get_vertices(compound))):
            for i, obj in enumerate(objects):
                result[f"{node['id']}/{label}/{label}_{i}"] = downcast(obj.Moved(loc))
    walk(mapping)
    return result


class Progress:
    def __init__(self):
        self.hits = self.misses = 0

    def update(self, mark):
        if mark == "c":
            self.hits += 1
        elif mark in ("+", "*"):
            self.misses += 1


def build_snapshot(packet, revision):
    started = time.perf_counter()
    group, instances, nodes = decode_scene(packet)
    config = {**DEFAULTS, **packet.get("config", {})}
    config.setdefault("grid", [config["grid_xy"], config["grid_xz"], config["grid_yz"]])
    config["render_edges"] = True
    progress = Progress()
    meshes, tree, mapping = tessellate_group(group, instances, config, progress)
    elapsed = time.perf_counter() - started
    # Cache keys include tessellation options, UV policy, and the engine version.
    engine = version("ocp-tessellate")
    policy = {k: config.get(k) for k in ("deviation", "angular_tolerance", "edge_accuracy")}
    material_refs = {}

    def refs(node):
        if "parts" in node:
            for child in node["parts"]:
                refs(child)
        elif node.get("material") and node["type"] == "shapes":
            material_refs[node["shape"]["ref"]] = node.get("normalize_uvs", True)
    refs(tree)
    keys = []
    for i, instance in enumerate(instances):
        spec = [instance["cache_id"], engine, policy, i in material_refs, material_refs.get(i, True)]
        keys.append(hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())

    def restore(node):
        node["name"] = nodes[node["id"]]["name"]
        for key in ("layer", "layers", "visible_if_any"):
            if key in nodes[node["id"]]:
                node[key] = nodes[node["id"]][key]
        if "parts" in node:
            for child in node["parts"]:
                restore(child)
        elif node["type"] == "shapes":
            node["shape"] = dict(mesh=keys[node["shape"]["ref"]])
    restore(tree)
    if packet.get("materials"):
        tree["materials"] = packet["materials"]
    config["normal_len"] = tree.get("normal_len", 0)
    config["_splash"] = False
    index_started = time.perf_counter()
    index = measurement_index(mapping)
    index_ms = (time.perf_counter() - index_started) * 1000
    snapshot = dict(type="scene_full", revision=revision, tree=tree,
                    meshes=dict(zip(keys, meshes)), config=config,
                    metrics=dict(tessellation_ms=elapsed * 1000,
                                 cache_hits=progress.hits, cache_misses=progress.misses,
                                 mesh_count=len(meshes), index_ms=index_ms,
                                 client_prepare_ms=packet.get("client_prepare_ms")))
    encoding_started = time.perf_counter()
    snapshot = json.loads(json.dumps(numpy_to_buffer_json(snapshot), default=json_default))
    snapshot["metrics"]["encode_ms"] = (time.perf_counter() - encoding_started) * 1000
    return snapshot, index


class SceneWorker:
    """One worker, one pending request; superseded results are never published."""

    def __init__(self, publish, error):
        self.publish, self.error = publish, error
        self.condition = threading.Condition()
        self.pending = None
        self.requested = 0
        self.completed = None
        self.thread = threading.Thread(target=self.run, daemon=True, name="ocp-scene")
        self.thread.start()

    def submit(self, packet):
        if packet.get("protocol") != 1:
            raise ValueError("Unsupported exact scene protocol")
        with self.condition:
            self.requested += 1
            self.pending = (self.requested, packet)
            self.condition.notify()
            return self.requested

    def cancel_pending(self):
        with self.condition:
            self.requested += 1
            self.pending = None
            self.completed = None

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None)
                revision, packet = self.pending
                self.pending = None
            try:
                snapshot, index = build_snapshot(packet, revision)
                with self.condition:
                    if revision != self.requested:
                        continue
                    previous = self.completed
                    self.completed = snapshot
                    self.publish(snapshot, previous, index)
            except Exception as exc:
                with self.condition:
                    if revision == self.requested:
                        self.error(revision, exc)
