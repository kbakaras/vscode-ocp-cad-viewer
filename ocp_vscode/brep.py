"""Versioned exact scene transport. No tessellation is performed by the encoder."""

import base64
import hashlib
import json
from enum import Enum
from pathlib import Path
from urllib.parse import quote

from ocp_tessellate.cad_objects import OcpGroup, OcpObject
from ocp_tessellate.convert import to_ocpgroup
from ocp_tessellate.ocp_utils import deserialize, loc_to_tq, serialize, tq_to_loc
from ocp_tessellate.utils import Color

PROTOCOL_VERSION = 1


def json_default(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def source_ids(objects, names, root_name):
    """Read optional viewer_id without depending on the modelling library."""
    result = {}

    def walk(obj, path):
        identity = getattr(obj, "viewer_id", None)
        if identity is not None:
            result[path] = str(identity)
        if isinstance(obj, dict):
            for name, child in obj.items():
                walk(child, f"{path}/{name}")

    for i, obj in enumerate(objects):
        name = names[i] if names and names[i] else getattr(obj, "label", None)
        path = f"/{root_name}" if len(objects) == 1 else f"/{root_name}/{name}"
        walk(obj, path)
    return result


def encode_scene(group, instances, config, identities=None, materials=None):
    geometries = {}
    identities = identities or {}

    def geometry(shape):
        raw = serialize(shape, triangles=False, normals=False)
        key = hashlib.sha256(raw).hexdigest()
        geometries.setdefault(key, base64.b64encode(raw).decode("ascii"))
        return key

    def walk(obj, parent, source_parent):
        source_path = f"{source_parent}/{obj.name}"
        segment = quote(identities.get(source_path, obj.name), safe="")
        node = dict(id=f"{parent}/{segment}", name=obj.name, kind=obj.kind,
                    loc=None if obj.loc is None else loc_to_tq(obj.loc))
        if isinstance(obj, OcpGroup):
            node["parts"] = [walk(child, node["id"], source_path) for child in obj.objects]
            ids = [child["id"] for child in node["parts"]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate viewer_id in {node['id']}")
        else:
            color = obj.color
            node.update(state=[obj.state_faces, obj.state_edges], width=obj.width,
                        material=obj.material, normalize_uvs=obj.normalize_uvs)
            if isinstance(color, list):
                node["color"] = [c.web_color for c in color]
                node["alpha"] = [c.a for c in color]
            else:
                node["color"] = color.web_color
                node["alpha"] = color.a
            if obj.ref is not None:
                node["geometry"] = geometry(instances[obj.ref]["obj"])
            else:
                shapes = obj.obj if isinstance(obj.obj, list) else [obj.obj]
                node["geometries"] = [geometry(shape) for shape in shapes]
            if obj.kind == "imageface":
                node.update(image=obj.image, image_type=obj.image_type, height=obj.height)
        return node

    packet = dict(protocol=PROTOCOL_VERSION, tree=walk(group, "", ""),
                  geometries=geometries, config=config, materials=materials)
    # Validate now, before a partially encoded message can reach the server.
    return json.loads(json.dumps(packet, default=json_default, ensure_ascii=False))


def decode_scene(packet):
    if packet.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("Unsupported exact scene protocol")
    objects = {}
    for key, encoded in packet["geometries"].items():
        raw = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(raw).hexdigest() != key:
            raise ValueError(f"BREP checksum mismatch: {key}")
        objects[key] = deserialize(raw)
    instances, refs, nodes = [], {}, {}

    def walk(node, parent):
        identity = node["id"]
        segment = identity.rsplit("/", 1)[-1]
        if not segment or identity != f"{parent}/{segment}" or identity in nodes:
            raise ValueError(f"Invalid or duplicate scene identity: {identity}")
        nodes[identity] = node
        loc = None if node.get("loc") is None else tq_to_loc(*node["loc"])
        if "parts" in node:
            return OcpGroup([walk(child, identity) for child in node["parts"]], segment, loc)
        kind = node["kind"]
        color = node["color"]
        if isinstance(color, list):
            color = [Color(c, a) for c, a in zip(color, node["alpha"])]
        else:
            color = Color(color, node["alpha"])
        args = dict(kind=kind, name=segment, loc=loc, color=color,
                    material=node.get("material"), width=node.get("width"),
                    show_faces=bool(node["state"][0]), show_edges=bool(node["state"][1]))
        if "geometry" in node:
            key = node["geometry"]
            uv_policy = (key, bool(node.get("material")),
                         node.get("normalize_uvs", True) if node.get("material") else True)
            if uv_policy not in refs:
                refs[uv_policy] = len(instances)
                instances.append(dict(obj=objects[key], cache_id=key, name=segment))
            args["ref"] = refs[uv_policy]
        else:
            args["obj"] = [objects[key] for key in node["geometries"]]
        obj = OcpObject(**args)
        obj.normalize_uvs = node.get("normalize_uvs", True)
        if kind == "imageface":
            obj.image, obj.image_type, obj.height = node["image"], node["image_type"], node["height"]
        return obj

    return walk(packet["tree"], ""), instances, nodes


def export_brep(filename, *cad_objs, names=None, colors=None, alphas=None,
                modes=None, materials=None, **config):
    """Save an exact scene without a running viewer or client tessellation."""
    group, instances = to_ocpgroup(*cad_objs, names=names, colors=colors,
                                  alphas=alphas, modes=modes, materials=materials)
    packet = encode_scene(group, instances, config,
                          source_ids(cad_objs, names, group.name))
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return packet
