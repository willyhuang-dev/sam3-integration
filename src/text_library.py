#!/usr/bin/env python3
"""Assemble a fixed-N_MAX text input from the concept library, + per-camera slot filter.

Deployment idea (L1/L2): run ONE engine over the UNION of all concepts (cheap — the
detector is nearly free), then each camera keeps only its own concepts downstream. This
sidesteps DeepStream per-stream side-input injection entirely: every stream computes the
same union; per-camera selection is just filtering output slots.

  assemble(lib, union, n_max) -> text_features[N_MAX,32,256], text_mask[N_MAX,32], slot_names
  camera_slots(slot_names, cam_concepts) -> [slot indices to keep for that camera]

Unused slots (union shorter than N_MAX) get zero features + zero mask -> produce nothing.
"""
import os, json
import numpy as np

SEQ, DIM = 32, 256


def load_manifest(lib):
    return json.load(open(os.path.join(lib, "manifest.json")))


def assemble(lib, union, n_max):
    man = load_manifest(lib)
    if len(union) > n_max:
        raise ValueError("union of %d concepts exceeds n_max=%d" % (len(union), n_max))
    tf = np.zeros((n_max, SEQ, DIM), np.float32)
    tm = np.zeros((n_max, SEQ), np.int64)
    names = [""] * n_max
    for i, c in enumerate(union):
        if c not in man:
            raise KeyError("concept %r not in library (run build_concept_library.py)" % c)
        d = np.load(os.path.join(lib, man[c]["file"]))
        tf[i], tm[i], names[i] = d["text_features"][0], d["text_mask"][0], c
    return tf, tm, names


def camera_slots(slot_names, cam_concepts):
    want = set(cam_concepts)
    missing = want - {n for n in slot_names if n}
    if missing:
        raise KeyError("camera concepts not in union: %s" % sorted(missing))
    return [i for i, n in enumerate(slot_names) if n in want]


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    lib = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "concept_library")
    man = load_manifest(lib)
    union = list(man.keys())
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else max(8, len(union))
    tf, tm, names = assemble(lib, union, n_max)
    print("union(%d) -> text_features%s text_mask%s" % (len(union), tf.shape, tm.shape))
    print("slots:", names)
    print("valid tokens/slot:", [int(m.sum()) for m in tm])
    cam = union[:2]
    print("demo camera wants %s -> keep output slots %s" % (cam, camera_slots(names, cam)))
