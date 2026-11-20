#!/usr/bin/env python3
"""Generates assets/bracket.mesh, the finite element model of cpu_cad_solver.

The part is an L-shaped steel mounting bracket: a base plate with four bolt holes, an upright wall, a fillet along the
inner corner and a stiffening rib. It is meshed with second order tetrahedra (tet10) by gmsh. The bolt hole surfaces
are the clamped boundary, the top face of the wall carries the load.

gmsh is only needed to regenerate the asset, never to build or run the task. The headless wheel works without X11:

    python3 -m venv venv
    venv/bin/pip install numpy
    venv/bin/pip install --pre --index-url https://gmsh.info/python-packages-dev-nox gmsh
    venv/bin/python tools/gen_model.py

Output format (little endian), read by src/mesh.cpp:

    char[8]  magic "BLZFEM01"
    u32      node count N, tet count T, clamped node count C, load triangle count L
    f64[3N]  node coordinates in mm
    u32[10T] tet10 connectivity: corners 0-3, then mid nodes of edges 01, 12, 02, 03, 13, 23
    u32[C]   clamped node ids
    u32[6L]  tri6 load faces: corners 0-2, then mid nodes of edges 01, 12, 02
"""

import os
import struct
import sys

import gmsh
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "bracket.mesh")

# Geometry in mm.
BASE_X, BASE_Y, BASE_T = 120.0, 80.0, 10.0
WALL_T, WALL_H = 10.0, 100.0
FILLET_R = 8.0
RIB_T, RIB_LEG = 10.0, 70.0
HOLE_R = 5.0
HOLES = [(45.0, 18.0), (45.0, 62.0), (100.0, 18.0), (100.0, 62.0)]

# Element sizes in mm.
SIZE_GLOBAL = 5.0
SIZE_FINE = 2.4

# gmsh tet10 order: edges 01, 12, 20, 30, 32, 31. Maps our edge order (01, 12, 02, 03, 13, 23) to gmsh positions.
GMSH_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
TET_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
TRI_EDGES = [(0, 1), (1, 2), (0, 2)]


def build_geometry():
    occ = gmsh.model.occ
    base = occ.addBox(0, 0, 0, BASE_X, BASE_Y, BASE_T)
    wall = occ.addBox(0, 0, 0, WALL_T, BASE_Y, WALL_H)
    # Fillet: a block in the inner corner minus a cylinder along y.
    block = occ.addBox(WALL_T, 0, BASE_T, FILLET_R, BASE_Y, FILLET_R)
    cut = occ.addCylinder(WALL_T + FILLET_R, 0, BASE_T + FILLET_R, 0, BASE_Y, 0, FILLET_R)
    fillet, _ = occ.cut([(3, block)], [(3, cut)])
    # Rib: right triangle in the xz plane, extruded over its thickness in y.
    y0 = 0.5 * (BASE_Y - RIB_T)
    pts = [occ.addPoint(WALL_T, y0, BASE_T), occ.addPoint(WALL_T + RIB_LEG, y0, BASE_T),
           occ.addPoint(WALL_T, y0, BASE_T + RIB_LEG)]
    lines = [occ.addLine(pts[0], pts[1]), occ.addLine(pts[1], pts[2]), occ.addLine(pts[2], pts[0])]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    rib = occ.extrude([(2, face)], 0, RIB_T, 0)
    rib_vol = [e for e in rib if e[0] == 3]
    body, _ = occ.fuse([(3, base)], [(3, wall)] + fillet + rib_vol)
    holes = [(3, occ.addCylinder(x, y, -1.0, 0, 0, BASE_T + 2.0, HOLE_R)) for x, y in HOLES]
    body, _ = occ.cut(body, holes)
    occ.synchronize()
    return body


def classify_surfaces():
    clamped, load = [], []
    for dim, tag in gmsh.model.getEntities(2):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if abs(z1 - WALL_H) < 1e-6 and abs(z0 - WALL_H) < 1e-6:
            load.append(tag)
        elif gmsh.model.getType(dim, tag) == "Cylinder" and z1 <= BASE_T + 1e-6 and any(
                abs(cx - hx) < 1e-3 and abs(cy - hy) < 1e-3 for hx, hy in HOLES):
            clamped.append(tag)
    if len(clamped) != len(HOLES) or len(load) != 1:
        sys.exit(f"surface classification failed: {len(clamped)} hole surfaces, {len(load)} load faces")
    return clamped, load


def mesh(clamped):
    gmsh.option.setNumber("Mesh.MeshSizeMax", SIZE_GLOBAL)
    gmsh.option.setNumber("Mesh.MeshSizeMin", SIZE_FINE)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    # Refine around the bolt holes, where the stresses concentrate.
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "SurfacesList", clamped)
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", SIZE_FINE)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", SIZE_GLOBAL)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", 1.0)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", 12.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT, deterministic for a fixed thread count
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    # Mid nodes on curved surfaces follow the CAD geometry.
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.setOrder(2)


def export(clamped, load):
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(node_tags)}

    etypes, etags, enodes = gmsh.model.mesh.getElements(3)
    tet_type = gmsh.model.mesh.getElementType("Tetrahedron", 2)
    pos = list(etypes).index(tet_type)
    tets = np.array([index[int(t)] for t in enodes[pos]], dtype=np.int64).reshape(-1, 10)
    tets = tets[:, GMSH_TET10]
    quality = np.array(gmsh.model.mesh.getElementQualities(etags[pos], "minSICN"))
    if quality.min() <= 0.0:
        sys.exit("mesh has degenerate elements")

    fixed = set()
    for tag in clamped:
        tags, _, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        fixed.update(index[int(t)] for t in tags)

    tri_type = gmsh.model.mesh.getElementType("Triangle", 2)
    tris = []
    for tag in load:
        types, _, nodes = gmsh.model.mesh.getElements(2, tag)
        tris.append(np.array([index[int(t)] for t in nodes[list(types).index(tri_type)]]).reshape(-1, 6))
    tris = np.vstack(tris)

    # Keep only nodes used by the volume mesh, numbered in order of first use for locality.
    used = np.unique(tets)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    coords, tets, tris = coords[used], remap[tets], remap[tris]
    fixed = np.array(sorted(remap[list(fixed)]), dtype=np.int64)
    if (tris < 0).any() or (fixed < 0).any():
        sys.exit("boundary node outside the volume mesh")

    check_order(coords, tets, TET_EDGES)
    check_order(coords, tris, TRI_EDGES)
    vol = np.einsum("ij,ij->i", np.cross(coords[tets[:, 1]] - coords[tets[:, 0]], coords[tets[:, 2]] - coords[tets[:, 0]]),
                    coords[tets[:, 3]] - coords[tets[:, 0]])
    if (vol <= 0).any():
        sys.exit(f"{int((vol <= 0).sum())} inverted tets")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(b"BLZFEM01")
        f.write(struct.pack("<4I", len(coords), len(tets), len(fixed), len(tris)))
        f.write(coords.astype("<f8").tobytes())
        f.write(tets.astype("<u4").tobytes())
        f.write(fixed.astype("<u4").tobytes())
        f.write(tris.astype("<u4").tobytes())

    print(f"nodes {len(coords)}  dofs {3 * len(coords)}  tet10 {len(tets)}  clamped nodes {len(fixed)}  "
          f"load tri6 {len(tris)}  volume {vol.sum() / 6:.0f} mm^3  SICN min {quality.min():.3f} mean "
          f"{quality.mean():.3f}  -> {os.path.relpath(OUT)}")


def check_order(coords, elems, edges):
    for k, (a, b) in enumerate(edges):
        mid = 0.5 * (coords[elems[:, a]] + coords[elems[:, b]])
        edge = np.linalg.norm(coords[elems[:, a]] - coords[elems[:, b]], axis=1)
        first_mid = 4 if len(edges) == 6 else 3
        off = np.linalg.norm(coords[elems[:, first_mid + k]] - mid, axis=1)
        if (off > 0.25 * edge).any():
            sys.exit(f"mid node of edge {a}{b} is not on its edge")


def main():
    gmsh.initialize(["-nopopup"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("bracket")
    build_geometry()
    clamped, load = classify_surfaces()
    mesh(clamped)
    export(clamped, load)
    gmsh.finalize()


if __name__ == "__main__":
    main()
