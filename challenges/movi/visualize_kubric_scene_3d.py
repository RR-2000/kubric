#!/usr/bin/env python3
"""Export an interactive 3D HTML visualization for a Kubric scene frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from string import Template
from typing import Any


SHAPE_LABELS = ["cube", "cylinder", "sphere"]
SIZE_LABELS = ["small", "large"]
COLOR_LABELS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
MATERIAL_LABELS = ["metal", "rubber"]


HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title}</title>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #f4f1ea;
      color: #1f1f1f;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    #app {
      width: 100%;
      height: 100%;
      position: relative;
    }
    #panel {
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 10;
      max-width: 360px;
      padding: 12px 14px;
      background: rgba(255, 252, 247, 0.88);
      border: 1px solid rgba(0, 0, 0, 0.12);
      border-radius: 12px;
      backdrop-filter: blur(8px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }
    #panel h1 {
      margin: 0 0 6px 0;
      font-size: 16px;
      font-weight: 700;
    }
    #panel p {
      margin: 0;
      font-size: 13px;
      line-height: 1.45;
    }
    #legend {
      margin-top: 10px;
      font-size: 12px;
      line-height: 1.45;
    }
    #legend div {
      margin-top: 4px;
    }
    #focus-panel {
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid rgba(0, 0, 0, 0.08);
    }
    #focus-panel strong {
      display: block;
      margin-bottom: 8px;
      font-size: 12px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: rgba(0, 0, 0, 0.66);
    }
    #focus-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }
    #object-links {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 240px;
      overflow: auto;
      padding-right: 4px;
    }
    .focus-link {
      display: block;
      width: 100%;
      text-align: left;
      cursor: pointer;
      border: 1px solid rgba(0, 0, 0, 0.12);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.82);
      color: #1b1b1b;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.35;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }
    .focus-link:hover {
      background: rgba(255, 255, 255, 0.97);
      border-color: rgba(0, 0, 0, 0.22);
      transform: translateY(-1px);
    }
    .focus-link.is-active {
      background: #1f1f1f;
      color: #fffaf2;
      border-color: #1f1f1f;
    }
    .focus-link .swatch {
      width: 9px;
      height: 9px;
      margin-top: 3px;
    }
    .focus-row {
      display: flex;
      gap: 8px;
      align-items: flex-start;
    }
    .focus-meta {
      min-width: 0;
      flex: 1;
    }
    .focus-name {
      font-weight: 700;
      word-break: break-word;
    }
    .focus-pos {
      margin-top: 2px;
      opacity: 0.7;
      font-size: 11px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 999px;
      vertical-align: middle;
    }
    #error {
      position: absolute;
      inset: 0;
      display: none;
      place-items: center;
      padding: 24px;
      background: #f4f1ea;
      font-size: 14px;
      line-height: 1.5;
    }
    code {
      background: rgba(0, 0, 0, 0.06);
      padding: 2px 4px;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="panel">
      <h1>${title}</h1>
      <p>${subtitle}</p>
      <div id="legend">
        <div><span class="swatch" style="background:#d62728;"></span>X axis</div>
        <div><span class="swatch" style="background:#2ca02c;"></span>Y axis</div>
        <div><span class="swatch" style="background:#1f77b4;"></span>Z axis</div>
        <div>Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.</div>
      </div>
      <div id="focus-panel">
        <strong>Focus</strong>
        <div id="focus-actions">
          <button class="focus-link" id="fit-all" type="button">Fit Entire Scene</button>
          <button class="focus-link" id="focus-camera" type="button">Focus Camera</button>
        </div>
        <div id="object-links"></div>
      </div>
    </div>
    <div id="error"></div>
  </div>
  <script type="application/json" id="scene-data">${scene_json}</script>
  <script type="module">
    import * as THREE from 'https://unpkg.com/three@0.168.0/build/three.module.js';
    import { OrbitControls } from 'https://unpkg.com/three@0.168.0/examples/jsm/controls/OrbitControls.js';

    const root = document.getElementById('app');
    const errorBox = document.getElementById('error');
    const objectLinks = document.getElementById('object-links');
    const fitAllButton = document.getElementById('fit-all');
    const focusCameraButton = document.getElementById('focus-camera');
    const data = JSON.parse(document.getElementById('scene-data').textContent);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(root.clientWidth, root.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    root.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(data.background);

    const camera = new THREE.PerspectiveCamera(45, root.clientWidth / root.clientHeight, 0.1, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;

    scene.add(new THREE.AmbientLight(0xffffff, 1.15));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(10, 14, 12);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xfff0d8, 0.55);
    fill.position.set(-8, -5, 9);
    scene.add(fill);

    const grid = new THREE.GridHelper(data.floorSize * 2, 20, 0x8f8f8f, 0xc8c8c8);
    grid.position.z = 0;
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(data.floorSize * 2, data.floorSize * 2),
      new THREE.MeshStandardMaterial({
        color: 0xd8d8d8,
        roughness: 0.95,
        metalness: 0.0,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
      })
    );
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    function makeTextSprite(text) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const fontSize = 30;
      ctx.font = '600 ' + fontSize + 'px Helvetica Neue, Arial, sans-serif';
      const metrics = ctx.measureText(text);
      canvas.width = Math.ceil(metrics.width + 24);
      canvas.height = 52;
      ctx.font = '600 ' + fontSize + 'px Helvetica Neue, Arial, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.strokeStyle = 'rgba(0,0,0,0.16)';
      ctx.lineWidth = 2;
      const r = 12;
      ctx.beginPath();
      ctx.moveTo(r, 0);
      ctx.lineTo(canvas.width - r, 0);
      ctx.quadraticCurveTo(canvas.width, 0, canvas.width, r);
      ctx.lineTo(canvas.width, canvas.height - r);
      ctx.quadraticCurveTo(canvas.width, canvas.height, canvas.width - r, canvas.height);
      ctx.lineTo(r, canvas.height);
      ctx.quadraticCurveTo(0, canvas.height, 0, canvas.height - r);
      ctx.lineTo(0, r);
      ctx.quadraticCurveTo(0, 0, r, 0);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#181818';
      ctx.fillText(text, 12, 36);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
      const sprite = new THREE.Sprite(material);
      const scale = 0.0085;
      sprite.scale.set(canvas.width * scale, canvas.height * scale, 1);
      return sprite;
    }

    function addOrientationTriad(group, origin, quaternion, scale) {
      const axes = [
        { dir: new THREE.Vector3(1, 0, 0), color: 0xd62728 },
        { dir: new THREE.Vector3(0, 1, 0), color: 0x2ca02c },
        { dir: new THREE.Vector3(0, 0, 1), color: 0x1f77b4 },
      ];
      for (const axis of axes) {
        const worldDir = axis.dir.clone().applyQuaternion(quaternion).normalize();
        const arrow = new THREE.ArrowHelper(worldDir, origin, scale, axis.color, scale * 0.25, scale * 0.13);
        group.add(arrow);
      }
    }

    function addBBoxEdges(group, bbox, dashed) {
      const edgeIndices = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];
      const points = [];
      for (const [a, b] of edgeIndices) {
        const pa = bbox[a];
        const pb = bbox[b];
        points.push(new THREE.Vector3(pa[0], pa[1], pa[2]));
        points.push(new THREE.Vector3(pb[0], pb[1], pb[2]));
      }
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = dashed
        ? new THREE.LineDashedMaterial({ color: 0x222222, dashSize: 0.14, gapSize: 0.09, linewidth: 1 })
        : new THREE.LineBasicMaterial({ color: 0x222222, linewidth: 1 });
      const lines = new THREE.LineSegments(geometry, material);
      if (dashed) {
        lines.computeLineDistances();
      }
      group.add(lines);
    }

    function addCameraGlyph(group, item) {
      const geom = new THREE.BufferGeometry();
      const s = item.scale;
      const vertices = new Float32Array([
        0, 0, 0,
        0.9 * s, 0.6 * s, -1.4 * s,
        0.9 * s, -0.6 * s, -1.4 * s,
        -0.9 * s, -0.6 * s, -1.4 * s,
        -0.9 * s, 0.6 * s, -1.4 * s,
      ]);
      const indices = [
        0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 1,
        1, 4, 3,
        1, 3, 2,
      ];
      geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      geom.setIndex(indices);
      geom.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({
        color: 0xf3f3f3,
        roughness: 0.7,
        metalness: 0.1,
        transparent: true,
        opacity: 0.75,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geom, material);
      mesh.position.fromArray(item.position);
      mesh.quaternion.set(item.quaternion_xyzw[0], item.quaternion_xyzw[1], item.quaternion_xyzw[2], item.quaternion_xyzw[3]);
      group.add(mesh);
      if (item.show_orientation) {
        addOrientationTriad(group, new THREE.Vector3(...item.position), mesh.quaternion, item.scale * 1.2);
      }
      return mesh;
    }

    function formatVec(vec) {
      return vec.map((value) => value.toFixed(2)).join(', ');
    }

    function clearActiveFocus() {
      for (const node of document.querySelectorAll('.focus-link')) {
        node.classList.remove('is-active');
      }
    }

    function markActiveFocus(node) {
      clearActiveFocus();
      if (node) {
        node.classList.add('is-active');
      }
    }

    function setCameraView(target, radiusScale, viewDir) {
      const safeRadius = Math.max(radiusScale, 0.8);
      const offset = viewDir.clone().normalize().multiplyScalar(safeRadius);
      camera.position.copy(target.clone().add(offset));
      controls.target.copy(target);
      camera.lookAt(target);
      controls.update();
    }

    function fitBox(box, buttonNode) {
      const target = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.8) * 2.1;
      setCameraView(target, radius, new THREE.Vector3(1.2, -1.35, 0.95));
      markActiveFocus(buttonNode);
    }

    function focusObject(entry, buttonNode) {
      const target = new THREE.Vector3(...entry.position);
      const radius = Math.max(entry.focus_radius * 2.4, 1.1);
      const viewDir = new THREE.Vector3(1.1, -1.2, 0.75);
      setCameraView(target, radius, viewDir);
      markActiveFocus(buttonNode);
    }

    const world = new THREE.Group();
    scene.add(world);
    const objectEntries = [];

    for (const item of data.objects) {
      const color = new THREE.Color(item.color[0], item.color[1], item.color[2]);
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: item.material === 'metal' ? 0.28 : 0.88,
        metalness: item.material === 'metal' ? 0.82 : 0.05,
        transparent: true,
        opacity: item.material === 'metal' ? 0.85 : 0.62,
      });

      let geometry;
      if (item.shape === 'sphere') {
        geometry = new THREE.SphereGeometry(1, 28, 20);
      } else if (item.shape === 'cylinder') {
        geometry = new THREE.CylinderGeometry(1, 1, 2, 28, 1, false);
        geometry.rotateX(Math.PI / 2);
      } else {
        geometry = new THREE.BoxGeometry(2, 2, 2);
      }

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.fromArray(item.position);
      mesh.quaternion.set(item.quaternion_xyzw[0], item.quaternion_xyzw[1], item.quaternion_xyzw[2], item.quaternion_xyzw[3]);
      mesh.scale.set(item.half_extents[0], item.half_extents[1], item.half_extents[2]);
      scene.add(mesh);
      objectEntries.push({ ...item, mesh });

      addBBoxEdges(scene, item.bbox_3d, item.material !== 'metal');

      if (item.show_orientation) {
        addOrientationTriad(scene, new THREE.Vector3(...item.position), mesh.quaternion, item.orientation_scale);
      }

      if (item.label) {
        const sprite = makeTextSprite(item.label);
        sprite.position.set(item.position[0], item.position[1], item.position[2] + item.label_height);
        scene.add(sprite);
      }
    }

    const cameraMesh = addCameraGlyph(scene, data.camera);

    const bounds = new THREE.Box3();
    for (const point of data.bounds_points) {
      bounds.expandByPoint(new THREE.Vector3(point[0], point[1], point[2]));
    }
    const cameraBox = new THREE.Box3().setFromObject(cameraMesh);

    fitAllButton.addEventListener('click', () => fitBox(bounds, fitAllButton));
    focusCameraButton.addEventListener('click', () => fitBox(cameraBox, focusCameraButton));

    for (const entry of objectEntries) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'focus-link';
      button.innerHTML =
        '<div class="focus-row">' +
        '<span class="swatch" style="background:' + entry.hex_color + ';"></span>' +
        '<div class="focus-meta">' +
        '<div class="focus-name">' + entry.name + '</div>' +
        '<div class="focus-pos">(' + formatVec(entry.position) + ')</div>' +
        '</div>' +
        '</div>';
      button.addEventListener('click', () => focusObject(entry, button));
      objectLinks.appendChild(button);
    }

    fitBox(bounds, fitAllButton);

    function resize() {
      const width = root.clientWidth;
      const height = root.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }

    window.addEventListener('resize', resize);

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    if (!window.WebGLRenderingContext) {
      errorBox.style.display = 'grid';
      errorBox.textContent = 'WebGL is not available in this browser, so the interactive viewer cannot render.';
    } else {
      animate();
    }
  </script>
</body>
</html>
"""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an interactive HTML 3D visualization of an exported Kubric scene "
            "from metadata.json. The saved HTML supports pan, rotate, and zoom."
        )
    )
    parser.add_argument("metadata_path", type=Path, help="Path to metadata.json for a Kubric export.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <scene_dir>/scene_3d_frame_XXXXX.html",
    )
    parser.add_argument(
        "--show-orientation",
        action="store_true",
        help="Draw local XYZ orientation arrows for all objects and the camera.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Do not annotate objects with text labels.",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=1.0,
        help="Multiplier for camera glyph size.",
    )
    parser.add_argument(
        "--orientation-scale",
        type=float,
        default=1.0,
        help="Multiplier for orientation-axis length.",
    )
    return parser.parse_args()


def decode_label(value: Any, labels: list[str]) -> str:
    if isinstance(value, str):
        return value
    return labels[int(value)]


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec_max_abs(points: list[list[float]]) -> list[float]:
    return [
        max(abs(point[0]) for point in points),
        max(abs(point[1]) for point in points),
        max(abs(point[2]) for point in points),
    ]


def mat3_transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]


def mat3_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]


def quaternion_wxyz_to_matrix(quaternion: list[float]) -> list[list[float]]:
    w, x, y, z = [float(v) for v in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quaternion_wxyz_to_xyzw(quaternion: list[float]) -> list[float]:
    w, x, y, z = [float(v) for v in quaternion]
    return [x, y, z, w]


def infer_half_extents(
    bbox_3d: list[list[float]],
    position: list[float],
    rotation: list[list[float]],
) -> list[float]:
    rotation_t = mat3_transpose(rotation)
    local_corners = [mat3_vec_mul(rotation_t, vec_sub(corner, position)) for corner in bbox_3d]
    half_extents = vec_max_abs(local_corners)
    return [max(value, 1e-3) for value in half_extents]


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_instances(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw_instances = metadata["instances"]
    if isinstance(raw_instances, list):
        normalized = []
        for object_idx, obj in enumerate(raw_instances):
            normalized.append(
                {
                    "object_idx": object_idx,
                    "shape": decode_label(obj.get("shape_label", obj.get("shape")), SHAPE_LABELS),
                    "size": decode_label(obj.get("size_label", obj.get("size_label", 0)), SIZE_LABELS),
                    "material": decode_label(obj.get("material_label", obj.get("material", 0)), MATERIAL_LABELS),
                    "color_label": decode_label(obj.get("color_label", 0), COLOR_LABELS),
                    "color": [float(v) for v in obj["color"]],
                    "positions": obj["positions"],
                    "quaternions": obj["quaternions"],
                    "bboxes_3d": obj["bboxes_3d"],
                    "visibility": obj["visibility"],
                }
            )
        return normalized

    num_instances = len(raw_instances["positions"])
    normalized = []
    for object_idx in range(num_instances):
        normalized.append(
            {
                "object_idx": object_idx,
                "shape": decode_label(raw_instances["shape_label"][object_idx], SHAPE_LABELS),
                "size": decode_label(raw_instances["size_label"][object_idx], SIZE_LABELS),
                "material": decode_label(raw_instances["material_label"][object_idx], MATERIAL_LABELS),
                "color_label": decode_label(raw_instances["color_label"][object_idx], COLOR_LABELS),
                "color": [float(v) for v in raw_instances["color"][object_idx]],
                "positions": raw_instances["positions"][object_idx],
                "quaternions": raw_instances["quaternions"][object_idx],
                "bboxes_3d": raw_instances["bboxes_3d"][object_idx],
                "visibility": raw_instances["visibility"][object_idx],
            }
        )
    return normalized


def build_label(instance: dict[str, Any]) -> str:
    return (
        f"{instance['object_idx']}: {instance['size']} {instance['color_label']} "
        f"{instance['material']} {instance['shape']}"
    )


def build_scene_payload(
    metadata: dict[str, Any],
    frame_idx: int,
    show_orientation: bool,
    hide_labels: bool,
    camera_scale: float,
    orientation_scale: float,
) -> tuple[str, str, dict[str, Any]]:
    scene_meta = metadata["metadata"]
    num_frames = int(scene_meta["num_frames"])
    if not 0 <= frame_idx < num_frames:
        raise ValueError(f"frame must be in [0, {num_frames - 1}], got {frame_idx}")

    instances = normalize_instances(metadata)
    bounds_points: list[list[float]] = []
    objects: list[dict[str, Any]] = []
    max_extent = 0.6

    for instance in instances:
        position = [float(v) for v in instance["positions"][frame_idx]]
        quaternion = [float(v) for v in instance["quaternions"][frame_idx]]
        bbox_3d = [[float(v) for v in corner] for corner in instance["bboxes_3d"][frame_idx]]
        rotation = quaternion_wxyz_to_matrix(quaternion)
        half_extents = infer_half_extents(bbox_3d, position, rotation)
        extent = max(half_extents)
        max_extent = max(max_extent, extent)

        bounds_points.extend(bbox_3d)
        bounds_points.append(position)

        objects.append(
            {
                "name": build_label(instance),
                "shape": instance["shape"],
                "material": instance["material"],
                "color": [min(max(float(v), 0.0), 1.0) for v in instance["color"]],
                "hex_color": "#{:02x}{:02x}{:02x}".format(
                    int(round(min(max(float(instance["color"][0]), 0.0), 1.0) * 255)),
                    int(round(min(max(float(instance["color"][1]), 0.0), 1.0) * 255)),
                    int(round(min(max(float(instance["color"][2]), 0.0), 1.0) * 255)),
                ),
                "position": position,
                "quaternion_xyzw": quaternion_wxyz_to_xyzw(quaternion),
                "half_extents": half_extents,
                "bbox_3d": bbox_3d,
                "focus_radius": extent,
                "show_orientation": show_orientation,
                "orientation_scale": extent * 1.35 * orientation_scale,
                "label": None if hide_labels else build_label(instance),
                "label_height": extent * 1.25,
            }
        )

    camera_position = [float(v) for v in metadata["camera"]["positions"][frame_idx]]
    camera_quaternion = [float(v) for v in metadata["camera"]["quaternions"][frame_idx]]
    bounds_points.append(camera_position)

    title = f"{scene_meta.get('video_name', 'scene')} | frame {frame_idx}"
    subtitle = (
        f"{len(objects)} objects | camera pose included"
        + (" | object and camera orientation shown" if show_orientation else "")
    )
    payload = {
        "background": "#f4f1ea",
        "floorSize": max(7.5, max_extent * 8.0),
        "bounds_points": bounds_points,
        "objects": objects,
        "camera": {
            "position": camera_position,
            "quaternion_xyzw": quaternion_wxyz_to_xyzw(camera_quaternion),
            "scale": max_extent * 1.8 * camera_scale,
            "show_orientation": show_orientation,
        },
    }
    return title, subtitle, payload


def render_scene_html(
    metadata: dict[str, Any],
    frame_idx: int,
    output_path: Path,
    show_orientation: bool,
    hide_labels: bool,
    camera_scale: float,
    orientation_scale: float,
) -> None:
    title, subtitle, payload = build_scene_payload(
        metadata=metadata,
        frame_idx=frame_idx,
        show_orientation=show_orientation,
        hide_labels=hide_labels,
        camera_scale=camera_scale,
        orientation_scale=orientation_scale,
    )
    html = HTML_TEMPLATE.substitute(
        title=title,
        subtitle=subtitle,
        scene_json=json.dumps(payload, separators=(",", ":")),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata_path)
    default_output = args.metadata_path.parent / f"scene_3d_frame_{args.frame:05d}.html"
    output_path = args.output or default_output
    render_scene_html(
        metadata=metadata,
        frame_idx=args.frame,
        output_path=output_path,
        show_orientation=args.show_orientation,
        hide_labels=args.hide_labels,
        camera_scale=args.camera_scale,
        orientation_scale=args.orientation_scale,
    )
    print(output_path)


if __name__ == "__main__":
    main()
