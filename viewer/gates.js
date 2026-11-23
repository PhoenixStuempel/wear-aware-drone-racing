import * as THREE from 'three';

export const GATE_OPENING = 1.2; // m, width of the hole

function roundedSquare(path, size, radius) {
  const h = size / 2;
  const r = radius;
  path.moveTo(-h + r, -h);
  path.lineTo(h - r, -h);
  path.quadraticCurveTo(h, -h, h, -h + r);
  path.lineTo(h, h - r);
  path.quadraticCurveTo(h, h, h - r, h);
  path.lineTo(-h + r, h);
  path.quadraticCurveTo(-h, h, -h, h - r);
  path.lineTo(-h, -h + r);
  path.quadraticCurveTo(-h, -h, -h + r, -h);
  return path;
}

function squareRing(outer, inner, outerRadius, innerRadius) {
  const shape = roundedSquare(new THREE.Shape(), outer, outerRadius);
  shape.holes.push(roundedSquare(new THREE.Path(), inner, innerRadius));
  return shape;
}

// polygon with each corner cut back by radius and rounded off with a curve
function softPolygon(points, radius) {
  const shape = new THREE.Shape();
  const n = points.length;
  points.forEach(([x, y], i) => {
    const [px, py] = points[(i + n - 1) % n];
    const [qx, qy] = points[(i + 1) % n];
    const toPrevious = Math.hypot(px - x, py - y);
    const toNext = Math.hypot(qx - x, qy - y);
    const ax = x + ((px - x) / toPrevious) * radius;
    const ay = y + ((py - y) / toPrevious) * radius;
    const bx = x + ((qx - x) / toNext) * radius;
    const by = y + ((qy - y) / toNext) * radius;
    if (i === 0) shape.moveTo(ax, ay);
    else shape.lineTo(ax, ay);
    shape.quadraticCurveTo(x, y, bx, by);
  });
  shape.closePath();
  return shape;
}

const rotate2d = (points, angle) => points.map(([x, y]) => [
  x * Math.cos(angle) - y * Math.sin(angle),
  x * Math.sin(angle) + y * Math.cos(angle),
]);

// one chevron, copied to all 4 sides pointing in
const CHEVRON = [[-0.15, 0.27], [0, 0.135], [0.15, 0.27], [0.15, 0.345], [0, 0.21], [-0.15, 0.345]];
const chevronsGeometry = new THREE.ShapeGeometry(
  [0, 1, 2, 3].map((i) => softPolygon(rotate2d(CHEVRON, (i * Math.PI) / 2), 0.016)), 12,
);
const bandGeometry = new THREE.ShapeGeometry(squareRing(1.37, GATE_OPENING, 0.21, 0.16), 16);
const frameGeometry = new THREE.ExtrudeGeometry(squareRing(1.5, 1.36, 0.28, 0.2), {
  depth: 0.08, bevelEnabled: true, bevelThickness: 0.012, bevelSize: 0.012, bevelSegments: 3, curveSegments: 16,
});
frameGeometry.translate(0, 0, -0.04);

const frameMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x2a2c30, roughness: 0.5, clearcoat: 0.25, clearcoatRoughness: 0.4, envMapIntensity: 0.5,
});
// lookAt points the gate's +z the way you're meant to fly through, so coming in the right way
// you see the back faces (green) and the wrong way the front faces (red)
const entrySide = new THREE.MeshBasicMaterial({ color: 0x2fd07a, side: THREE.BackSide });
const wrongSide = new THREE.MeshBasicMaterial({ color: 0xe5484d, side: THREE.FrontSide });

// direction is the unit vector you fly through it along
export function buildGate(position, direction) {
  const gate = new THREE.Group();
  gate.position.copy(position);
  gate.lookAt(position.clone().add(direction));

  const frame = new THREE.Mesh(frameGeometry, frameMaterial);
  frame.castShadow = true;
  gate.add(frame);

  // band and chevrons, a green and a red copy of each
  for (const geometry of [bandGeometry, chevronsGeometry]) {
    gate.add(new THREE.Mesh(geometry, entrySide), new THREE.Mesh(geometry, wrongSide));
  }
  return gate;
}
