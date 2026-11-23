import * as THREE from 'three';

const ARM = 0.17; // m, body centre to motor axis
const PROP_RADIUS = 0.098; // m

// real rotor speeds alias at 60 fps (props crawl or spin backwards), so spin slow and let the
// blur disc do the rest
const SPIN_RATE = 50; // rad/s

const BODY = new THREE.MeshPhysicalMaterial({
  color: 0x202226, roughness: 0.56, clearcoat: 0.25, clearcoatRoughness: 0.42,
  envMapIntensity: 0.45, flatShading: true,
});
const FRAME = new THREE.MeshStandardMaterial({ color: 0x66686c, roughness: 0.42, metalness: 0.25, envMapIntensity: 0.8 });
const MOTOR_BASE = new THREE.MeshStandardMaterial({ color: 0x9a9c9f, metalness: 1, roughness: 0.42, envMapIntensity: 0.75 });
const MOTOR_BELL = new THREE.MeshStandardMaterial({ color: 0x5b5d61, metalness: 1, roughness: 0.38, envMapIntensity: 0.8 });
const HUB = new THREE.MeshStandardMaterial({ color: 0x202124, metalness: 0.9, roughness: 0.3 });
const PROP = new THREE.MeshPhysicalMaterial({
  color: 0x121315, roughness: 0.5, clearcoat: 0.15, clearcoatRoughness: 0.45, envMapIntensity: 0.4,
});
const LEG = new THREE.MeshStandardMaterial({ color: 0x74767a, roughness: 0.62 });
const WEAR_RING = new THREE.MeshBasicMaterial({ color: 0x2fd07a });
const BLUR = new THREE.MeshBasicMaterial({
  color: 0x5a5c5f, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false,
});

function addMesh(parent, geometry, material, x = 0, y = 0, z = 0) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  parent.add(mesh);
  return mesh;
}

const disc = (topRadius, bottomRadius, height) => new THREE.CylinderGeometry(topRadius, bottomRadius, height, 40);

// outline extruded up and centred on y = 0, small bevel softens the corners
function slab(outline, depth, bevel) {
  const shape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 2,
  });
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, -depth / 2, 0);
  return geometry;
}

const BODY_OUTLINE = [
  [0.094, 0.014], [0.062, 0.033], [-0.05, 0.033], [-0.094, 0.02],
  [-0.094, -0.02], [-0.05, -0.033], [0.062, -0.033], [0.094, -0.014],
];

function bladeGeometry() {
  const outline = [[0, 0.0055], [0.03, 0.0125], [0.072, 0.0085], [PROP_RADIUS, 0.003], [PROP_RADIUS, -0.002], [0.05, -0.006], [0, -0.005]];
  const shape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: 0.0018, bevelEnabled: true, bevelThickness: 0.0004, bevelSize: 0.0004, bevelSegments: 1,
  });
  geometry.rotateX(-Math.PI / 2);
  return geometry;
}

// flat strip swept along a curve. +x is out along the arm, so it starts straight down under
// the motor and bends in under the body. Each side gets its own vertices so the corners stay
// sharp (shared vertices would get averaged normals)
function legGeometry() {
  const STATIONS = 40;
  const path = new THREE.CubicBezierCurve(
    new THREE.Vector2(0.0165, -0.004), new THREE.Vector2(0.0165, -0.036),
    new THREE.Vector2(0.0145, -0.065), new THREE.Vector2(0, -0.082),
  );
  const rings = [];
  for (let s = 0; s <= STATIONS; s++) {
    const t = s / STATIONS;
    const centre = path.getPoint(t);
    const along = path.getTangent(t);
    const halfThickness = THREE.MathUtils.lerp(0.005, 0.0024, t) / 2;
    const halfWidth = THREE.MathUtils.lerp(0.0135, 0.0048, t) / 2;
    const corner = (across, side) => [
      centre.x - along.y * halfThickness * across,
      centre.y + along.x * halfThickness * across,
      side * halfWidth,
    ];
    rings.push([corner(1, 1), corner(1, -1), corner(-1, -1), corner(-1, 1)]);
  }

  const position = [];
  const index = [];
  for (let side = 0; side < 4; side++) {
    const start = position.length / 3;
    rings.forEach((ring) => position.push(...ring[side], ...ring[(side + 1) % 4]));
    for (let s = 0; s < STATIONS; s++) {
      const v = start + s * 2;
      index.push(v, v + 2, v + 1, v + 1, v + 2, v + 3);
    }
  }
  // end caps, far one wound the other way so both face out
  [[rings[0], false], [rings[STATIONS], true]].forEach(([ring, flip]) => {
    const start = position.length / 3;
    ring.forEach((p) => position.push(...p));
    if (flip) index.push(start, start + 2, start + 1, start, start + 3, start + 2);
    else index.push(start, start + 1, start + 2, start, start + 2, start + 3);
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
  geometry.setIndex(index);
  geometry.computeVertexNormals();
  return geometry;
}

// +x is forward, +y up. Rotor i is at 45 + 90i degrees from +x towards +z, neighbours spin
// opposite ways. The sim has to use the same motor order or wear shows on the wrong motor.
export function buildDrone() {
  const root = new THREE.Group();
  addMesh(root, slab(BODY_OUTLINE, 0.024, 0.0035), BODY, 0, 0.009, 0); // body

  const blade = bladeGeometry();
  const leg = legGeometry();
  const ringGeometry = new THREE.TorusGeometry(0.025, 0.0058, 12, 40);

  const rotors = [0, 1, 2, 3].map((i) => {
    const angle = THREE.MathUtils.degToRad(45 + 90 * i);
    const dx = Math.cos(angle);
    const dz = Math.sin(angle);

    // arm, starts at the body edge
    const armStart = 0.034;
    const armLength = ARM - armStart;
    const arm = addMesh(root, new THREE.BoxGeometry(armLength, 0.0085, 0.0165), FRAME,
      dx * (armStart + armLength / 2), 0.009, dz * (armStart + armLength / 2));
    arm.rotation.y = -angle;

    // motor
    const motor = new THREE.Group();
    motor.position.set(dx * ARM, 0.009, dz * ARM);
    root.add(motor);
    addMesh(motor, disc(0.0195, 0.0205, 0.0095), MOTOR_BASE, 0, -0.001, 0);
    addMesh(motor, disc(0.0155, 0.0165, 0.0165), MOTOR_BELL, 0, 0.012, 0);
    addMesh(motor, disc(0.0075, 0.0075, 0.009), HUB, 0, 0.0245, 0);
    const cap = addMesh(motor, new THREE.SphereGeometry(0.0062, 20, 12), HUB, 0, 0.0285, 0);
    cap.scale.y = 0.72;

    addMesh(motor, leg, LEG).rotation.y = -angle; // landing leg

    // wear ring, just green until the sim gives it a wear value
    const wearRing = new THREE.Mesh(ringGeometry, WEAR_RING);
    wearRing.rotation.x = Math.PI / 2;
    wearRing.position.y = -0.001;
    motor.add(wearRing);

    // prop, 2 blades
    const prop = new THREE.Group();
    prop.position.y = 0.024;
    motor.add(prop);
    [0, Math.PI].forEach((turn) => {
      const holder = new THREE.Group();
      holder.rotation.y = turn;
      prop.add(holder);
      addMesh(holder, blade, PROP).rotation.x = 0.07;
    });

    // blur disc
    const blur = new THREE.Mesh(new THREE.CircleGeometry(PROP_RADIUS, 48), BLUR);
    blur.rotation.x = -Math.PI / 2;
    blur.position.y = 0.0255;
    motor.add(blur);

    return { prop, direction: i % 2 === 0 ? 1 : -1 };
  });

  return {
    root,
    spinProps(dt) {
      rotors.forEach((r) => { r.prop.rotation.y += r.direction * SPIN_RATE * dt; });
    },
  };
}
