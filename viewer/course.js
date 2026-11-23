import * as THREE from 'three';
import { GATE_OPENING } from './gates.js';

// Placeholder until the sim exists: the drone just follows a fixed loop at constant speed.
// No physics, nothing trained.

const GRAVITY = 9.81;
const SPEED = 6; // m/s

// gates in flying order (m, y up)
const GATE_POINTS = [
  [3.5, 1.3, -3.8], [3.8, 4.5, -0.5], [1, 8, 1.5], [-2.5, 5.5, -1.5], [-3.8, 1.5, 2.5],
  [-0.5, 1.2, 4], [3.5, 2.8, 3], [1.5, 6.5, -1.5], [-2, 8, -3.8], [-3.6, 3.5, -3], [-0.5, 1.4, -0.8],
];

export function createScriptedFlight() {
  const path = new THREE.CatmullRomCurve3(GATE_POINTS.map((p) => new THREE.Vector3(...p)), true, 'centripetal');
  path.arcLengthDivisions = 800;
  const lapLength = path.getLength();

  const gates = GATE_POINTS.map((p, i) => ({
    position: new THREE.Vector3(...p),
    direction: path.getTangent(i / GATE_POINTS.length).normalize(),
  }));

  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const tangent = new THREE.Vector3();
  const tangentAhead = new THREE.Vector3();
  const tangentBehind = new THREE.Vector3();
  const thrust = new THREE.Vector3();
  const bodyUp = new THREE.Vector3();
  const bodyForward = new THREE.Vector3();
  const bodySide = new THREE.Vector3();
  const attitude = new THREE.Matrix4();
  const fromGate = new THREE.Vector3();

  let lapFraction = 0.92; // partway along the last leg, so gate 0 is still ahead
  let nextGate = 0;
  let previousSide = -1;

  const sideOfGate = (index) => fromGate.subVectors(position, gates[index].position).dot(gates[index].direction);

  // a quad turns by tilting its thrust, so body up = path acceleration + gravity.
  // Acceleration is the change in tangent over a small step either side (h is a fraction of a lap)
  function setAttitude() {
    const h = 0.002;
    path.getTangentAt((lapFraction + h) % 1, tangentAhead);
    path.getTangentAt((lapFraction - h + 1) % 1, tangentBehind);
    thrust.subVectors(tangentAhead, tangentBehind).multiplyScalar((SPEED * SPEED) / (2 * h * lapLength));
    thrust.y += GRAVITY;
    bodyUp.copy(thrust).normalize();
    bodyForward.set(tangent.x, 0, tangent.z).normalize();
    bodyForward.addScaledVector(bodyUp, -bodyForward.dot(bodyUp)).normalize();
    bodySide.crossVectors(bodyForward, bodyUp);
    quaternion.setFromRotationMatrix(attitude.makeBasis(bodyForward, bodyUp, bodySide));
  }

  // passed = crossed the gate's plane going the right way, close enough to the centre
  function checkGate() {
    const side = sideOfGate(nextGate);
    const close = position.distanceTo(gates[nextGate].position) < GATE_OPENING;
    if (previousSide < 0 && side >= 0 && close) {
      nextGate = (nextGate + 1) % gates.length;
      previousSide = sideOfGate(nextGate);
    } else {
      previousSide = side;
    }
  }

  // gives back the same position/quaternion objects every call, copy them if you want to keep one
  function step(dt) {
    lapFraction = (lapFraction + (SPEED * dt) / lapLength) % 1;
    path.getPointAt(lapFraction, position);
    path.getTangentAt(lapFraction, tangent);
    setAttitude();
    checkGate();
    return { position, quaternion, nextGate };
  }

  return { gates, step };
}
