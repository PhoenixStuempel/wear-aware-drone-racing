import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { BOX_SIZE, buildArena } from './arena.js';
import { createScriptedFlight } from './course.js';
import { buildDrone } from './drone.js';
import { buildGate } from './gates.js';

const MAX_DT = 0.05; // s, so the drone doesn't jump ahead after the tab was hidden

const stage = document.getElementById('stage');
stage.replaceChildren(); // removes the "didn't start" message

// renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.VSMShadowMap;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe4e6e9);
// room reflections, without them the metal and clearcoat just look flat grey
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.03).texture;
scene.environmentIntensity = 0.25; // tuned with the lights in arena.js
scene.add(buildArena());

const flight = createScriptedFlight();
const gates = flight.gates.map((g) => buildGate(g.position, g.direction));
scene.add(...gates);

const drone = buildDrone();
scene.add(drone.root);

// camera
const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 120);
const controls = new OrbitControls(camera, renderer.domElement);
controls.minDistance = 0.3;
controls.maxDistance = 45;

const up = new THREE.Vector3(0, 1, 0);
const heading = new THREE.Vector3();
const moved = new THREE.Vector3();

function showFlight(state) {
  drone.root.position.copy(state.position);
  drone.root.quaternion.copy(state.quaternion);
  // only the next 2 gates are shown, so it's obvious which one is next
  gates.forEach((gate, i) => {
    gate.visible = i === state.nextGate || i === (state.nextGate + 1) % gates.length;
  });
}

// tracking cam starts 2.2 m behind the drone (along the heading, ignoring tilt) and 0.8 m up
function behindDrone() {
  heading.set(1, 0, 0).applyQuaternion(drone.root.quaternion);
  heading.y = 0;
  heading.normalize();
  return new THREE.Vector3().addScaledVector(heading, -2.2).addScaledVector(up, 0.8);
}

// free cam starts looking at the middle of the box from a high corner, far enough back that the
// whole box fits (its corners sit BOX_SIZE * sqrt(3) / 2 from the middle)
const boxMiddle = new THREE.Vector3(0, BOX_SIZE / 2, 0);
function wholeBox() {
  const radius = (BOX_SIZE * Math.sqrt(3)) / 2;
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const fit = Math.min(halfFov, Math.atan(Math.tan(halfFov) * camera.aspect)); // narrower side
  return new THREE.Vector3(1, 0.75, 1.3).normalize().multiplyScalar((radius / Math.sin(fit)) * 1.05);
}

// camera buttons. Each mode remembers where you left it: tracking as an offset from the drone,
// free as a fixed spot
let cameraMode = 'tracking';
let trackingOffset = null;
let freeView = null;
for (const button of document.querySelectorAll('[data-camera]')) {
  button.addEventListener('click', () => {
    const mode = button.dataset.camera;
    if (mode === cameraMode) return;
    if (mode === 'free') {
      trackingOffset = camera.position.clone().sub(controls.target);
      freeView ??= { position: wholeBox().add(boxMiddle), target: boxMiddle.clone() };
      camera.position.copy(freeView.position);
      controls.target.copy(freeView.target);
    } else {
      freeView = { position: camera.position.clone(), target: controls.target.clone() };
      controls.target.copy(drone.root.position);
      camera.position.copy(drone.root.position).add(trackingOffset);
    }
    controls.update();
    cameraMode = mode;
    for (const other of document.querySelectorAll('[data-camera]')) {
      other.setAttribute('aria-pressed', String(other === button));
    }
  });
}

// tracking keeps your orbit and moves it along with the drone. Free doesn't move on its own
function moveCamera(previous, current) {
  if (cameraMode === 'tracking') {
    moved.subVectors(current, previous);
    camera.position.add(moved);
    controls.target.add(moved);
  }
  controls.update();
}

function resize() {
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
resize();

// start with the camera already behind the drone
showFlight(flight.step(0));
camera.position.copy(drone.root.position).add(behindDrone());
controls.target.copy(drone.root.position);
controls.update();

// frame loop. dt only comes from the loop's own timestamps, mixing in performance.now() gave a
// negative first step
const previous = new THREE.Vector3();
let lastTime = null;
function frame(time) {
  const dt = lastTime === null ? 0 : Math.min((time - lastTime) / 1000, MAX_DT);
  lastTime = time;
  previous.copy(drone.root.position);
  showFlight(flight.step(dt));
  drone.spinProps(dt);
  moveCamera(previous, drone.root.position);
  renderer.render(scene, camera);
}
// three.js stops calling the loop for good if it throws once, so log and keep going
renderer.setAnimationLoop((time) => {
  try {
    frame(time);
  } catch (e) {
    console.error(e);
  }
});
