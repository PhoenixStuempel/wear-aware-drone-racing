import * as THREE from 'three';

export const BOX_SIZE = 10; // m, the course stays inside this

// fill, spot and scene.environmentIntensity (main.js) were tuned together so the floor is near
// white and shadows still show. More fill washes the shadows out
function buildLights() {
  const lights = new THREE.Group();
  lights.add(new THREE.HemisphereLight(0xffffff, 0xedeef0, 0.12)); // sky fill

  // spotlight, the only thing casting shadows
  const spot = new THREE.SpotLight(0xfffbf6, 2.1, 0, 0.62, 1, 0);
  spot.position.set(3.2, 12, -2.4);
  spot.castShadow = true;
  spot.shadow.mapSize.set(4096, 4096);
  spot.shadow.radius = 5;
  spot.shadow.blurSamples = 18;
  spot.shadow.bias = -0.001; // any smaller and the floor gets striped (shadow acne)
  spot.shadow.camera.near = 4;
  spot.shadow.camera.far = 20;
  lights.add(spot, spot.target);
  return lights;
}

function buildBox() {
  const box = new THREE.Group();

  // floor
  const floor = new THREE.Mesh(
    new THREE.BoxGeometry(BOX_SIZE, 0.08, BOX_SIZE),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.7, envMapIntensity: 0.2 }),
  );
  floor.position.y = -0.04;
  floor.receiveShadow = true;
  box.add(floor);

  // grid, 1 m squares
  const grid = new THREE.GridHelper(BOX_SIZE, BOX_SIZE, 0xd5d8dc, 0xd5d8dc);
  grid.position.y = 0.003;
  box.add(grid);

  // glass walls + edges. Box faces go +x, -x, +y, -y, +z, -z, bottom one hidden so it doesn't
  // cover the floor
  const glass = new THREE.MeshBasicMaterial({
    color: 0xc5d3df, transparent: true, opacity: 0.045, side: THREE.DoubleSide, depthWrite: false,
  });
  const hidden = new THREE.MeshBasicMaterial({ visible: false });
  const shell = new THREE.BoxGeometry(BOX_SIZE, BOX_SIZE, BOX_SIZE);
  const walls = new THREE.Mesh(shell, [glass, glass, glass, hidden, glass, glass]);
  walls.position.y = BOX_SIZE / 2 + 0.004;
  walls.add(new THREE.LineSegments(new THREE.EdgesGeometry(shell), new THREE.LineBasicMaterial({ color: 0xafb5bd })));
  box.add(walls);

  return box;
}

export function buildArena() {
  const arena = new THREE.Group();
  arena.add(buildLights(), buildBox());
  return arena;
}
