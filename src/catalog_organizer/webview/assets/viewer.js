// Catalogue web viewer (product page: viewer.html?p=<file_id>).
//
// A port of the server render (PyVista/VTK) to three.js, NOT a re-imagining of
// it: the same GLB data, the same facet-plane stone tracer, and the tone curve
// that was MEASURED off the server (RMS 0.0011 against its output).
//
// Pipeline, deliberately the same shape as a luxury-retail viewer:
//   scene -> half-float HDR target (jittered sub-pixel camera) -> running mean
//   over up to 16 frames -> ONE full-screen pass that tone-maps, so the stones
//   and the metal share a single operator.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const Q = new URLSearchParams(location.search);
const AUTO = Q.get('auto') === '1';
const HOST = Q.get('host') === '1';                     // embedded in the catalogue app: it drives the viewer
const SAVE = Q.get('save') || 'web.png';
const FIXED = parseInt(Q.get('size') || '0', 10);     // square CSS px; DPR forced to 1
const VIEW = Q.get('view') || 'iso';
const MAX_FRAMES = parseInt(Q.get('frames') || '16', 10);
const BASE = Q.get('base') || '';
const P = Q.get('p');                                   // which product
if (!P) { location.replace('index.html'); throw new Error('no product'); }
const getJSON = async u => (await fetch(BASE + u)).json();
const getBuf = async u => (await fetch(BASE + u)).arrayBuffer();
const meta = await getJSON('data/_shared/meta.json');          // needed first: it carries the look defaults
const GEM_GAIN = parseFloat(Q.get('gemgain') ?? meta.defaults?.gemgain ?? '1');   // stone brightness multiplier
const GEM_LIFT = parseFloat(Q.get('gemlift') ?? meta.defaults?.gemlift ?? '0');   // radiance a stone never falls below
// Look parameters (all optional): the target is the look of
// a luxury-retail viewer, measured, so each is a knob toward it.
const D = meta.defaults || {};                             // look presets shipped with the piece
const TONE = Q.get('tone') || D.tone || 'vtk';                       // 'vtk' = measured server curve, 'three' = three.js ACES
const EXPOSURE = parseFloat(Q.get('exp') || '0');          // 0 = the mode's default
const ENVI = parseFloat(Q.get('envi') || D.envi || '1');             // metal environment intensity
const GOLD = Q.get('gold') || D.gold || null;                                // 'r,g,b' linear F0 override
const ROUGH = Q.get('rough') ?? D.rough ?? null;                              // roughness override
const AO = parseFloat(Q.get('ao') ?? D.ao ?? '0');           // baked vertex occlusion strength, 0 = off
const BG = Q.get('bg') || D.bg || null;                                    // flat page colour, hex without '#'
if (BG) { document.documentElement.style.background = '#' + BG; document.body.style.background = '#' + BG; }

// ---------------------------------------------------------------- diagnostics
const errBox = document.getElementById('err');
function report(msg) {
  console.error(msg);
  errBox.style.display = 'block';
  errBox.textContent += msg + '\n';
  if (AUTO) fetch('/log', { method: 'POST', body: String(msg) }).catch(() => {});
}
// three.js reports shader compile failures through console.error, not an
// 'error' event -- forward the console too, or a headless run fails silently.
for (const k of ['error', 'warn']) {
  const orig = console[k].bind(console);
  console[k] = (...a) => { orig(...a); if (AUTO) fetch('/log', { method: 'POST', body: k + ': ' + a.map(String).join(' ').slice(0, 3000) }).catch(() => {}); };
}
window.addEventListener('error', e => report('error: ' + e.message));
window.addEventListener('unhandledrejection', e => report('promise: ' + (e.reason && e.reason.stack || e.reason)));

// ---------------------------------------------------------------- data
const envMetalBuf = new Uint16Array(await getBuf('data/_shared/env_metal.f16'));
const envGemBuf = new Uint16Array(await getBuf('data/_shared/env_gem.f16'));
const glbBuf = await getBuf('data/' + encodeURIComponent(P) + '/piece.glb');

function halfTexture(data, w, h) {
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.flipY = false;
  t.needsUpdate = true;
  return t;
}

// ---------------------------------------------------------------- renderer
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({
  antialias: false, alpha: true, premultipliedAlpha: true,
  preserveDrawingBuffer: true, powerPreference: 'high-performance',
});
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setClearColor(0x000000, 0);
stage.appendChild(renderer.domElement);
if (FIXED) { renderer.domElement.style.width = FIXED + 'px'; renderer.domElement.style.height = FIXED + 'px'; stage.style.position = 'absolute'; }

const DPR = FIXED ? 1 : Math.min(window.devicePixelRatio || 1, 2);
let W = 0, H = 0;

const rtOpts = { type: THREE.HalfFloatType, format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false };
let frameRT, ping, pong;
function makeTargets() {
  [frameRT, ping, pong].forEach(t => t && t.dispose());
  frameRT = new THREE.WebGLRenderTarget(W, H, { ...rtOpts, depthBuffer: true, samples: 4 });
  ping = new THREE.WebGLRenderTarget(W, H, rtOpts);
  pong = new THREE.WebGLRenderTarget(W, H, rtOpts);
}

// ---------------------------------------------------------------- scene
const scene = new THREE.Scene();
const gltf = await new Promise((res, rej) => new GLTFLoader().parse(glbBuf, '', res, rej));
const info = gltf.scene.userData;           // scene `extras` from the GLB
scene.add(gltf.scene);

// Metal: a plain PBR material. Its base colour is F0 (a reflectance), so it is
// set as LINEAR values, not parsed as an sRGB paint swatch.
const hexLinear = hex => new THREE.Color().setRGB(
  parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255,
  parseInt(hex.slice(5, 7), 16) / 255, THREE.LinearSRGBColorSpace);

const pmrem = new THREE.PMREMGenerator(renderer);
const envMetalTex = halfTexture(envMetalBuf, meta.envMetal.w, meta.envMetal.h);
envMetalTex.mapping = THREE.EquirectangularReflectionMapping;
scene.environment = pmrem.fromEquirectangular(envMetalTex).texture;

// DoubleSide: VTK does not cull back faces, and a Brep's per-face meshes do not
// agree on winding, so FrontSide would punch holes in the metal.
const metalMat = new THREE.MeshStandardMaterial({ metalness: 1, roughness: 0.12, side: THREE.DoubleSide });
// Gold relies on a baked occlusion map; this piece carries it per vertex (`_AO`).
// Applied to the image-based light only: there are no direct lights, so that IS all the light.
metalMat.onBeforeCompile = (sh) => {
  sh.uniforms.uAo = { value: AO };
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', '#include <common>\nattribute float _ao;\nvarying float vAo;')
    .replace('#include <begin_vertex>', '#include <begin_vertex>\nvAo = _ao;');
  sh.fragmentShader = sh.fragmentShader
    .replace('#include <common>', '#include <common>\nvarying float vAo;\nuniform float uAo;')
    .replace('#include <aomap_fragment>',
             'float aoV = max(pow(max(vAo, 0.001), uAo), 0.3);\nreflectedLight.indirectDiffuse *= aoV;\nreflectedLight.indirectSpecular *= aoV;');
};
let metalKey = Q.get('metal') || 'yellow_gold';
function applyMetal(key) {
  metalKey = key;
  const m = meta.metals[key];
  metalMat.color.copy(hexLinear(m.color));
  metalMat.metalness = m.metallic;
  metalMat.roughness = m.roughness;
  const own = (D.metals || {})[key] || (key === 'yellow_gold' ? GOLD : null);
  if (own) metalMat.color.setRGB(...own.split(',').map(Number), THREE.LinearSRGBColorSpace);
  if (ROUGH !== null) metalMat.roughness = parseFloat(ROUGH);
  metalMat.envMapIntensity = ENVI;
  metalMat.needsUpdate = true;
  restart();
}

// Stones: the analytic facet-plane tracer from render/gem_trace.py, as a
// ShaderMaterial. The facets are baked into the source as constants (one compile
// serves every stone of the cut); where each stone sits arrives as uniforms.
const cut = info.cut || { points: [[0, 0, 1]], normals: [[0, 0, 1]], planes: 0 };
const f6 = x => x.toFixed(6);
const vec3s = a => a.map(p => `vec3(${f6(p[0])},${f6(p[1])},${f6(p[2])})`).join(',\n');
const NPLANE = cut.points.length;

const GEM_FRAG = /* glsl */`
precision highp float;
in vec3 vWorld;
uniform sampler2D uEnv;
uniform vec3 uRight, uUp, uAxis, uCentre;
uniform float uRadius, uF0, uExposureRatio, uLift;
uniform mat3 uEnvRot;
uniform vec3 uEta, uAbsorb, uBoost;

const vec3 PA[${NPLANE}] = vec3[${NPLANE}](
${vec3s(cut.points)}
);
const vec3 PN[${NPLANE}] = vec3[${NPLANE}](
${vec3s(cut.normals)}
);

vec3 lookup(vec3 wWorld) {
  vec3 wv = uEnvRot * wWorld;
  vec3 s = normalize(vec3(wv.x, wv.y, -wv.z));
  vec2 uv = vec2(atan(s.z, s.x) / 6.2831853 + 0.5,
                 1.0 - acos(clamp(s.y, -1.0, 1.0)) / 3.14159265);
  return textureLod(uEnv, uv, 0.0).rgb;
}

void main() {
  vec3 Iv = normalize(vWorld - cameraPosition);
  vec3 Nv = cross(dFdx(vWorld), dFdy(vWorld));
  Nv = (dot(Nv, Nv) > 1e-20) ? normalize(Nv) : -Iv;
  if (dot(Nv, Iv) > 0.0) Nv = -Nv;

  // The stone's own frame: centred, radius 1, table along +Z.
  mat3 toWorld = mat3(uRight, uUp, uAxis);
  mat3 toLocal = transpose(toWorld);
  vec3 localDir = normalize(toLocal * Iv);
  vec3 localNrm = normalize(toLocal * Nv);
  vec3 localOrg = (toLocal * (vWorld - uCentre)) / uRadius;

  vec3 through = vec3(0.0);
  for (int ch = 0; ch < 3; ++ch) {
    float eta = uEta[ch];
    vec3 d = refract(localDir, localNrm, 1.0 / eta);
    if (dot(d, d) < 1e-6) { d = reflect(localDir, localNrm); }
    vec3 o = localOrg;
    float pathLength = 0.0;
    for (int bounce = 0; bounce < BOUNCES; ++bounce) {
      // Convex body: the nearest plane the ray is heading toward IS the exit.
      float theta = 1.0e9;
      vec3 hitN = vec3(0.0, 0.0, 1.0);
      for (int i = 0; i < ${NPLANE}; ++i) {
        float dn = dot(d, PN[i]);
        if (dn > 1.0e-6) {
          float t = (dot(PA[i], PN[i]) - dot(o, PN[i])) / dn;
          if (t > 1.0e-4 && t < theta) { theta = t; hitN = PN[i]; }
        }
      }
      if (theta > 1.0e8) break;
      o += d * theta;
      pathLength += theta;
      vec3 outDir = refract(d, hitN, eta);
      if (dot(outDir, outDir) > 1.0e-6) { d = outDir; break; }
      d = reflect(d, hitN);                       // total internal reflection
    }
    vec3 w = normalize(toWorld * d);
    through[ch] = lookup(w)[ch] * exp(-uAbsorb[ch] * pathLength);
  }

  vec3 mirror = lookup(reflect(Iv, Nv));
  float ct = clamp(dot(-Iv, Nv), 0.0, 1.0);
  float F = uF0 + (1.0 - uF0) * pow(1.0 - ct, 5.0);
  // uExposureRatio: the server tone-maps stones at a lower exposure than metal;
  // one shared tone pass here, so the difference is applied to the radiance.
  gl_FragColor = vec4(mix(through, mirror, F) * uBoost * uExposureRatio + uLift, 1.0);
}`;

const GEM_VERT = /* glsl */`
out vec3 vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

const envGemTex = halfTexture(envGemBuf, meta.envGem.w, meta.envGem.h);
envGemTex.wrapS = THREE.RepeatWrapping;
envGemTex.wrapT = THREE.ClampToEdgeWrapping;
const viewRows = info.pose.viewRows;
const envRot = new THREE.Matrix3().set(...viewRows.flat());      // world -> initial view

const stoneMeshes = [];
const crowded = gltf.scene.children.filter(o => o.userData.role === 'gem').length >= meta.crowdedCount;
const bounces = crowded ? meta.crowdedBounces : meta.stoneBounces;
const gemSource = GEM_FRAG.replace('BOUNCES', String(bounces));

const opaqueMat = new THREE.MeshStandardMaterial({ metalness: 1, roughness: 0.08 });
let stoneKey = Q.get('stone') || 'white';

function gemMaterial(frame) {
  const b = frame.basis;
  return new THREE.ShaderMaterial({
    vertexShader: GEM_VERT, fragmentShader: gemSource,
    side: THREE.DoubleSide,
    uniforms: {
      uEnv: { value: envGemTex },
      uRight: { value: new THREE.Vector3(...b[0]) },
      uUp: { value: new THREE.Vector3(...b[1]) },
      uAxis: { value: new THREE.Vector3(...b[2]) },
      uCentre: { value: new THREE.Vector3(...frame.centre) },
      uRadius: { value: frame.radius },
      uEnvRot: { value: envRot },
      uF0: { value: 0.17 }, uLift: { value: GEM_LIFT }, uExposureRatio: { value: GEM_GAIN * meta.gemExposure / meta.metalExposure },
      uEta: { value: new THREE.Vector3(2.4, 2.4, 2.4) },
      uAbsorb: { value: new THREE.Vector3() }, uBoost: { value: new THREE.Vector3(1, 1, 1) },
    },
  });
}

function applyStone(key) {
  stoneKey = key;
  const s = meta.stones[key];
  for (const mesh of stoneMeshes) {
    if (s.opaque) {
      opaqueMat.color.copy(hexLinear(s.color)); opaqueMat.roughness = s.roughness;
      mesh.material = opaqueMat; continue;
    }
    mesh.material = mesh.userData.gemMat;
    const u = mesh.material.uniforms;
    // (n - d/2, n, n + d/2): the same per-channel indices the server tracer uses.
    u.uEta.value.set(s.ior - 0.5 * s.dispersion, s.ior, s.ior + 0.5 * s.dispersion);
    u.uF0.value = ((s.ior - 1) / (s.ior + 1)) ** 2;
    u.uAbsorb.value.set(...s.absorption);
  }
  restart();
}

gltf.scene.traverse(o => {
  if (!o.isMesh) return;
  if (o.userData.role === 'metal') o.material = metalMat;
  else if (o.userData.role === 'gem') {
    o.userData.gemMat = gemMaterial(o.userData.frame);
    stoneMeshes.push(o);
  }
});

// ---------------------------------------------------------------- camera
const pose = info.pose;
const fov = pose.fov;
const distance = (info.extentMm * pose.zoom) / Math.tan(THREE.MathUtils.degToRad(fov / 2));
const camera = new THREE.PerspectiveCamera(fov, 1, distance * 0.2, distance * 4);
camera.up.set(...pose.up);
const dir = new THREE.Vector3(...(pose[VIEW] || [0, 0, 1])).normalize();   // exported products carry `cameraPos`, which wins below
if (pose.cameraPos) camera.position.set(...pose.cameraPos); else camera.position.copy(dir.multiplyScalar(distance));
const camDist = camera.position.length();
camera.near = camDist * 0.3; camera.far = camDist * 3; camera.updateProjectionMatrix();
camera.lookAt(0, 0, 0);

const CTL = D.controls || {};                         // orbit limits shipped with the piece
let controls = null;
function makeControls() {
  if (controls) controls.dispose();
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableDamping = !!CTL.damping;
  if (CTL.damping) controls.dampingFactor = CTL.damping;
  controls.minDistance = camDist * (CTL.min ?? 0.35);
  controls.maxDistance = camDist * (CTL.max ?? 2.5);
  if (CTL.polar) controls.maxPolarAngle = CTL.polar;      // the pole stays out of reach
  if (CTL.zoom) controls.zoomSpeed = CTL.zoom * 6.7;       // three's zoomSpeed 1 ~ one 0.95 step per notch; theirs is a dolly of .15
  controls.rotateSpeed = 1.0;
  controls.target.set(0, 0, 0);
  controls.update();
  controls.addEventListener('change', () => restart());
}
makeControls();

// ---------------------------------------------------------------- post
const quadGeo = new THREE.PlaneGeometry(2, 2);
const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const QUAD_VERT = /* glsl */`out vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const accumMat = new THREE.ShaderMaterial({
  depthTest: false, depthWrite: false, blending: THREE.NoBlending,
  uniforms: { tPrev: { value: null }, tNew: { value: null }, uW: { value: 1 } },
  vertexShader: QUAD_VERT,
  fragmentShader: /* glsl */`
    in vec2 vUv; uniform sampler2D tPrev, tNew; uniform float uW;
    void main(){
      vec4 n = texture(tNew, vUv);
      gl_FragColor = uW >= 0.999 ? n : mix(texture(tPrev, vUv), n, uW);
    }`,
});

const T = meta.tone;
const toneMat = new THREE.ShaderMaterial({
  depthTest: false, depthWrite: false, blending: THREE.NoBlending,
  uniforms: { tAccum: { value: null }, uExposure: { value: EXPOSURE || (TONE === 'three' ? 1.0 : meta.metalExposure) }, uMode: { value: TONE === 'three' ? 1 : 0 } },
  vertexShader: QUAD_VERT,
  // The operator measured off the server: ACES input matrix -> Lottes generic
  // filmic (contrast / shoulder / mid / hdrMax as read from VTK) -> ACES output
  // matrix -> clamp -> gamma 1/2.2. RMS against the server's own output: 0.0011.
  fragmentShader: /* glsl */`
    in vec2 vUv; uniform sampler2D tAccum; uniform float uExposure; uniform float uMode;
    const float A = ${f6(T.contrast)}, D = ${f6(T.shoulder)}, MI = ${f6(T.midIn)},
                MO = ${f6(T.midOut)}, HM = ${f6(T.hdrMax)};
    const mat3 ACES_IN  = transpose(mat3(0.59719, 0.35458, 0.04823,
                                         0.07600, 0.90834, 0.01566,
                                         0.02840, 0.13383, 0.83777));
    const mat3 ACES_OUT = transpose(mat3( 1.60475, -0.53108, -0.07367,
                                         -0.10208,  1.10813, -0.00605,
                                         -0.00327, -0.07276,  1.07602));
    vec3 lottes(vec3 x) {
      x = max(x, vec3(0.0));
      float b = (-pow(MI, A) + pow(HM, A) * MO) / ((pow(HM, A * D) - pow(MI, A * D)) * MO);
      float c = (pow(HM, A * D) * pow(MI, A) - pow(HM, A) * pow(MI, A * D) * MO)
                / ((pow(HM, A * D) - pow(MI, A * D)) * MO);
      return pow(x, vec3(A)) / (pow(x, vec3(A * D)) * b + c);
    }
    // three.js's own ACESFilmicToneMapping (toneMapping 4 in three r152), then the sRGB transfer.
    vec3 rrtOdt(vec3 v) {
      vec3 a = v * (v + 0.0245786) - 0.000090537;
      vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
      return a / b;
    }
    vec3 srgbOetf(vec3 x) {
      return mix(pow(x, vec3(0.41666)) * 1.055 - 0.055, x * 12.92, vec3(lessThanEqual(x, vec3(0.0031308))));
    }
    void main(){
      vec4 t = texture(tAccum, vUv);
      vec3 c = t.rgb / max(t.a, 1e-4);        // straight colour before the curve
      if (uMode > 0.5) {
        c = ACES_OUT * rrtOdt(ACES_IN * (c * uExposure / 0.6));
        c = srgbOetf(clamp(c, 0.0, 1.0));
      } else {
        c = ACES_OUT * lottes(ACES_IN * (c * uExposure));
        c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
      }
      gl_FragColor = vec4(c * t.a, t.a);      // canvas is premultiplied
    }`,
});

const accumScene = new THREE.Scene(); accumScene.add(new THREE.Mesh(quadGeo, accumMat));
const toneScene = new THREE.Scene();  toneScene.add(new THREE.Mesh(quadGeo, toneMat));

// ---------------------------------------------------------------- loop
function halton(i, b) { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; }
const JITTER = Array.from({ length: 64 }, (_, i) => [halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5]);

let n = 0, prog = true, captured = false;
function restart() { n = 0; }

function resize() {
  const cw = FIXED || stage.clientWidth, ch = FIXED || stage.clientHeight;
  W = Math.max(2, Math.floor(cw * DPR)); H = Math.max(2, Math.floor(ch * DPR));
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  camera.aspect = W / H; camera.updateProjectionMatrix();
  makeTargets(); restart();
}
window.addEventListener('resize', resize);

function renderOne(cam = camera) {
  const j = JITTER[n % JITTER.length];
  cam.setViewOffset(W, H, n === 0 && !prog ? 0 : j[0], n === 0 && !prog ? 0 : j[1], W, H);
  renderer.setRenderTarget(frameRT);
  renderer.setClearColor(0x000000, 0);
  renderer.clear();
  renderer.render(scene, cam);

  accumMat.uniforms.tPrev.value = ping.texture;
  accumMat.uniforms.tNew.value = frameRT.texture;
  accumMat.uniforms.uW.value = 1 / (n + 1);
  renderer.setRenderTarget(pong);
  renderer.render(accumScene, quadCam);
  [ping, pong] = [pong, ping];
  n++;

  toneMat.uniforms.tAccum.value = ping.texture;
  renderer.setRenderTarget(null);
  renderer.render(toneScene, quadCam);
}

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  const limit = prog ? MAX_FRAMES : 1;
  if (n < limit) renderOne();
  document.getElementById('stat').textContent =
    `${W}×${H}  ·  kare ${Math.min(n, limit)}/${limit}  ·  ${(glbBuf.byteLength / 1e6).toFixed(2)} MB GLB`;
  window.__state = { n, ready: true };
  if (AUTO && !captured && n >= MAX_FRAMES) {
    captured = true;
    renderer.domElement.toBlob(async blob => {
      await fetch('/save?name=' + encodeURIComponent(SAVE), { method: 'POST', body: blob });
      document.title = 'saved'; window.__done = true;
    }, 'image/png');
  }
}

// ---------------------------------------------------------------- UI
function chips(id, items, current, onPick) {
  const host = document.getElementById(id);
  for (const [key, v] of Object.entries(items)) {
    const b = document.createElement('button');
    b.className = 'chip'; b.type = 'button';
    b.innerHTML = `<i style="background:${v.color}"></i>${v.label}`;
    b.setAttribute('aria-pressed', String(key === current));
    b.onclick = () => { [...host.children].forEach(c => c.setAttribute('aria-pressed', 'false')); b.setAttribute('aria-pressed', 'true'); onPick(key); };
    host.appendChild(b);
  }
}
if (HOST) document.getElementById('ui').style.display = 'none';   // the app has its own controls (CSS display:flex would beat [hidden])
if (!HOST && (!AUTO || Q.get('ui') === '1')) {
  document.getElementById('ui').hidden = false;
  document.getElementById('sub').textContent =
    [info.label, info.category, cut.planes ? `${stoneMeshes.length} taş` : 'taşsız'].filter(Boolean).join(' · ');
  chips('metals', meta.metals, metalKey, applyMetal);
  chips('stones', meta.stones, stoneKey, applyStone);
  document.getElementById('prog').onchange = e => { prog = e.target.checked; restart(); };
}

applyMetal(metalKey);
for (const m of stoneMeshes) m.material = m.userData.gemMat;
applyStone(stoneKey);
if (gltf.scene.children.length) gltf.scene.children.forEach(o => { if (o.userData.role === 'metal') o.material = metalMat; });
resize();
if (AUTO) {
  const ex = n => !!renderer.extensions.get(n);
  fetch('/log', { method: 'POST', body: 'info: ' + JSON.stringify({
    webgl2: renderer.capabilities.isWebGL2, floatRT: ex('EXT_color_buffer_float'),
    halfRT: ex('EXT_color_buffer_half_float'), W, H, cam: camera.position.toArray().map(x => +x.toFixed(2)),
    distance: +distance.toFixed(2), metalMeshes: gltf.scene.children.filter(o => o.userData.role === 'metal').length,
    stones: stoneMeshes.length, planes: NPLANE, bounces }) }).catch(() => {});
}
// ---------------------------------------------------------------- host API
// The catalogue app drives the viewer through this: materials, the two catalogue views, the camera
// the user orbited to, and a synchronous capture (no requestAnimationFrame, so it also works in a
// window the user cannot see).
function goToView(view) {
  camera.up.set(...view.up);
  camera.position.copy(new THREE.Vector3(...view.dir).normalize().multiplyScalar(camDist));
  camera.lookAt(0, 0, 0);
  makeControls();            // orbit around THIS view's up axis
  restart();
}

function flatten(canvas, bg) {
  const c = document.createElement('canvas');
  c.width = canvas.width; c.height = canvas.height;
  const g = c.getContext('2d');
  if (bg) { g.fillStyle = '#' + bg; g.fillRect(0, 0, c.width, c.height); }
  g.drawImage(canvas, 0, 0);
  return c.toDataURL('image/png');
}

function capture(spec) {
  const size = Math.max(64, Math.min(4096, spec.size | 0 || 1400));
  const frames = Math.max(1, spec.frames | 0 || MAX_FRAMES);
  const [keepW, keepH, keepProg] = [W, H, prog];
  prog = true;
  renderer.setPixelRatio(1); renderer.setSize(size, size, false);
  W = H = size; makeTargets();
  const cam = new THREE.PerspectiveCamera(fov, 1, camDist * 0.3, camDist * 3);
  const out = {};
  try {
    for (const [name, v] of Object.entries(spec.views)) {
      cam.up.set(...v.up);
      cam.position.copy(new THREE.Vector3(...v.dir).normalize().multiplyScalar(camDist));
      cam.lookAt(0, 0, 0); cam.updateMatrixWorld();
      n = 0;
      for (let i = 0; i < frames; i++) renderOne(cam);
      out[name] = flatten(renderer.domElement, spec.bg === undefined ? (BG || null) : spec.bg);
    }
  } finally {
    prog = keepProg; resize();
  }
  return out;
}

window.catalogViewer = {
  ready: true,
  metal: () => metalKey, stone: () => stoneKey,
  stones: stoneMeshes.length,
  setMetal: applyMetal, setStone: applyStone,
  views: () => info.pose.views || {},
  goToView: name => { const v = (info.pose.views || {})[name]; if (v) goToView(v); },
  setCamera: v => goToView(v),                       // {dir, up} in the canonical frame
  cameraState: () => ({
    dir: camera.position.clone().normalize().toArray(),
    up: new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).toArray(),
  }),
  capture,
};
tick();
