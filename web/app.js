import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const NUM_FIELDS = ['width', 'width_smoothing', 'near_ground', 'shoulder', 'corner_smoothing', 'elev_smoothing', 'camber', 'start_offset',
  'wall_offset', 'grid', 'pits', 'margin', 'terrain_spacing'];
const SEL_FIELDS = ['tyres', 'markings', 'sign_style', 'walls', 'dem_zoom', 'ground', 'ground_res', 'tree_density', 'grass_density', 'width_mode'];
const CHK_FIELDS = ['ml_ground', 'power_lines', 'road_signs', 'street_lights', 'terrain', 'buildings', 'buildings_collide', 'side_roads', 'overture_buildings', 'trees', 'overture_landcover',
  'photo_trees', 'photo_buildings', 'photo_paved', 'lidar', 'grass', 'wild_grass', 'grass_fx', 'csp_trees',
  'crosswalks', 'guardrails', 'csp_lights', 'csp_rain', 'hrdem'];

const state = {
  waypoints: [],        // [[lat,lng],...]
  modes: [],            // per segment, segment i = wp i -> wp i+1 (last = closing segment)
  route: null,          // {coords, segments, length_m}
  forceInclude: new Set(),
  widthOverrides: [],   // [{lat, lon, width, span}]
  forceExclude: new Set(),
  history: [],
  report: null,
  slug: null,
  mode: 'route',        // 'route' (waypoints) | 'area' (every street in a rectangle, free roam)
  area: null,           // [south, west, north, east] (the polygon's bounding box)
  areaPoly: null,       // [[lat, lng], ...] when the area is a polygon
  drawingPoly: false,   // clicking the corners of a polygon
  spawn: null,          // [lat, lng] where the cars start (area mode)
  drawing: false,       // drawing the area rectangle
  pickSpawn: false,     // next map click sets the start spot
};
const AREA_MAX_SIDE = 6000;   // m: the area's bounding box
const AREA_MAX_KM2 = 10;

async function api(path, body, method) {
  const opt = body !== undefined ? { method: method || 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

function status(msg) { $('status').textContent = msg ? tidyMsg(msg) : ''; }

// Long technical errors ("HTTPSConnectionPool(host=…): Max retries exceeded …") -> "(no connection)"; the full
// text stays in the tooltip.
function tidyMsg(t) {
  t = String(t);
  let out = '', i = 0;
  while (i < t.length) {
    if (t[i] === '(') {
      let d = 0, j = i;
      for (; j < t.length; j++) { if (t[j] === '(') d++; else if (t[j] === ')') { d--; if (!d) break; } }
      const inner = t.slice(i, j + 1);
      if (/ConnectionPool|Max retries|ProxyError|Tunnel connection|Errno|timed out/i.test(inner)) { out += '(no connection)'; i = j + 1; continue; }
    }
    out += t[i]; i++;
  }
  return out.replace(/\s+/g, ' ').trim();
}
// the same message for several route segments -> one line ("Segments 1, 2, 3: …")
function mergeWarnings(list) {
  const out = [], bySeg = new Map();
  (list || []).forEach((raw) => {
    const w = tidyMsg(raw);
    const m = w.match(/^Segment (\d+): (.*)$/);
    if (!m) { out.push({ text: w, raw: [raw] }); return; }
    let g = bySeg.get(m[2]);
    if (!g) { g = { msg: m[2], segs: [], raw: [] }; bySeg.set(m[2], g); out.push(g); }
    g.segs.push(m[1]); g.raw.push(raw);
  });
  out.forEach((o) => { if (o.segs) o.text = `Segment${o.segs.length > 1 ? 's' : ''} ${o.segs.join(', ')}: ${o.msg}`; });
  return out;
}
function renderWarnings(el, list, keep = 2) {
  const ws = mergeWarnings(list);
  el.innerHTML = ws.map((w, i) => `<div class="warn${i >= keep ? ' hidden more' : ''}" title="${esc(w.raw.join('\n'))}">⚠ ${esc(w.text)}</div>`).join('')
    + (ws.length > keep ? `<button class="warn-more">${ws.length - keep} more warning${ws.length - keep > 1 ? 's' : ''}…</button>` : '');
  const b = el.querySelector('.warn-more');
  if (b) b.onclick = () => { el.querySelectorAll('.warn.more').forEach((x) => x.classList.remove('hidden')); b.remove(); };
}

// ------------------------------------------------------------------ map
const map = L.map('map', { zoomControl: true, maxZoom: 21 }).setView([45.5017, -73.5673], 14);
const osmLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 21, maxNativeZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(map);
const satLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 21, maxNativeZoom: 19, attribution: 'Imagery &copy; Esri' });
const routeLayer = L.layerGroup().addTo(map);
const markerLayer = L.layerGroup().addTo(map);
const builtLayer = L.layerGroup().addTo(map);
const previewLayer = L.layerGroup().addTo(map);
const areaLayer = L.layerGroup().addTo(map);
let bldLayer = null;
const layersCtl = L.control.layers({ 'Map': osmLayer, 'Satellite (Esri)': satLayer }, { 'Generated track': builtLayer, 'Road preview': previewLayer },
  { position: 'topright', collapsed: false }).addTo(map);

// ---- aerial photo layers (served by the app from the imagery sources, cached on disk)
const photoLayers = [];
const PREF_KEY = 'maptotrack.photoLayer';
function pref(k, v) {
  try {
    const p = JSON.parse(localStorage.getItem(PREF_KEY) || '{}');
    if (v === undefined) return p[k];
    p[k] = v; localStorage.setItem(PREF_KEY, JSON.stringify(p));
  } catch (e) { /* storage unavailable */ }
  return undefined;
}
const OpacityCtl = L.Control.extend({
  options: { position: 'bottomleft' },
  onAdd() {
    const div = L.DomUtil.create('div', 'leaflet-bar opacity-ctl hidden');
    div.innerHTML = '<label title="See the map through the photo">Photo <input type="range" min="10" max="100" step="5" value="100"></label>';
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    const inp = div.querySelector('input');
    inp.value = pref('opacity') || 100;
    inp.oninput = () => { photoLayers.forEach((l) => l.setOpacity(inp.value / 100)); pref('opacity', +inp.value); };
    this._div = div;
    return div;
  },
  refresh() { this._div.classList.toggle('hidden', !photoLayers.some((l) => map.hasLayer(l))); },
});
const opacityCtl = new OpacityCtl().addTo(map);
function addPhotoLayers(sources) {
  (sources || []).filter((s) => s.type === 'wms').forEach((s) => {
    // 512 px tiles one zoom level down: same detail, 4x fewer requests to the WMS
    const opt = { maxZoom: 21, maxNativeZoom: 20, tileSize: 512, zoomOffset: -1, minZoom: 1,
      opacity: (pref('opacity') || 100) / 100, zIndex: 5, attribution: esc(s.attribution || '') };
    if (s.coverage) opt.bounds = L.latLngBounds([s.coverage[0], s.coverage[1]], [s.coverage[2], s.coverage[3]]);
    const lyr = L.tileLayer(`/api/tiles/${s.id}/{z}/{x}/{y}.jpg?size=512`, opt);
    lyr._srcId = s.id;
    photoLayers.push(lyr);
    layersCtl.addOverlay(lyr, 'Aerial photo: ' + esc(s.name.replace(/ aerial photos/i, '').replace(/^Québec – /, '')));
    if (pref('on') === s.id) lyr.addTo(map);
  });
  autoPhoto();
  opacityCtl.refresh();
}
// first time the map shows an area with a photo, switch it on (until the user chooses otherwise)
function autoPhoto() {
  if (pref('on') !== undefined || photoLayers.some((l) => map.hasLayer(l))) return;
  const c = map.getCenter();
  const l = photoLayers.find((x) => x.options.bounds && x.options.bounds.contains(c));
  if (l) l.addTo(map);
}
map.on('moveend', autoPhoto);
map.on('overlayadd overlayremove', (e) => {
  if (photoLayers.includes(e.layer)) {
    pref('on', e.type === 'overlayadd' ? e.layer._srcId : null);
    opacityCtl.refresh();
  }
});

function snapshot() {
  state.history.push(JSON.stringify({ w: state.waypoints, m: state.modes }));
  if (state.history.length > 100) state.history.shift();
}

function defaultMode() { return $('defaultMode').value; }

function segCount() {
  const n = state.waypoints.length;
  if (n < 2) return 0;
  return $('loop').checked && n >= 3 ? n : n - 1;
}

function normalizeModes() {
  const need = Math.max(state.waypoints.length, 1);
  while (state.modes.length < need) state.modes.push(defaultMode());
  state.modes.length = need;
}

function addWaypoint(latlng, index) {
  snapshot();
  const p = [latlng.lat, latlng.lng];
  if (index === undefined) {
    state.waypoints.push(p);
    // the segment arriving at the new point uses the default mode
    state.modes.splice(state.waypoints.length - 2, 0, defaultMode());
  } else {
    state.waypoints.splice(index, 0, p);
    state.modes.splice(index, 0, state.modes[index - 1] || defaultMode());
  }
  normalizeModes();
  changed();
}

function removeWaypoint(i) {
  snapshot();
  state.waypoints.splice(i, 1);
  state.modes.splice(Math.max(i - 1, 0), 1);
  normalizeModes();
  changed();
}

function makeIcon(i) {
  return L.divIcon({ className: '', html: `<div class="wp-icon ${i === 0 ? 'first' : ''}">${i + 1}</div>`, iconSize: [24, 24], iconAnchor: [12, 12] });
}

function drawMarkers() {
  markerLayer.clearLayers();
  state.waypoints.forEach((p, i) => {
    const m = L.marker(p, { icon: makeIcon(i), draggable: true, autoPan: true });
    m.on('dragstart', snapshot);
    m.on('dragend', () => { const ll = m.getLatLng(); state.waypoints[i] = [ll.lat, ll.lng]; changed(); });
    m.on('contextmenu', () => removeWaypoint(i));
    m.addTo(markerLayer);
  });
}

// Segments as drawn: the routed geometry when available, else straight lines between waypoints.
function segmentLines() {
  if (state.route && state.route.segments && state.route.segments.length === segCount()) return state.route.segments;
  const w = state.waypoints, out = [];
  for (let i = 0; i < segCount(); i++) out.push([w[i], w[(i + 1) % w.length]]);
  return out;
}

// Index of the segment nearest to a map position (in screen pixels).
function nearestSegment(latlng) {
  const p = map.latLngToLayerPoint(latlng);
  let best = -1, bestD = Infinity;
  segmentLines().forEach((seg, i) => {
    for (let k = 0; k + 1 < seg.length; k++) {
      const d = L.LineUtil.pointToSegmentDistance(p, map.latLngToLayerPoint(L.latLng(seg[k])), map.latLngToLayerPoint(L.latLng(seg[k + 1])));
      if (d < bestD) { bestD = d; best = i; }
    }
  });
  return { index: best, px: bestD };
}

// Insert a waypoint into the segment closest to latlng (keeps the order of the track).
function insertWaypoint(latlng) {
  const { index } = nearestSegment(latlng);
  if (index < 0) { addWaypoint(latlng); return; }
  addWaypoint(latlng, index + 1);
  status(`Point ${index + 2} added`);
}

// Point half-way along a polyline.
function midAlong(seg) {
  const pts = seg.map((q) => L.latLng(q));
  let total = 0;
  for (let k = 0; k + 1 < pts.length; k++) total += pts[k].distanceTo(pts[k + 1]);
  let acc = 0;
  for (let k = 0; k + 1 < pts.length; k++) {
    const d = pts[k].distanceTo(pts[k + 1]);
    if (acc + d >= total / 2 && d > 0) {
      const t = (total / 2 - acc) / d;
      return L.latLng(pts[k].lat + (pts[k + 1].lat - pts[k].lat) * t, pts[k].lng + (pts[k + 1].lng - pts[k].lng) * t);
    }
    acc += d;
  }
  return pts[0];
}

const midIcon = L.divIcon({ className: '', html: '<div class="mid-handle" title="Click or drag to add a point here">+</div>', iconSize: [16, 16], iconAnchor: [8, 8] });
const handleLayer = L.layerGroup().addTo(map);

function drawHandles() {
  handleLayer.clearLayers();
  segmentLines().forEach((seg, i) => {
    const a = map.latLngToLayerPoint(L.latLng(seg[0])), b = map.latLngToLayerPoint(L.latLng(seg[seg.length - 1]));
    if (a.distanceTo(b) < 36) return;                     // too short on screen: zoom in to see its handle
    const mid = midAlong(seg);
    const h = L.marker(mid, { icon: midIcon, draggable: true, zIndexOffset: -100, keyboard: false });
    h.on('click', (e) => { L.DomEvent.stopPropagation(e); addWaypoint(mid, i + 1); });
    h.on('dragend', () => addWaypoint(h.getLatLng(), i + 1));
    h.addTo(handleLayer);
  });
}
map.on('zoomend', drawHandles);

function drawRoute() {
  routeLayer.clearLayers();
  segmentLines().forEach((seg, i) => {
    const mode = state.modes[i] || 'road';
    const line = L.polyline(seg, { color: mode === 'road' ? '#2f80ed' : '#f2994a', weight: 6, opacity: state.route ? 0.85 : 0.4,
      dashArray: mode === 'free' ? '8 8' : null });
    line.on('click', (e) => { L.DomEvent.stopPropagation(e); addWaypoint(e.latlng, i + 1); });
    line.on('contextmenu', (e) => {
      L.DomEvent.stopPropagation(e); L.DomEvent.preventDefault(e);
      if (state.road && state.road.built_line) widthPopup(e.latlng);
    });
    line.bindTooltip(state.road ? 'Click: add a point here · Right-click: road width' : 'Click to add a point here',
      { sticky: true, direction: 'top', opacity: 0.8 });
    line.addTo(routeLayer);
  });
  drawHandles();
}

function renderList() {
  const ol = $('wpList');
  ol.innerHTML = '';
  const n = state.waypoints.length;
  const segs = segCount();
  state.waypoints.forEach((p, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="num ${i === 0 ? 'first' : ''}">${i + 1}</span><span class="coord">${p[0].toFixed(5)}, ${p[1].toFixed(5)}</span>`;
    if (i < segs) {
      const b = document.createElement('button');
      const mode = state.modes[i] || 'road';
      const to = i + 1 < n ? i + 2 : 1;
      b.className = `seg ${mode}`;
      b.textContent = `→${to} ${mode === 'road' ? 'roads' : 'free'}`;
      b.title = 'Toggle: follow roads / free line';
      b.onclick = () => { snapshot(); state.modes[i] = mode === 'road' ? 'free' : 'road'; changed(); };
      li.appendChild(b);
    }
    if (i < segs) {
      const ins = document.createElement('button');
      const to = i + 1 < n ? i + 2 : 1;
      ins.className = 'ins'; ins.textContent = '+'; ins.title = `Insert a point between ${i + 1} and ${to}`;
      ins.onclick = () => addWaypoint(midAlong(segmentLines()[i]), i + 1);
      li.appendChild(ins);
    }
    const del = document.createElement('button');
    del.className = 'del'; del.textContent = '✕'; del.title = 'Remove';
    del.onclick = () => removeWaypoint(i);
    li.appendChild(del);
    ol.appendChild(li);
  });
}

let routeTimer = null;
function changed() {
  normalizeModes();
  drawMarkers();
  drawRoute();
  renderList();
  saveDraft();
  clearTimeout(routeTimer);
  routeTimer = setTimeout(updateRoute, 250);
}

async function updateRoute() {
  if (state.mode === 'area') return;
  if (state.waypoints.length < 2) {
    state.route = null; drawRoute(); $('routeInfo').textContent = ''; return;
  }
  status('Routing…');
  try {
    const r = await api('/api/route', { waypoints: state.waypoints, modes: state.modes.slice(0, segCount()),
      loop: $('loop').checked, ignore_oneway: $('ignoreOneway').checked });
    state.route = r;
    drawRoute();
    let txt = `Route length: ${(r.length_m / 1000).toFixed(2)} km · ${state.waypoints.length} waypoints`;
    if (r.warnings.length) txt += ' · ⚠ ' + mergeWarnings(r.warnings).map((w) => w.text).join(' ');
    $('routeInfo').textContent = txt;
    status('');
    schedulePreview();
  } catch (e) { status('Routing failed: ' + e.message); }
}

map.on('click', (e) => {
  if (state.mode === 'area') {
    if (ignoreClick) { ignoreClick = false; return; }
    if (state.drawingPoly) { polyClick(e.latlng); return; }
    if (state.pickSpawn) {
      state.spawn = [e.latlng.lat, e.latlng.lng]; state.pickSpawn = false;
      $('spawnBtn').classList.remove('on'); drawArea(); saveDraft(); status('Start spot set'); schedulePreview();
    }
    return;
  }
  // close to the track (within 14 px): insert into that part of the track; elsewhere: add at the end
  if (state.waypoints.length >= 2 && nearestSegment(e.latlng).px <= 14) insertWaypoint(e.latlng);
  else addWaypoint(e.latlng);
});

// ------------------------------------------------------------------ area mode (free roam: every street in a rectangle / polygon)
let ignoreClick = false, drawStart = null, drawRect = null;
let polyDraft = [], polyLine = null, polyRubber = null;
// size of an outline [[lat, lng], ...]: area (m²) and its bounding box (m)
function outlineSize(pts) {
  const lat0 = pts.reduce((a, p) => a + p[0], 0) / pts.length, k = Math.cos(lat0 * Math.PI / 180);
  const xy = pts.map((p) => [p[1] * 111320 * k, p[0] * 111320]);
  let A = 0;
  for (let i = 0; i < xy.length; i++) { const [x1, y1] = xy[i], [x2, y2] = xy[(i + 1) % xy.length]; A += x1 * y2 - x2 * y1; }
  const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
  return { area: Math.abs(A) / 2, w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
}
function rectPts(a) { return [[a[0], a[1]], [a[0], a[3]], [a[2], a[3]], [a[2], a[1]]]; }
function bboxOf(pts) {
  const la = pts.map((p) => p[0]), lo = pts.map((p) => p[1]);
  return [Math.min(...la), Math.min(...lo), Math.max(...la), Math.max(...lo)];
}
function tooBig(sz) { return sz.area > AREA_MAX_KM2 * 1e6 * 1.005 || Math.max(sz.w, sz.h) > AREA_MAX_SIDE * 1.005; }
function sizeText(sz) { return `${(sz.area / 1e6).toFixed(2)} km² (${(sz.w / 1000).toFixed(2)} x ${(sz.h / 1000).toFixed(2)} km)`; }
// rectangle: keep the anchor corner; at most AREA_MAX_SIDE a side and AREA_MAX_KM2
function clampArea(anchor, other) {
  const k = Math.cos(anchor.lat * Math.PI / 180);
  let dy = (other.lat - anchor.lat) * 111320, dx = (other.lng - anchor.lng) * 111320 * k;
  dy = Math.max(-AREA_MAX_SIDE, Math.min(AREA_MAX_SIDE, dy)); dx = Math.max(-AREA_MAX_SIDE, Math.min(AREA_MAX_SIDE, dx));
  const A = Math.abs(dx * dy), lim = AREA_MAX_KM2 * 1e6;
  if (A > lim) { const f = Math.sqrt(lim / A); dx *= f; dy *= f; }
  const lat = anchor.lat + dy / 111320, lng = anchor.lng + dx / (111320 * k);
  return [Math.min(anchor.lat, lat), Math.min(anchor.lng, lng), Math.max(anchor.lat, lat), Math.max(anchor.lng, lng)];
}
function areaOutline() { return state.areaPoly || (state.area ? rectPts(state.area) : null); }
function areaInfo() {
  const pts = areaOutline();
  if (!pts) { $('areaInfo').textContent = 'No area yet.'; return; }
  const sz = outlineSize(pts);
  let t = `${state.areaPoly ? `Polygon (${pts.length} corners)` : 'Rectangle'}: ${sizeText(sz)}`;
  if (sz.area > 4e6 * 1.01 || sz.w * sz.h > 4e6 * 1.01) t += ' · big area: coarser ground texture';
  t += state.spawn ? ' · start spot set' : ' · start: main street near the middle';
  $('areaInfo').textContent = t;
}
const handleIcon = L.divIcon({ className: '', html: '<div class="area-handle"></div>', iconSize: [14, 14], iconAnchor: [7, 7] });
const midAreaIcon = L.divIcon({ className: '', html: '<div class="area-mid" title="Drag to add a corner">+</div>', iconSize: [12, 12], iconAnchor: [6, 6] });
const spawnIcon = L.divIcon({ className: '', html: '<div class="spawn-icon" title="Start spot">S</div>', iconSize: [22, 22], iconAnchor: [11, 11] });
function setPoly(pts) {
  const sz = outlineSize(pts);
  if (tooBig(sz)) { status(`Too big: ${sizeText(sz)} - at most ${AREA_MAX_KM2} km² and ${AREA_MAX_SIDE / 1000} km across`); drawArea(); return false; }
  state.areaPoly = pts; state.area = bboxOf(pts); drawArea(); saveDraft(); schedulePreview(); return true;
}
function drawArea() {
  areaLayer.clearLayers();
  if (state.mode !== 'area') return;
  areaInfo();
  if (state.areaPoly) {
    const pts = state.areaPoly;
    const poly = L.polygon(pts, { color: '#e8483b', weight: 2, fillOpacity: 0.04, interactive: false }).addTo(areaLayer);
    pts.forEach((p, i) => {
      const h = L.marker(p, { icon: handleIcon, draggable: true, keyboard: false })
        .bindTooltip('Drag to move · right-click to remove').addTo(areaLayer);
      h.on('drag', () => { const q = pts.slice(); const ll = h.getLatLng(); q[i] = [ll.lat, ll.lng]; poly.setLatLngs(q);
        poly.setStyle({ color: tooBig(outlineSize(q)) ? '#ff2d2d' : '#e8483b', dashArray: tooBig(outlineSize(q)) ? '6 4' : null }); });
      h.on('dragend', () => { const q = pts.slice(); const ll = h.getLatLng(); q[i] = [ll.lat, ll.lng]; setPoly(q); });
      h.on('contextmenu', () => { if (pts.length > 3) setPoly(pts.filter((_, k) => k !== i)); });
    });
    pts.forEach((p, i) => {          // edge handles: drag to add a corner
      const q = pts[(i + 1) % pts.length];
      const m = L.marker([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], { icon: midAreaIcon, draggable: true, keyboard: false, zIndexOffset: -100 }).addTo(areaLayer);
      m.on('dragend', () => { const ll = m.getLatLng(); const r = pts.slice(); r.splice(i + 1, 0, [ll.lat, ll.lng]); setPoly(r); });
    });
  } else if (state.area) {
    const a = state.area;
    const rect = L.rectangle([[a[0], a[1]], [a[2], a[3]]], { color: '#e8483b', weight: 2, fillOpacity: 0.04, interactive: false }).addTo(areaLayer);
    // corner handles: drag to resize (the opposite corner stays)
    [[0, 1, 2, 3], [2, 3, 0, 1], [0, 3, 2, 1], [2, 1, 0, 3]].forEach(([i, j, oi, oj]) => {
      const h = L.marker([a[i], a[j]], { icon: handleIcon, draggable: true, keyboard: false }).addTo(areaLayer);
      const opp = L.latLng(a[oi], a[oj]);
      h.on('drag', () => { const b = clampArea(opp, h.getLatLng()); rect.setBounds([[b[0], b[1]], [b[2], b[3]]]); });
      h.on('dragend', () => { state.area = clampArea(opp, h.getLatLng()); drawArea(); saveDraft(); schedulePreview(); });
    });
  }
  if (state.spawn) {
    const m = L.marker(state.spawn, { icon: spawnIcon, draggable: true }).bindTooltip('Cars start on the street nearest to this point').addTo(areaLayer);
    m.on('dragend', () => { const ll = m.getLatLng(); state.spawn = [ll.lat, ll.lng]; saveDraft(); schedulePreview(); });
    m.on('contextmenu', () => { state.spawn = null; drawArea(); saveDraft(); schedulePreview(); });
  }
}
// ---- rectangle: drag on the map
function setDrawing(on) {
  state.drawing = on;
  $('drawAreaBtn').classList.toggle('on', on);
  $('map').classList.toggle('drawing', on || state.drawingPoly);
  if (on) { setPolyDrawing(false); map.dragging.disable(); status(`Drag a rectangle on the map (at most ${AREA_MAX_KM2} km²)`); } else map.dragging.enable();
}
map.on('mousedown', (e) => {
  if (!state.drawing) return;
  drawStart = e.latlng;
  drawRect = L.rectangle([e.latlng, e.latlng], { color: '#e8483b', weight: 2, dashArray: '6 4', fillOpacity: 0.06, interactive: false }).addTo(map);
});
map.on('mousemove', (e) => {
  if (state.drawingPoly && polyDraft.length) {
    const pts = polyDraft.concat([[e.latlng.lat, e.latlng.lng]]);
    if (polyRubber) polyRubber.setLatLngs(pts.concat([polyDraft[0]]));
    if (pts.length >= 3) { const sz = outlineSize(pts); status((tooBig(sz) ? 'Too big: ' : '') + sizeText(sz) + ' · click the first corner or double-click to finish'); }
    return;
  }
  if (!state.drawing || !drawStart) return;
  const b = clampArea(drawStart, e.latlng);
  drawRect.setBounds([[b[0], b[1]], [b[2], b[3]]]);
  status(sizeText(outlineSize(rectPts(b))));
});
map.on('mouseup', (e) => {
  if (!state.drawing || !drawStart) return;
  const b = clampArea(drawStart, e.latlng);
  map.removeLayer(drawRect); drawRect = null; drawStart = null;
  setDrawing(false);
  ignoreClick = true; setTimeout(() => { ignoreClick = false; }, 300);
  const sz = outlineSize(rectPts(b));
  if (sz.w < 50 || sz.h < 50) { status('Too small - drag a bigger rectangle'); return; }
  state.area = b; state.areaPoly = null; drawArea(); saveDraft(); status(''); schedulePreview();
});
// ---- polygon: click the corners, click the first one (or double-click) to finish
function setPolyDrawing(on) {
  state.drawingPoly = on;
  $('drawPolyBtn').classList.toggle('on', on);
  $('map').classList.toggle('drawing', on || state.drawing);
  polyDraft = [];
  [polyLine, polyRubber].forEach((l) => { if (l) map.removeLayer(l); });
  polyLine = polyRubber = null;
  if (on) {
    map.doubleClickZoom.disable();
    status(`Click the corners of the area (at most ${AREA_MAX_KM2} km²); click the first corner or double-click to finish · Esc cancels`);
  } else map.doubleClickZoom.enable();
}
function polyClick(ll) {
  const p = [ll.lat, ll.lng];
  if (polyDraft.length >= 3) {
    const first = map.latLngToContainerPoint(L.latLng(polyDraft[0]));
    if (first.distanceTo(map.latLngToContainerPoint(ll)) < 12) { finishPoly(); return; }
  }
  polyDraft.push(p);
  if (!polyLine) {
    polyLine = L.polyline(polyDraft, { color: '#e8483b', weight: 2, interactive: false }).addTo(map);
    polyRubber = L.polyline([], { color: '#e8483b', weight: 1.5, dashArray: '5 5', interactive: false }).addTo(map);
  } else polyLine.setLatLngs(polyDraft);
}
function finishPoly() {
  // a double-click also sends two clicks: drop the corners they added on the same spot
  const pts = polyDraft.filter((p, i) => i === 0 || L.latLng(p).distanceTo(L.latLng(polyDraft[i - 1])) > 1);
  if (pts.length < 3) { status('An area needs at least 3 corners'); return; }
  const sz = outlineSize(pts);
  if (tooBig(sz)) { status(`Too big: ${sizeText(sz)} - at most ${AREA_MAX_KM2} km² and ${AREA_MAX_SIDE / 1000} km across. Keep clicking to fix the last corner, or Esc.`); polyDraft = pts.slice(0, -1); polyLine.setLatLngs(polyDraft); return; }
  if (sz.area < 2500) { status('Too small'); return; }
  setPolyDrawing(false);
  setPoly(pts); status('');
}
map.on('dblclick', (e) => { if (state.drawingPoly) { L.DomEvent.stop(e); finishPoly(); } });
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (state.drawingPoly) { setPolyDrawing(false); status('Polygon cancelled'); }
  if (state.drawing) { setDrawing(false); status(''); }
});
function setMode(m) {
  state.mode = m === 'area' ? 'area' : 'route';
  const area = state.mode === 'area';
  document.body.classList.toggle('area-mode', area);
  document.querySelectorAll('#modeToggle button').forEach((b) => b.classList.toggle('active', b.dataset.mode === state.mode));
  [routeLayer, markerLayer, handleLayer].forEach((l) => { if (area) map.removeLayer(l); else if (!map.hasLayer(l)) l.addTo(map); });
  if (!area) { setDrawing(false); setPolyDrawing(false); state.pickSpawn = false; }
  $('generateBtn').textContent = area ? 'Generate area' : 'Generate track';
  drawArea();
}
document.querySelectorAll('#modeToggle button').forEach((b) => {
  b.onclick = () => { if (b.dataset.mode === state.mode) return; setMode(b.dataset.mode); saveDraft();
    previewLayer.clearLayers(); $('pvResult').classList.add('hidden'); state.preview = null; state.road = null; };
});
$('drawAreaBtn').onclick = () => { state.pickSpawn = false; $('spawnBtn').classList.remove('on'); setDrawing(!state.drawing); };
$('drawPolyBtn').onclick = () => { state.pickSpawn = false; $('spawnBtn').classList.remove('on'); setDrawing(false); setPolyDrawing(!state.drawingPoly); };
$('spawnBtn').onclick = () => {
  setDrawing(false); setPolyDrawing(false);
  state.pickSpawn = !state.pickSpawn; $('spawnBtn').classList.toggle('on', state.pickSpawn);
  if (state.pickSpawn) status('Click the street where the cars should start');
};
$('clearAreaBtn').onclick = () => { state.area = null; state.areaPoly = null; state.spawn = null; previewLayer.clearLayers(); drawArea(); saveDraft(); };
function ready() {
  return state.mode === 'area' ? !!state.area : !!(state.route && state.route.coords.length >= 2);
}
function notReady() { status(state.mode === 'area' ? 'Draw the area first' : 'Add at least two waypoints'); }

$('loop').onchange = changed;
$('ignoreOneway').onchange = () => { saveDraft(); updateRoute(); };
$('undoBtn').onclick = () => {
  const h = state.history.pop(); if (!h) return;
  const s = JSON.parse(h); state.waypoints = s.w; state.modes = s.m; changed();
};
$('clearBtn').onclick = () => { if (!state.waypoints.length) return; snapshot(); state.waypoints = []; state.modes = []; changed(); };

// ------------------------------------------------------------------ search
async function doSearch() {
  const q = $('search').value.trim(); if (!q) return;
  const m = q.match(/^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (m) { map.setView([+m[1], +m[2]], 16); return; }
  status('Searching…');
  try {
    const res = await api('/api/geocode?q=' + encodeURIComponent(q));
    const ul = $('searchResults'); ul.innerHTML = '';
    res.forEach((r) => {
      const li = document.createElement('li'); li.textContent = r.name;
      li.onclick = () => { map.setView([r.lat, r.lon], 15); ul.classList.add('hidden'); };
      ul.appendChild(li);
    });
    ul.classList.toggle('hidden', !res.length);
    status(res.length ? '' : 'No results');
  } catch (e) { status('Search failed: ' + e.message); }
}
$('searchBtn').onclick = doSearch;
$('search').onkeydown = (e) => { if (e.key === 'Enter') doSearch(); };

// ------------------------------------------------------------------ import
$('importBtn').onclick = () => $('importFile').click();
$('importFile').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const txt = await f.text();
  const doc = new DOMParser().parseFromString(txt, 'application/xml');
  let pts = [], line = [];
  if (f.name.toLowerCase().endsWith('.gpx') || doc.querySelector('gpx')) {
    line = [...doc.querySelectorAll('trkpt, rtept')].map((n) => [+n.getAttribute('lat'), +n.getAttribute('lon')]);
    pts = [...doc.querySelectorAll('wpt')].map((n) => [+n.getAttribute('lat'), +n.getAttribute('lon')]);
  } else {
    const parse = (s) => s.trim().split(/\s+/).map((c) => c.split(',')).filter((c) => c.length >= 2).map((c) => [+c[1], +c[0]]);
    doc.querySelectorAll('Placemark').forEach((pm) => {
      const ls = pm.querySelector('LineString coordinates');
      const pt = pm.querySelector('Point coordinates');
      if (ls) line = line.concat(parse(ls.textContent));
      else if (pt) pts = pts.concat(parse(pt.textContent));
    });
  }
  snapshot();
  if (line.length >= 2) {
    // a drawn line: keep its shape, thin it to <= 120 waypoints, free segments
    const step = Math.max(1, Math.ceil(line.length / 120));
    state.waypoints = line.filter((_, i) => i % step === 0);
    state.modes = state.waypoints.map(() => 'free');
  } else if (pts.length >= 2) {
    state.waypoints = pts; state.modes = pts.map(() => 'road');
  } else { status('No points found in file'); return; }
  e.target.value = '';
  fitWaypoints(); changed();
};

function parseGoogleUrl(url) {
  url = decodeURIComponent(url);
  const pts = [];
  const dir = url.match(/\/dir\/([^?@]*)/);
  if (dir) {
    const parts = dir[1].split('/').filter(Boolean);
    const coords = parts.map((p) => p.match(/^(-?\d+\.\d+),\s*(-?\d+\.\d+)$/));
    if (coords.length && coords.every(Boolean)) return coords.map((m) => [+m[1], +m[2]]);
  }
  const re = /!1d(-?\d+\.\d+)!2d(-?\d+\.\d+)/g; let m;
  while ((m = re.exec(url))) pts.push([+m[2], +m[1]]);
  if (pts.length) return pts;
  const at = url.match(/@(-?\d+\.\d+),(-?\d+\.\d+)/);
  return at ? [[+at[1], +at[2]]] : [];
}

$('gmapsBtn').onclick = () => {
  const pts = parseGoogleUrl($('gmapsUrl').value);
  if (!pts.length) { status('No coordinates found in that link (short links goo.gl/maps must be opened first and the full URL copied)'); return; }
  if (pts.length === 1) { map.setView(pts[0], 16); return; }
  snapshot(); state.waypoints = pts; state.modes = pts.map(() => 'road');
  fitWaypoints(); changed();
};

function fitWaypoints() {
  if (state.mode === 'area' && state.area) { const a = state.area; map.fitBounds([[a[0], a[1]], [a[2], a[3]]], { padding: [30, 30] }); return; }
  if (state.waypoints.length) map.fitBounds(L.latLngBounds(state.waypoints).pad(0.2));
}

// ------------------------------------------------------------------ params
function collectParams() {
  const p = { name: $('name').value || 'My Track', loop: $('loop').checked, reverse: $('reverse').checked,
    waypoints: state.waypoints, modes: state.modes, ignore_oneway: $('ignoreOneway').checked,
    coords: state.route ? state.route.coords : [],
    force_include: [...state.forceInclude], force_exclude: [...state.forceExclude],
    width_overrides: state.widthOverrides, mode: state.mode, area: state.area, area_poly: state.areaPoly, spawn: state.spawn };
  NUM_FIELDS.forEach((k) => { p[k] = parseFloat($(k).value); });
  SEL_FIELDS.forEach((k) => { p[k] = $(k).value; });
  CHK_FIELDS.forEach((k) => { p[k] = $(k).checked; });
  p.conifer_ratio = Math.min(Math.max(parseFloat($('conifer_pct').value) || 0, 0), 100) / 100;
  ['ground_res', 'tree_density', 'grass_density'].forEach((k) => { p[k] = parseFloat(p[k]); });
  const s = window.__settings || {};
  p.author = s.author || '';
  return p;
}

function applyParams(p) {
  if (p.name) $('name').value = p.name;
  ['loop', 'reverse'].forEach((k) => { if (k in p) $(k).checked = !!p[k]; });
  if ('ignore_oneway' in p) $('ignoreOneway').checked = !!p.ignore_oneway;
  NUM_FIELDS.forEach((k) => { if (p[k] !== undefined && p[k] !== null) $(k).value = p[k]; });
  SEL_FIELDS.forEach((k) => { if (p[k] !== undefined) $(k).value = String(p[k]); });
  CHK_FIELDS.forEach((k) => { if (k in p) $(k).checked = !!p[k]; });
  if (p.conifer_ratio !== undefined) $('conifer_pct').value = Math.round(p.conifer_ratio * 100);
  state.waypoints = p.waypoints || [];
  state.modes = p.modes || [];
  state.forceInclude = new Set(p.force_include || []);
  state.forceExclude = new Set(p.force_exclude || []);
  state.widthOverrides = p.width_overrides || [];
  state.area = p.area || null;
  state.areaPoly = p.area_poly || null;
  state.spawn = p.spawn || null;
  setMode(p.mode || 'route');
  renderOverrides();
}

function saveDraft() {
  try { localStorage.setItem('maptotrack.draft', JSON.stringify(collectParams())); } catch (e) { /* ignore */ }
}

// ------------------------------------------------------------------ jobs
async function waitJob(jid, onLog) {
  for (;;) {
    const j = await api('/api/jobs/' + jid);
    onLog && onLog(j.log);
    if (j.status === 'done') return j.result;
    if (j.status === 'error') throw new Error(j.error);
    await new Promise((r) => setTimeout(r, 600));
  }
}

function showLog(lines) {
  const el = $('log'); el.classList.remove('hidden');
  el.textContent = lines.join('\n'); el.scrollTop = el.scrollHeight;
}

// ------------------------------------------------------------------ scenery (buildings on map)
let sceneryData = null;
function bldStyle(f) {
  const id = f.properties.id;
  let c = '#9aa7b4';
  if (state.forceInclude.has(id)) c = '#27ae60';
  else if (state.forceExclude.has(id)) c = '#b05cc8';
  else if (f.properties.auto_excluded) c = '#e8483b';
  return { color: c, weight: 1, fillColor: c, fillOpacity: 0.45 };
}

function drawBuildings() {
  if (bldLayer) map.removeLayer(bldLayer);
  if (!sceneryData) return;
  bldLayer = L.geoJSON(sceneryData.buildings, {
    style: bldStyle,
    onEachFeature: (f, layer) => {
      layer.bindTooltip(esc(`${f.properties.name || f.properties.kind} · ${f.properties.height} m · ${f.properties.source === 'overture' ? 'Overture (AI footprint)' : 'OpenStreetMap'}`), { sticky: true });
      layer.on('click', (e) => {
        L.DomEvent.stopPropagation(e);
        const id = f.properties.id;
        if (f.properties.auto_excluded) {
          state.forceInclude.has(id) ? state.forceInclude.delete(id) : state.forceInclude.add(id);
        } else {
          state.forceExclude.has(id) ? state.forceExclude.delete(id) : state.forceExclude.add(id);
        }
        layer.setStyle(bldStyle(f));
        saveDraft();
      });
    },
  }).addTo(map);
}

$('sceneryBtn').onclick = async () => {
  if (!ready()) { notReady(); return; }
  $('sceneryBtn').disabled = true; status('Loading OpenStreetMap buildings…');
  try {
    const { job } = await api('/api/scenery', collectParams());
    sceneryData = await waitJob(job, (l) => status(l[l.length - 1] || 'Loading…'));
    drawBuildings();
    const c = sceneryData.counts;
    let info = `${c.buildings} buildings (${c.osm_buildings} OpenStreetMap + ${c.extra_buildings} Overture), ${c.roads} streets, ${c.trees} trees · ${c.auto_excluded} on the ${state.mode === 'area' ? 'streets' : 'track'} will be removed`;
    if (!c.buildings) info += ' · No building footprints exist for this area in the map data.';
    if (sceneryData.warnings && sceneryData.warnings.length) info += ' · ⚠ ' + mergeWarnings(sceneryData.warnings).map((w) => w.text).join(' ');
    $('sceneryInfo').textContent = info;
    $('bldLegend').classList.remove('hidden');
    status('');
  } catch (e) { status('Scenery failed: ' + e.message); }
  $('sceneryBtn').disabled = false;
};

// ------------------------------------------------------------------ generate
$('generateBtn').onclick = async () => {
  if (!ready()) { notReady(); return; }
  const btn = $('generateBtn'); btn.disabled = true; btn.textContent = 'Generating…';
  try {
    const { job } = await api('/api/generate', collectParams());
    const rep = await waitJob(job, showLog);
    showReport(rep);
    refreshProjects();
    loadGLB(rep.files.glb + '?t=' + Date.now());
  } catch (e) { showLog([...$('log').textContent.split('\n'), 'ERROR: ' + e.message]); }
  btn.disabled = false; btn.textContent = state.mode === 'area' ? 'Generate area' : 'Generate track';
};

function stat(v, k) { return `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`; }

function showReport(rep) {
  state.report = rep; state.slug = rep.slug; state.road = rep; state.preview = null;
  previewLayer.clearLayers();
  $('pvResult').classList.add('hidden'); $('pvState').textContent = '';
  if (!map.hasLayer(builtLayer)) builtLayer.addTo(map);
  renderOverrides();
  $('result').classList.remove('hidden');
  const e = rep.elevation;
  const w = rep.width || {};
  const ph = rep.photo || {};
  const ts = rep.tree_sources || {};
  if (rep.area) { showAreaReport(rep); return; }
  $('stats').innerHTML = [
    stat((rep.length_m / 1000).toFixed(2) + ' km', 'length'),
    w.used ? stat(w.median + ' m', `road width (photo, ${w.min}–${w.max})`) : '',
    stat(rep.min_radius_m + ' m', 'tightest radius'),
    stat(e.max_grade_pct + '%', 'max gradient'),
    stat(e.gain + ' m', 'climb'),
    stat(rep.buildings, 'buildings' + (rep.building_stats && (rep.building_stats.found_photo + rep.building_stats.found_lidar)
      ? ` (${rep.building_stats.found_photo + rep.building_stats.found_lidar} found)` : '')),
    stat((rep.trees || 0).toLocaleString(), 'trees & bushes' + (ts.hedges ? ` (${ts.hedges} in hedges)` : '')),
    rep.grass && (rep.grass.lawn || rep.grass.tall) ? stat(((rep.grass.lawn + rep.grass.tall) / 1000).toFixed(1) + 'k', 'grass clumps') : '',
    ph.paved_areas !== undefined ? stat(ph.paved_areas, 'driveways/lots') : '',
    stat((rep.total_verts / 1000).toFixed(0) + 'k', 'vertices'),
  ].join('');
  $('kn5Status').textContent = rep.kn5
    ? `${rep.slug}.kn5 written (${rep.kn5.meshes} meshes, ${rep.kn5.textures} textures) - ready to install. After editing in Blender, rebuild it here.`
    : 'No .kn5 from this generation. Build it from your Blender scene or an FBX file.';
  renderWarnings($('warnings'), rep.warnings);
  drawProfile(rep.profile);
  drawWidth(rep);
  builtLayer.clearLayers();
  const mk = rep.markings || {};
  drawTyres(rep.tyres, builtLayer);
  if (mk.lines && mk.lines.length) drawMarkings(mk.lines, builtLayer);
  else L.polyline(rep.built_line, { color: '#fff', weight: 2, dashArray: '4 4', opacity: 0.6 }).addTo(builtLayer);
  drawEdges(rep);
  roadHitLine(rep, builtLayer);
  const det = rep.detections || {};
  (det.paved || []).forEach((p) => L.polygon(p, { color: '#bdbdbd', weight: 1, fillOpacity: 0.25 }).bindTooltip('paved area (drivable)').addTo(builtLayer));
  (det.buildings || []).forEach((p) => L.polygon(p, { color: '#bb6bd9', weight: 2, fillOpacity: 0.2 }).bindTooltip('building found on the photo / LiDAR').addTo(builtLayer));
  rep.objects.forEach((o) => {
    const c = o.name.startsWith('AC_START') ? '#27ae60' : o.name.startsWith('AC_PIT') ? '#2f80ed' : '#f2c94c';
    L.circleMarker([o.lat, o.lon], { radius: 3, color: c, weight: 2 }).bindTooltip(o.name).addTo(builtLayer);
  });
}

// area mode: every street (asphalt, edges coloured by how clear they are on the photo), markings, start spots
function drawStreets(r, layer) {
  const pts = [], ws = [], info = [];
  r.streets.forEach((st) => {
    const [a, b] = st.edges;
    L.polygon(a.concat(b.slice().reverse()), { stroke: false, fillColor: '#1d2024', fillOpacity: 0.55, interactive: false }).addTo(layer);
    const col = st.measured ? (st.conf >= 0.3 ? '#27ae60' : '#f2994a') : '#f2c94c';
    [a, b].forEach((e) => L.polyline(e, { color: col, weight: 1.5, interactive: false }).addTo(layer));
    const tip = `${st.highway} · ${st.width} m wide (${st.measured ? 'measured on the photo' : 'typical width'}) · ${st.length} m · right-click: set the width`;
    const hit = L.polyline(st.centre, { color: '#000', weight: 16, opacity: 0.001 }).bindTooltip(tip, { sticky: true }).addTo(layer);
    hit.on('contextmenu', (ev) => { L.DomEvent.stopPropagation(ev); L.DomEvent.preventDefault(ev); widthPopup(ev.latlng); });
    st.centre.forEach((p) => { pts.push(p); ws.push(st.width); info.push(st.highway); });
  });
  r.built_line = pts; r.built_w = ws; r.built_s = null; r.built_info = info;
  const mk = r.markings || {};
  if (mk.lines && mk.lines.length) drawMarkings(mk.lines, layer);
  r.objects.forEach((o) => {
    const c = o.name.startsWith('AC_START') ? '#27ae60' : o.name.startsWith('AC_PIT') ? '#2f80ed' : '#f2c94c';
    L.circleMarker([o.lat, o.lon], { radius: 3, color: c, weight: 2 }).bindTooltip(o.name).addTo(layer);
  });
}
function areaStats(r) {
  const w = r.width || {};
  const mk = ((r.markings || {}).info || {}).streets || {};
  return [
    stat((r.length_m / 1000).toFixed(2) + ' km', `streets (${r.n_streets})`),
    stat(r.n_junctions, 'junctions'),
    stat((w.median || '?') + ' m', w.measured ? `typical width (${w.measured} measured)` : 'typical width'),
    stat(mk.photo || 0, 'streets with lines read on the photo'),
    stat(`${r.elevation.min}–${r.elevation.max} m`, 'elevation'),
  ];
}
function showAreaReport(rep) {
  const ph = rep.photo || {};
  const ts = rep.tree_sources || {};
  $('stats').innerHTML = areaStats(rep).concat([
    stat(rep.buildings, 'buildings' + (rep.building_stats && (rep.building_stats.found_photo + rep.building_stats.found_lidar)
      ? ` (${rep.building_stats.found_photo + rep.building_stats.found_lidar} found)` : '')),
    stat((rep.trees || 0).toLocaleString(), 'trees & bushes' + (ts.hedges ? ` (${ts.hedges} in hedges)` : '')),
    ph.paved_areas !== undefined ? stat(ph.paved_areas, 'driveways/lots') : '',
    stat((rep.total_verts / 1000).toFixed(0) + 'k', 'vertices'),
  ]).join('');
  $('kn5Status').textContent = rep.kn5
    ? `${rep.slug}.kn5 written (${rep.kn5.meshes} meshes, ${rep.kn5.textures} textures) - ready to install. After editing in Blender, rebuild it here.`
    : 'No .kn5 from this generation. Build it from your Blender scene or an FBX file.';
  renderWarnings($('warnings'), rep.warnings);
  builtLayer.clearLayers();
  drawStreets(rep, builtLayer);
  const det = rep.detections || {};
  (det.paved || []).forEach((p) => L.polygon(p, { color: '#bdbdbd', weight: 1, fillOpacity: 0.25 }).bindTooltip('paved area (drivable)').addTo(builtLayer));
  (det.buildings || []).forEach((p) => L.polygon(p, { color: '#bb6bd9', weight: 2, fillOpacity: 0.2 }).bindTooltip('building found on the photo / LiDAR').addTo(builtLayer));
}

// single-series line chart with crosshair tooltip (elevation, road width)
function lineChart(canvasId, tipId, xs, ys, opt) {
  const cv = $(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = 110;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const pad = { l: 38, r: 6, t: 6, b: 18 };
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const span = Math.max(ymax - ymin, opt.minSpan || 1);
  const y0 = opt.floor !== undefined ? Math.min(opt.floor, ymin) : ymin - span * 0.15;
  const y1 = ymax + span * 0.15;
  const xmax = xs[xs.length - 1];
  const X = (v) => pad.l + (v / xmax) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const draw = (hover) => {
    g.clearRect(0, 0, W, H);
    g.strokeStyle = '#2c3137'; g.lineWidth = 1; g.fillStyle = '#7c858f'; g.font = '10px system-ui';
    for (let i = 0; i <= 2; i++) {
      const v = y0 + (y1 - y0) * i / 2; const y = Y(v);
      g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
      g.fillText(v.toFixed(opt.decimals || 0) + ' m', 2, y + 3);
    }
    g.fillText('0', pad.l, H - 4);
    const tot = (xmax / 1000).toFixed(2) + ' km';
    g.fillText(tot, W - pad.r - g.measureText(tot).width, H - 4);
    (opt.bands || []).forEach((b) => {   // shaded spans (e.g. low-confidence stretches)
      g.fillStyle = b.color; g.fillRect(X(b.a), pad.t, Math.max(X(b.b) - X(b.a), 1), H - pad.t - pad.b);
    });
    g.beginPath();
    xs.forEach((v, i) => (i ? g.lineTo(X(v), Y(ys[i])) : g.moveTo(X(v), Y(ys[i]))));
    g.strokeStyle = opt.color; g.lineWidth = 2; g.lineJoin = 'round'; g.stroke();
    (opt.markers || []).forEach((m) => {
      g.strokeStyle = '#f2c94c'; g.lineWidth = 1; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(X(m), pad.t); g.lineTo(X(m), H - pad.b); g.stroke(); g.setLineDash([]);
    });
    if (hover !== null) {
      const x = X(xs[hover]);
      g.strokeStyle = '#aab1ba'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, H - pad.b); g.stroke();
      g.fillStyle = opt.color; g.strokeStyle = '#1b1e22'; g.lineWidth = 2;
      g.beginPath(); g.arc(x, Y(ys[hover]), 4, 0, 7); g.fill(); g.stroke();
    }
  };
  draw(null);
  const tip = $(tipId);
  cv.onmousemove = (ev) => {
    const r = cv.getBoundingClientRect();
    const sv = ((ev.clientX - r.left - pad.l) / (W - pad.l - pad.r)) * xmax;
    let i = 0; while (i < xs.length - 1 && xs[i] < sv) i++;
    draw(i);
    tip.classList.remove('hidden');
    tip.textContent = opt.label(i);
    tip.style.left = Math.min(ev.clientX - r.left + 14, W - 150) + 'px'; tip.style.top = '20px';
  };
  cv.onmouseleave = () => { draw(null); tip.classList.add('hidden'); };
}

function roadHitLine(r, layer) {
  const hit = L.polyline(r.built_line, { color: '#000', weight: 18, opacity: 0.001 }).addTo(layer);
  hit.on('click', (ev) => { L.DomEvent.stopPropagation(ev); insertWaypoint(ev.latlng); });
  hit.on('contextmenu', (ev) => { L.DomEvent.stopPropagation(ev); L.DomEvent.preventDefault(ev); widthPopup(ev.latlng); });
  hit.bindTooltip('Click: add a point here · Right-click: check / set the road width', { sticky: true });
}

// ------------------------------------------------------------------ road preview (the road only, no 3D)
let pvTimer = null, pvBusy = false, pvAgain = false;
const ROAD_FIELDS = ['tyres', 'markings', 'width_mode', 'width', 'width_smoothing', 'shoulder', 'corner_smoothing', 'elev_smoothing', 'camber',
  'start_offset', 'walls', 'wall_offset', 'grid', 'pits', 'reverse', 'ground', 'ground_res', 'dem_zoom'];

async function runPreview() {
  if (!ready()) { notReady(); return; }
  if (pvBusy) { pvAgain = true; return; }
  pvBusy = true;
  const btn = $('previewBtn'); btn.disabled = true; btn.textContent = 'Previewing…';
  try {
    const { job } = await api('/api/road_preview', collectParams());
    const r = await waitJob(job, (l) => status(l[l.length - 1] || 'Previewing the road…'));
    showPreview(r);
    status('');
  } catch (e) { status('Road preview failed: ' + e.message); }
  pvBusy = false; btn.disabled = false; btn.textContent = 'Preview road on map';
  if (pvAgain) { pvAgain = false; runPreview(); }
}

function schedulePreview() {
  if (!ready()) return;
  if ($('livePreview').checked) { clearTimeout(pvTimer); pvTimer = setTimeout(runPreview, 900); return; }
  if (state.preview) $('pvState').textContent = 'out of date - preview again';
}

function showPreview(r) {
  state.preview = r; state.road = r;
  if (map.hasLayer(builtLayer)) map.removeLayer(builtLayer);   // the last generated track's overlay
  previewLayer.clearLayers();
  if (r.area) {
    drawStreets(r, previewLayer);
    $('pvResult').classList.remove('hidden');
    $('pvStats').innerHTML = areaStats(r).join('');
    renderWarnings($('pvWarnings'), r.warnings);
    $('pvState').textContent = `updated ${new Date().toLocaleTimeString()} (${r.elapsed_s} s)`;
    renderOverrides();
    return;
  }
  const [a, b] = r.edges;
  const fill = { stroke: false, fillColor: '#1d2024', fillOpacity: 0.55, fillRule: 'evenodd', interactive: false };
  L.polygon(r.loop ? [a, b] : [a.concat(b.slice().reverse())], fill).addTo(previewLayer);
  (r.shoulder_lines || []).forEach((l) => L.polyline(l, { color: '#c9b98f', weight: 1.5, dashArray: '3 4', interactive: false }).addTo(previewLayer));
  (r.walls_lines || []).forEach((l) => L.polyline(l, { color: '#e8483b', weight: 2, opacity: 0.8 }).bindTooltip('concrete barrier').addTo(previewLayer));
  drawEdges(r, previewLayer);
  const mk = r.markings || {};
  drawTyres(r.tyres, previewLayer);
  if (mk.lines && mk.lines.length) drawMarkings(mk.lines, previewLayer);
  else L.polyline(r.built_line, { color: '#fff', weight: 1.5, dashArray: '6 6', interactive: false, opacity: 0.6 }).addTo(previewLayer);
  roadHitLine(r, previewLayer);
  r.objects.forEach((o) => {
    const c = o.name.startsWith('AC_START') ? '#27ae60' : o.name.startsWith('AC_PIT') ? '#2f80ed' : '#f2c94c';
    L.circleMarker([o.lat, o.lon], { radius: 3, color: c, weight: 2 }).bindTooltip(o.name).addTo(previewLayer);
  });
  const w = r.width || {};
  const e = r.elevation;
  $('pvResult').classList.remove('hidden');
  $('pvStats').innerHTML = [
    stat((r.length_m / 1000).toFixed(2) + ' km', 'length'),
    w.used ? stat(w.median + ' m', `width on photo (${w.min}–${w.max})`) : stat($('width').value + ' m', 'width (fixed)'),
    w.used ? stat(Math.round(w.confident_fraction * 100) + '%', 'clear edges') : '',
    markingsStat(mk),
    (r.tyres && r.tyres.info && r.tyres.info.used) ? stat(`${r.tyres.info.length_m} m`, 'tyre walls on the photo') : '',
    stat(r.min_radius_m + ' m', 'tightest radius'),
    stat(e.max_grade_pct + '%', 'max gradient'),
    stat(e.gain + ' m', 'climb'),
  ].join('');
  renderWarnings($('pvWarnings'), r.warnings);
  $('pvState').textContent = `updated ${new Date().toLocaleTimeString()} (${r.elapsed_s} s)`;
  drawWidth(r, 'pvWidth', 'pvWidthTip');
  drawProfile(r.profile, 'pvProfile', 'pvProfileTip');
  renderOverrides();
}

$('previewBtn').onclick = runPreview;
$('livePreview').checked = !!pref('livePreview');
$('livePreview').onchange = () => { pref('livePreview', $('livePreview').checked); if ($('livePreview').checked) runPreview(); };
ROAD_FIELDS.forEach((k) => { const el = $(k); if (el) el.addEventListener('change', schedulePreview); });

// road markings: each painted piece in its colour (a dashed line is drawn dash by dash, as it will be built)
function drawMarkings(lines, layer) {
  lines.forEach((ln) => {
    const color = ln.colour === 'yellow' ? '#f2c94c' : '#ffffff';
    const tip = `${ln.colour} ${ln.style} line (${ln.side})`;
    ln.segments.forEach((seg) => {
      if (seg.length > 1) L.polyline(seg, { color, weight: 2, opacity: 0.95 }).bindTooltip(tip).addTo(layer);
    });
  });
}

// tyre walls found on the photo: thick dark lines with a light outline
function drawTyres(ty, layer) {
  if (!ty || !ty.lines || !ty.lines.length) return;
  ty.lines.forEach((seg) => {
    if (seg.length < 2) return;
    L.polyline(seg, { color: '#f2f2f2', weight: 6, opacity: 0.7, interactive: false }).addTo(layer);
    L.polyline(seg, { color: '#111', weight: 4, opacity: 0.95, dashArray: '2 3' }).bindTooltip('tyre wall (from the photo)').addTo(layer);
  });
}

function markingsStat(mk) {
  const info = (mk && mk.info) || {};
  if (info.source === 'photo') {
    const n = (mk.lines || []).length;
    return stat(n ? `${n} line${n > 1 ? 's' : ''}` : 'none', n ? 'markings read on the photo' : 'painted lines on the photo');
  }
  if (info.source === 'standard') return stat('standard', 'markings');
  if (info.source === 'none') return stat('none', 'markings');
  return '';
}

function drawProfile(pr, canvas = 'profile', tip = 'profileTip') {
  lineChart(canvas, tip, pr.s, pr.z, { color: '#e8483b', minSpan: 5,
    label: (i) => `${(pr.s[i] / 1000).toFixed(2)} km · ${pr.z[i].toFixed(1)} m` });
}

function drawWidth(rep, canvas = 'widthChart', tipId = 'widthTip') {
  const s = rep.built_s, w = rep.built_w, c = rep.edge_conf || [];
  if (!s || !w) return;
  const bands = [];
  let a = null;
  for (let i = 0; i < s.length; i++) {
    const low = c.length && c[i] < 0.3;
    if (low && a === null) a = s[i];
    if ((!low || i === s.length - 1) && a !== null) { if (s[i] - a >= 10) bands.push({ a, b: s[i], color: 'rgba(242,153,74,0.14)' }); a = null; }
  }
  const markers = state.widthOverrides.map((o) => s[nearestBuilt(L.latLng(o.lat, o.lon))]);
  lineChart(canvas, tipId, s, w, { color: '#56ccf2', minSpan: 2, decimals: 1, floor: 0, bands, markers,
    label: (i) => `${(s[i] / 1000).toFixed(2)} km · ${w[i].toFixed(1)} m wide` + (c.length && c[i] < 0.3 ? ' · faint edges here (check on the map)' : '') });
}

function drawEdges(rep, layer = builtLayer) {
  const conf = rep.edge_conf || [];
  (rep.edges || []).forEach((line) => {
    if (!conf.length) { L.polyline(line, { color: '#f2c94c', weight: 1.5 }).addTo(layer); return; }
    // split into runs of confident / unclear edges
    let start = 0;
    for (let i = 1; i <= line.length; i++) {
      const cur = conf[Math.min(i, conf.length - 1)] >= 0.3;
      const prev = conf[Math.min(i - 1, conf.length - 1)] >= 0.3;
      if (i === line.length || cur !== prev) {
        L.polyline(line.slice(start, Math.min(i + 1, line.length)), { color: prev ? '#27ae60' : '#f2994a', weight: 2 }).addTo(layer);
        start = i;
      }
    }
  });
}

function nearestBuilt(ll) {
  const pts = state.road.built_line;
  const k = Math.cos(ll.lat * Math.PI / 180);
  let best = 0, bd = Infinity;
  pts.forEach((p, i) => { const d = (p[0] - ll.lat) ** 2 + ((p[1] - ll.lng) * k) ** 2; if (d < bd) { bd = d; best = i; } });
  return best;
}

function widthPopup(ll) {
  const rep = state.road;
  const i = nearestBuilt(ll);
  const at = L.latLng(rep.built_line[i][0], rep.built_line[i][1]);
  const w = rep.built_w ? rep.built_w[i] : null;
  const c = rep.edge_conf && rep.edge_conf.length ? rep.edge_conf[i] : null;
  const html = `<div class="wpop">
    <div><b>${rep.built_s ? (rep.built_s[i] / 1000).toFixed(2) + ' km' : esc(rep.built_info ? rep.built_info[i] + ' street' : 'street')}</b> · built width <b>${w !== null ? w.toFixed(1) + ' m' : '?'}</b></div>
    <div class="muted">${rep.area ? 'the whole street gets this width' : c === null ? 'fixed width' : c >= 0.3 ? 'edges clearly visible on the photo' : 'edges unclear on the photo here'}</div>
    <div class="row">Set width <input id="wpW" type="number" step="0.1" min="3" max="30" value="${w !== null ? w.toFixed(1) : 8}"> m</div>
    <div class="row">over <select id="wpS"><option value="20">20 m</option><option value="40" selected>40 m</option><option value="100">100 m</option><option value="300">300 m</option><option value="100000" ${rep.area ? 'selected' : ''}>whole ${rep.area ? 'street' : 'track'}</option></select></div>
    <div class="row"><button class="go" id="wpOk">Apply</button><span class="muted">${$('livePreview').checked ? 'the preview updates' : 'then preview or generate again'}</span></div></div>`;
  const pop = L.popup({ maxWidth: 280 }).setLatLng(at).setContent(html).openOn(map);
  setTimeout(() => {
    const ok = document.getElementById('wpOk');
    if (!ok) return;
    ok.onclick = () => {
      const width = parseFloat(document.getElementById('wpW').value);
      const span = parseFloat(document.getElementById('wpS').value);
      if (!(width > 2 && width < 40)) return;
      state.widthOverrides.push({ lat: at.lat, lon: at.lng, width, span });
      renderOverrides(); saveDraft(); map.closePopup(pop);
      L.circleMarker(at, { radius: 6, color: '#f2c94c', weight: 3 }).bindTooltip(`width ${width} m`).addTo(state.preview ? previewLayer : builtLayer);
      status('Width set');
      schedulePreview();
    };
  }, 0);
}

function renderOverrides() {
  const ul = $('ovList');
  if (!ul) return;
  ul.innerHTML = '';
  state.widthOverrides.forEach((o, i) => {
    const li = document.createElement('li');
    const where = state.road && state.road.built_s ? `${(state.road.built_s[nearestBuilt(L.latLng(o.lat, o.lon))] / 1000).toFixed(2)} km` : `${o.lat.toFixed(5)}, ${o.lon.toFixed(5)}`;
    li.innerHTML = `<span>${esc(where)}: <b>${o.width.toFixed(1)} m</b> over ${o.span >= 100000 ? 'whole track' : o.span + ' m'}</span>`;
    const del = document.createElement('button'); del.textContent = '✕'; del.title = 'Remove';
    del.onclick = () => { state.widthOverrides.splice(i, 1); renderOverrides(); saveDraft(); schedulePreview(); };
    li.appendChild(del);
    ul.appendChild(li);
  });
  $('ovHint').classList.toggle('hidden', state.widthOverrides.length > 0);
}

// ------------------------------------------------------------------ result actions
$('view3dBtn').onclick = () => switchTab('3d');
$('openBtn').onclick = () => state.slug && api(`/api/projects/${state.slug}/open`, {}).catch((e) => status(e.message));
$('blenderBtn').onclick = async () => {
  if (!state.slug) return;
  $('blenderBtn').disabled = true;
  try {
    const { job } = await api(`/api/projects/${state.slug}/blender`, {});
    const r = await waitJob(job, showLog);
    showLog([...$('log').textContent.split('\n'), 'Blender scene ready: ' + r.blend, 'Edit and save it in Blender, then Build KN5.']);
  } catch (e) { showLog([...$('log').textContent.split('\n'), 'ERROR: ' + e.message]); }
  $('blenderBtn').disabled = false;
};
$('kn5Source').onchange = () => $('kn5PathRow').classList.toggle('hidden', $('kn5Source').value !== 'fbx_other');
$('kn5Btn').onclick = async () => {
  if (!state.slug) return;
  const v = $('kn5Source').value;
  const body = { source: v === 'blend' ? 'blend' : 'fbx', path: v === 'fbx_other' ? $('kn5Path').value.trim() : null };
  if (v === 'fbx_other' && !body.path) { status('Enter the path of the FBX file'); return; }
  $('kn5Btn').disabled = true;
  try {
    const { job } = await api(`/api/projects/${state.slug}/kn5`, body);
    const r = await waitJob(job, showLog);
    showLog([...$('log').textContent.split('\n'), `KN5 ready: ${r.kn5}`, r.starts ? 'Click Install to AC to try it.' : 'WARNING: no AC_START_ empty - the game cannot spawn a car.']);
    $('kn5Status').textContent = `KN5 rebuilt from ${v === 'blend' ? 'your Blender scene' : 'FBX'}: ${r.meshes} meshes.`;
  } catch (e) { showLog([...$('log').textContent.split('\n'), 'ERROR: ' + e.message]); }
  $('kn5Btn').disabled = false;
};
$('rf2Btn').onclick = async () => {
  if (!state.slug) return;
  const btn = $('rf2Btn'); btn.disabled = true; btn.textContent = 'Exporting…';
  try {
    const { job } = await api(`/api/projects/${state.slug}/rf2`, {});
    const r = await waitJob(job, showLog);
    let msg = `rF2 location ${r.loc}: ${r.gmt} .gmt files, ${r.waypoints} AI waypoints, ${r.grid} grid spots.`;
    if (r.missing && r.missing.length) msg += ' Not complete - missing: ' + r.missing.join('; ') + '.';
    else if (r.installed) msg += ` Installed in ModDev - start rFactor 2 Mod Mode and pick "${r.loc}".`;
    $('rf2Status').textContent = msg;
  } catch (e) { showLog([...$('log').textContent.split('\n'), 'rFactor 2 export failed: ' + e.message]); }
  btn.disabled = false; btn.textContent = 'Export to rFactor 2';
};
$('rf2PkgBtn').onclick = async () => {
  if (!state.slug) return;
  const btn = $('rf2PkgBtn'); btn.disabled = true; btn.textContent = 'Packaging…';
  $('rf2Status').innerHTML = 'Packing… when <b>MAS2</b> opens: click the yellow box button → <b>Create Single Cmp Package</b> → <b>Package</b>, then close MAS2.';
  try {
    const { job } = await api(`/api/projects/${state.slug}/rf2_package`, {});
    const r = await waitJob(job, showLog);
    $('rf2Status').textContent = `"${r.loc}" ${r.version} installed - start rFactor 2 normally and pick it in the track list. Package to share: ${r.rfcmp}`;
  } catch (e) {
    $('rf2Status').textContent = 'Packaging stopped: ' + e.message;
    showLog([...$('log').textContent.split('\n'), 'rFactor 2 packaging failed: ' + e.message]);
  }
  btn.disabled = false; btn.textContent = 'Package for rFactor 2';
};
$('installBtn').onclick = async () => {
  if (!state.slug) return;
  try {
    const r = await api(`/api/projects/${state.slug}/install`, {});
    showLog([...$('log').textContent.split('\n'), 'Installed to ' + r.dest, r.note || 'Track is ready in Assetto Corsa.']);
  } catch (e) { showLog([...$('log').textContent.split('\n'), 'Install failed: ' + e.message]); }
};

// ------------------------------------------------------------------ projects & settings
async function refreshProjects() {
  const list = await api('/api/projects');
  const sel = $('projectSelect');
  sel.innerHTML = '<option value="">Open project…</option>' + list.map((p) => `<option value="${p.slug}">${p.slug}${p.has_kn5 ? ' ✓kn5' : ''}</option>`).join('');
}
$('projectSelect').onchange = async (e) => {
  const slug = e.target.value; if (!slug) return;
  const { project, report } = await api('/api/projects/' + slug);
  applyParams(project); changed(); fitWaypoints();
  if (report) { showReport(report); loadGLB(report.files.glb + '?t=' + Date.now()); }
  e.target.value = '';
};

$('settingsBtn').onclick = () => {
  const s = window.__settings || {};
  $('set_blender').value = s.blender_path || ''; $('set_ac').value = s.ac_tracks_path || ''; $('set_author').value = s.author || '';
  $('set_rf2').value = s.rf2_path || ''; $('set_b36').value = s.rf2_blender_path || '';
  $('rf2AddonState').textContent = window.__rf2addon ? `Exporter found: ${window.__rf2addon} - it needs ${window.__rf2need || 'Blender 5'}. A portable blender-5.0.x-windows-x64 folder unzipped into the maptotrack folder is found by itself.`
    : 'Exporter not found: put Studio 397\'s io_rfactor2_gmt….zip (forum: "Blender rFactor2 Export Scripts") in the maptotrack folder.';
  $('settingsDlg').showModal();
};
$('saveSettings').onclick = async () => {
  window.__settings = await api('/api/settings', { blender_path: $('set_blender').value, ac_tracks_path: $('set_ac').value, author: $('set_author').value,
    rf2_path: $('set_rf2').value, rf2_blender_path: $('set_b36').value });
};

// ------------------------------------------------------------------ tabs
function switchTab(t) {
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === t));
  $('map').classList.toggle('hidden', t !== 'map');
  $('view3d').classList.toggle('hidden', t !== '3d');
  if (t === 'map') map.invalidateSize();
  else { init3D(); resize3D(); }
}
document.querySelectorAll('.tabs button').forEach((b) => { b.onclick = () => switchTab(b.dataset.tab); });

// ------------------------------------------------------------------ 3D preview
let renderer, scene, camera, controls, model = null, pendingGLB = null;
function init3D() {
  if (renderer) return;
  const el = $('three');
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9fc4e8);
  scene.fog = new THREE.Fog(0x9fc4e8, 1500, 6000);
  camera = new THREE.PerspectiveCamera(55, 1, 0.5, 20000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.maxPolarAngle = Math.PI * 0.495;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445533, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.0); sun.position.set(400, 800, 300); scene.add(sun);
  const loop = () => { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); };
  loop();
  window.addEventListener('resize', resize3D);
  if (pendingGLB) { const u = pendingGLB; pendingGLB = null; loadGLB(u); }
}
function resize3D() {
  if (!renderer) return;
  const el = $('three'); const w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
function resetCam() {
  if (!model) return;
  const box = new THREE.Box3().setFromObject(model);
  const c = box.getCenter(new THREE.Vector3()); const size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.z) * 0.7;
  controls.target.copy(c);
  camera.position.set(c.x + r * 0.4, c.y + r * 0.8, c.z + r * 0.9);
  camera.far = r * 10; camera.updateProjectionMatrix();
}
$('resetCam').onclick = resetCam;
$('startCam').onclick = () => {
  const o = state.report && state.report.objects.find((x) => x.name === 'AC_START_0');
  if (!o || !camera) return;
  // track frame (x east, y north, z up) -> three.js (x, y up, z south)
  const P = new THREE.Vector3(o.pos[0], o.pos[2], -o.pos[1]);
  const F = new THREE.Vector3(o.forward[0], 0, -o.forward[1]).normalize();
  camera.position.copy(P).addScaledVector(F, -14).add(new THREE.Vector3(0, 4, 0));
  controls.target.copy(P).addScaledVector(F, 25);
  camera.near = 0.2; camera.updateProjectionMatrix();
};
function loadGLB(url) {
  if (!renderer) { pendingGLB = url; return; }
  status('Loading 3D model…');
  new GLTFLoader().load(url, (g) => {
    if (model) scene.remove(model);
    model = g.scene;
    model.traverse((o) => {
      if (o.isMesh) {
        o.material.side = THREE.FrontSide;
        if (o.material.map) { o.material.map.anisotropy = 8; }
      }
    });
    scene.add(model);
    $('empty3d').classList.add('hidden');
    applyGroupVisibility(); resize3D(); resetCam(); status('');
  }, undefined, (err) => status('3D load failed: ' + err.message));
}
function applyGroupVisibility() {
  if (!model) return;
  document.querySelectorAll('.overlay3d input[data-grp]').forEach((cb) => {
    const prefixes = cb.dataset.grp.split(',');
    model.traverse((o) => {
      if (o.isMesh && prefixes.some((p) => o.name.startsWith(p))) o.visible = cb.checked;
    });
  });
}
document.querySelectorAll('.overlay3d input[data-grp]').forEach((cb) => { cb.onchange = applyGroupVisibility; });

// ------------------------------------------------------------------ boot
(async function boot() {
  const cfg = await api('/api/config');
  window.__settings = cfg.settings;
  window.__rf2addon = cfg.rf2_addon || '';
  window.__rf2need = cfg.rf2_need || '';
  if (cfg.offline) $('offlineBanner').classList.remove('hidden');
  const gsel = $('ground');
  (cfg.imagery || []).forEach((src) => { const o = document.createElement('option'); o.value = src.id; o.textContent = 'Aerial: ' + src.name; gsel.appendChild(o); });
  if (!cfg.offline) addPhotoLayers(cfg.imagery);
  [['landcover', 'Land cover colours (clean colours, surfaces read on the photo)'], ['plain', 'Plain grass texture']].forEach(([v, t]) => {
    const o = document.createElement('option'); o.value = v; o.textContent = t; gsel.appendChild(o); });
  const d = cfg.defaults;
  NUM_FIELDS.forEach((k) => { $(k).value = d[k]; });
  SEL_FIELDS.forEach((k) => { $(k).value = String(d[k]); });
  CHK_FIELDS.forEach((k) => { $(k).checked = !!d[k]; });
  $('conifer_pct').value = Math.round(d.conifer_ratio * 100);
  let draft = null;
  try { draft = JSON.parse(localStorage.getItem('maptotrack.draft') || 'null'); } catch (e) { /* ignore */ }
  if (draft && ((draft.waypoints && draft.waypoints.length) || draft.area)) { applyParams(draft); fitWaypoints(); }
  changed();
  refreshProjects();
})();
