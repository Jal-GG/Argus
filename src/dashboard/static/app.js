/* Surveillance dashboard — vanilla JS, no build step. */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  zones: [],
  planImage: null,
  planSize: { w: 800, h: 600 },
  live: { positions: {}, paths: {}, alerts: [], tracks: [], predictions: {} },
  heatImage: null,
  alertCount: 0,
  replay: null,          // {trails, t0, frames} while a replay animates
};

/* ------------------------------------------------------------------ */
/* WebSocket live feed                                                 */
/* ------------------------------------------------------------------ */
let ws = null;
let wsRetryMs = 1500;

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    $("#conn-dot").style.background = "var(--ok)";
    $("#conn-text").textContent = "live";
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") onState(msg);
  };
  ws.onclose = () => {
    $("#conn-dot").style.background = "#888";
    $("#conn-text").textContent = "reconnecting…";
    setTimeout(connectWs, wsRetryMs);
  };
  ws.onerror = () => ws.close();
}

function onState(msg) {
  state.live.positions = msg.positions || {};
  state.live.paths = msg.paths || {};
  state.live.alerts = msg.alerts || [];
  state.live.tracks = msg.tracks || [];
  state.live.predictions = msg.predictions || {};

  // People = distinct global IDs with a position; fallback to track count.
  const people = Object.keys(state.live.positions).length ||
                 new Set(state.live.tracks.map(t => t.global_id ?? t.track_id)).size;
  $("#stat-people").textContent = people;
  $("#stat-tracks").textContent = state.live.tracks.length;
  $("#stat-alerts").textContent = Math.max(state.alertCount, state.live.alerts.length);

  if (msg.cameras) {
    const online = Object.values(msg.cameras).filter(c => c.online).length;
    $("#stat-cameras").textContent = `${online}/${Object.keys(msg.cameras).length} cams`;
  }
  drawPlan();
}

/* ------------------------------------------------------------------ */
/* Floor plan canvas                                                   */
/* ------------------------------------------------------------------ */
async function loadPlan() {
  try {
    const [info, imgBlob] = await Promise.all([
      fetch("/api/floorplan").then(r => r.ok ? r.json() : null),
      fetch("/api/floorplan/image").then(r => r.ok ? r.blob() : null),
    ]);
    if (info) {
      state.zones = info.zones;
      state.planSize = { w: info.width, h: info.height };
      $("#floor-id").textContent = `(${info.floor_id})`;
      const canvas = $("#plan-canvas");
      canvas.width = info.width;
      canvas.height = info.height;
    }
    if (imgBlob) {
      state.planImage = await createImageBitmap(imgBlob);
    }
  } catch (err) {
    console.warn("Floor plan unavailable yet:", err);
  }
}

function zoneColor(type) {
  switch (type) {
    case "required": return getCss("--required");
    case "restricted": return getCss("--restricted");
    case "safe": return getCss("--safe");
    default: return "#8a8f98";
  }
}
function getCss(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

function idColor(gidStr) {
  let h = 0;
  for (const ch of String(gidStr)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsl(${h % 360} 75% 60%)`;
}

function drawPlan() {
  const canvas = $("#plan-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (state.planImage) {
    ctx.drawImage(state.planImage, 0, 0, canvas.width, canvas.height);
  }
  if ($("#toggle-heat")?.checked && state.heatImage) {
    ctx.drawImage(state.heatImage, 0, 0, canvas.width, canvas.height);
  }

  // Zones
  for (const z of state.zones) {
    ctx.beginPath();
    z.polygon.forEach(([x, y], i) =>
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fillStyle = zoneColor(z.type) + "22";
    ctx.fill();
    ctx.strokeStyle = zoneColor(z.type);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    if (z.polygon.length) {
      const cx = z.polygon.reduce((s, p) => s + p[0], 0) / z.polygon.length;
      const cy = z.polygon.reduce((s, p) => s + p[1], 0) / z.polygon.length;
      ctx.fillStyle = zoneColor(z.type);
      ctx.font = "11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(z.name, cx, cy);
    }
  }

  // Paths (trails)
  for (const [gid, pts] of Object.entries(state.live.paths)) {
    if (!pts || pts.length < 2) continue;
    const alerted = state.live.alerts.includes(Number(gid));
    ctx.beginPath();
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.strokeStyle = alerted ? getCss("--alert") : idColor(gid);
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Predictions (dashed extrapolation)
  if ($("#toggle-predict")?.checked) {
    for (const pred of Object.values(state.live.predictions || {})) {
      const pts = pred.points || [];
      if (pts.length < 2) continue;
      ctx.beginPath();
      pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "#b388ff";
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.setLineDash([]);
      // Next-zone hint
      if (pred.next_zone && pts.length) {
        const [ex, ey] = pts[pts.length - 1];
        ctx.fillStyle = "#b388ff";
        ctx.font = "10px system-ui";
        ctx.fillText(`→ ${pred.next_zone}`, ex + 6, ey + 4);
      }
    }
  }

  // Positions
  for (const [gid, pos] of Object.entries(state.live.positions)) {
    const [x, y] = pos;
    const alerted = state.live.alerts.includes(Number(gid));
    const color = alerted ? getCss("--alert") : idColor(gid);

    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Enrolled name (Phase 11) if any track carries one.
    const named = state.live.tracks.find(
      t => String(t.global_id) === gid && t.name);
    const label = named ? `#${gid} ${named.name}` : `#${gid}`;
    ctx.fillStyle = "#fff";
    ctx.font = "bold 11px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(label, x + 10, y - 8);
  }
}

/* ------------------------------------------------------------------ */
/* Camera tiles                                                        */
/* ------------------------------------------------------------------ */
async function refreshCameras() {
  try {
    const cams = await fetch("/api/cameras").then(r => r.json());
    const grid = $("#camera-grid");

    for (const cam of cams) {
      let tile = document.getElementById(`cam-${cam.id}`);
      if (!tile) {
        tile = document.createElement("div");
        tile.className = "cam-tile";
        tile.id = `cam-${cam.id}`;
        const img = document.createElement("img");
        img.alt = cam.id;
        img.src = `/api/cameras/${encodeURIComponent(cam.id)}/stream?fps=10`;
        const label = document.createElement("span");
        label.className = "cam-label";
        label.textContent = cam.name || cam.id;
        tile.append(img, label);
        grid.appendChild(tile);
      }
      tile.classList.toggle("cam-offline", !cam.online);
      const label = tile.querySelector(".cam-label");
      label.textContent = `${cam.name || cam.id} · ${cam.fps.toFixed(0)} fps`;
    }
  } catch (err) { /* server restarting */ }
}

/* ------------------------------------------------------------------ */
/* Event feed                                                          */
/* ------------------------------------------------------------------ */
let lastEventTs = 0;

async function pollEvents() {
  try {
    const events = await fetch(`/api/events?limit=25`).then(r => r.json());
    const feed = $("#event-feed");
    if (events.length && feed.firstElementChild?.classList.contains("muted")) {
      feed.innerHTML = "";
    }
    for (const ev of events.slice().reverse()) {
      if (ev.ts <= lastEventTs) continue;
      lastEventTs = Math.max(lastEventTs, ev.ts);
      if (/restricted|loitering|speed|violation/i.test(ev.rule_type)) {
        state.alertCount++;
      }
      const li = document.createElement("li");
      li.className = /restricted|speed/.test(ev.rule_type) ? "violation" : "";
      const t = new Date(ev.ts * 1000).toLocaleTimeString();
      li.innerHTML =
        `<div><b>#${ev.global_id ?? "?"}</b> ${ev.reason}</div>` +
        `<span class="ev-time">${t} · ${ev.zone_name} · ${ev.rule_type}</span>`;
      feed.prepend(li);
      while (feed.children.length > 40) feed.lastChild.remove();
    }
  } catch (err) { /* ignore */ }
}

/* ------------------------------------------------------------------ */
/* Heatmap layer                                                       */
/* ------------------------------------------------------------------ */
let heatFetchInFlight = false;

async function pollHeatmap() {
  if (!$("#toggle-heat")?.checked || heatFetchInFlight) return;
  heatFetchInFlight = true;
  try {
    const blob = await fetch("/api/floorplan/heatmap").then(r => r.ok ? r.blob() : null);
    state.heatImage = blob ? await createImageBitmap(blob) : null;
  } catch { /* ignore */ }
  heatFetchInFlight = false;
}

/* ------------------------------------------------------------------ */
/* Historical replay                                                   */
/* ------------------------------------------------------------------ */
async function startReplay() {
  const btn = $("#btn-replay");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const data = await fetch("/api/history/positions?minutes=10&limit=50000")
      .then(r => r.ok ? r.json() : null);
    if (data && data.count > 0) {
      // Downsample each trail to ~120 points for smooth animation.
      const trails = {};
      for (const [gid, pts] of Object.entries(data.trails)) {
        const step = Math.max(1, Math.ceil(pts.length / 120));
        trails[gid] = pts.filter((_, i) => i % step === 0);
      }
      state.replay = { trails, progress: 0 };
      animateReplay();
    } else {
      btn.textContent = "No history yet";
      setTimeout(() => { btn.textContent = "▶ Replay 10 min"; }, 2000);
    }
  } catch {
    btn.textContent = "▶ Replay 10 min";
  }
  btn.disabled = false;
}

function animateReplay() {
  const rep = state.replay;
  if (!rep) return;
  rep.progress += 0.01;
  drawPlan();   // drawPlan renders replay overlay when present
  if (rep.progress >= 1.0) {
    state.replay = null;
    $("#btn-replay").textContent = "▶ Replay 10 min";
    return;
  }
  requestAnimationFrame(animateReplay);
}

// Replay overlay: animate stored trails up to current progress.
if (state.replay) {
  const rep = state.replay;
  for (const [gid, pts] of Object.entries(rep.trails)) {
    const upto = Math.max(2, Math.floor(pts.length * rep.progress));
    const slice = pts.slice(0, upto);
    if (slice.length < 2) continue;
    ctx.beginPath();
    slice.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.strokeStyle = idColor(gid) + "cc";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(0,0,0,.55)";
  ctx.fillRect(canvas.width - 130, 10, 118, 22);
  ctx.fillStyle = "#fff";
  ctx.font = "12px system-ui";
  ctx.textAlign = "left";
  ctx.fillText(`replay ${Math.round(rep.progress * 100)}%`, canvas.width - 120, 25);
}

/* ------------------------------------------------------------------ */
setInterval(refreshCameras, 4000);
setInterval(pollEvents, 2000);
$("#toggle-heat")?.addEventListener("change", () => { state.heatImage = null; });
setInterval(pollHeatmap, 2500);
$("#btn-replay")?.addEventListener("click", startReplay);
loadPlan().then(() => connectWs());
