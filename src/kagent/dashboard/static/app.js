"use strict";
/* kagent dashboard. Every piece of text from the server is untrusted (script output, model
   replies), so it only ever reaches the page through textContent and text nodes. */
(() => {
  const POLL_MS = 2000;
  const NS = "http://www.w3.org/2000/svg";
  const OPERATORS = ["draft", "improve", "debug"];
  const STATUSES = ["ok", "buggy", "dead", "blocked"];
  const STAGES = ["start", "load", "features", "cv", "predict", "validate"];
  const STAGE_TEXT = {
    start: "no progress seen",
    load: "data loaded",
    features: "features built",
    cv: "CV finished",
    predict: "submission written",
    validate: "graded ok",
  };
  const GLYPH = { ok: "✓", buggy: "!", dead: "✕", blocked: "…" };
  const STATUS_WORD = {
    ok: "ok",
    buggy: "failed, will be debugged",
    dead: "dead, gave up",
    blocked: "waiting on a package",
  };
  const VERDICT_WORD = {
    better: "beats the noise",
    noise: "within the noise",
    worse: "worse than its parent",
  };
  const RUN_STATE = {
    running: { glyph: "▶", label: "Running" },
    finished: { glyph: "✓", label: "Finished" },
    stopped: { glyph: "■", label: "Stopped" },
    lost: { glyph: "✕", label: "Stopped unexpectedly" },
    starting: { glyph: "…", label: "Starting" },
    crashed: { glyph: "✕", label: "Did not start" },
    never: { glyph: "○", label: "No search yet" },
  };
  const DEP_GLYPH = { pending: "…", approved: "✓", denied: "✕", expired: "✕" };
  const DEP_CLASS = { pending: "blocked", approved: "ok", denied: "dead", expired: "buggy" };

  const S = {
    state: null,
    etag: null,
    skew: 0,
    events: [],
    eventLast: 0,
    calls: [],
    callLast: -1,
    filter: { op: "all", status: "all" },
    sort: { key: "seq", dir: -1 },
    selected: null,
    logTab: "stdout",
    lastOk: 0,
    error: null,
    seenNodes: 0,
  };

  // ---- small helpers -------------------------------------------------------------------

  const $ = (selector) => document.querySelector(selector);

  function build(el, props, kids) {
    for (const [key, value] of Object.entries(props || {})) {
      if (value == null || value === false) continue;
      if (key === "text") el.textContent = value;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? "" : String(value));
    }
    el.append(...kids.flat(Infinity).filter((k) => k != null && k !== false));
    return el;
  }
  const h = (tag, props, ...kids) => build(document.createElement(tag), props, kids);
  const sv = (tag, props, ...kids) => build(document.createElementNS(NS, tag), props, kids);

  const f4 = (x) => (x == null ? "–" : x.toFixed(4));
  const fgain = (g) => (g == null ? "–" : (g > 0 ? "+" : "") + g.toFixed(4));
  const compact = (x) =>
    x == null ? "–" : x >= 1e6 ? (x / 1e6).toFixed(1) + "M" : x >= 1e3 ? (x / 1e3).toFixed(1) + "K" : String(Math.round(x));
  function dur(sec) {
    if (sec == null || !isFinite(sec)) return "–";
    sec = Math.max(0, Math.round(sec));
    const hours = Math.floor(sec / 3600);
    const minutes = Math.floor((sec % 3600) / 60);
    const seconds = sec % 60;
    if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
    if (minutes) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
    return `${seconds}s`;
  }
  const dur1 = (x) => (x == null ? "–" : x < 10 ? x.toFixed(1) + "s" : dur(x));
  function clock(iso) {
    const d = new Date(iso);
    return isNaN(d) ? "" : d.toLocaleTimeString([], { hour12: false });
  }
  const nowSec = () => Date.now() / 1000 + S.skew;
  const op = (name) => (OPERATORS.includes(name) ? name : "draft");
  const matches = (d) =>
    (S.filter.op === "all" || d.operator === S.filter.op) &&
    (S.filter.status === "all" || d.status === S.filter.status);

  function niceTicks(lo, hi, count) {
    const raw = (hi - lo) / count || 0.01;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(+v.toFixed(10));
    return { ticks, decimals: Math.max(0, -Math.floor(Math.log10(step))) };
  }

  // Bar with a rounded data end and a square baseline end.
  function barPath(x, w, y0, y1, r) {
    const h = Math.abs(y1 - y0);
    if (h < 0.5) return `M${x},${y0}h${w}`;
    r = Math.min(r, h, w / 2);
    const dir = y1 < y0 ? 1 : -1; // 1 when the bar grows upward
    const yr = y1 + dir * r;
    return `M${x},${y0}V${yr}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${yr}V${y0}Z`;
  }
  function barPathH(x0, y, x1, thick, r) {
    const w = x1 - x0;
    if (w < 0.5) return `M${x0},${y}v${thick}`;
    r = Math.min(r, w, thick / 2);
    return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + thick - r}Q${x1},${y + thick} ${x1 - r},${y + thick}H${x0}Z`;
  }

  function marker(operator, x, y, size, cls) {
    if (operator === "improve")
      return sv("polygon", { class: cls, points: `${x},${y - size} ${x + size},${y} ${x},${y + size} ${x - size},${y}` });
    if (operator === "debug")
      return sv("rect", { class: cls, x: x - size * 0.85, y: y - size * 0.85, width: size * 1.7, height: size * 1.7 });
    return sv("circle", { class: cls, cx: x, cy: y, r: size });
  }

  const statusChip = (status) =>
    h("span", { class: "inline" }, h("span", { class: `glyph ${status}`, "aria-hidden": "true", text: GLYPH[status] || "?" }), status);

  async function get(path, etag) {
    const res = await fetch(path, { headers: etag ? { "If-None-Match": etag } : {} });
    if (res.status === 304) return { same: true };
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${res.status}`);
    }
    return { data: await res.json(), etag: res.headers.get("ETag") };
  }

  // ---- tooltip -------------------------------------------------------------------------

  const tip = $("#tip");
  function showTip(content, point) {
    tip.replaceChildren(...content);
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    let x = point.x + 14;
    let y = point.y + 14;
    if (x + box.width > innerWidth - 8) x = point.x - box.width - 14;
    if (y + box.height > innerHeight - 8) y = point.y - box.height - 14;
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  }
  const hideTip = () => {
    tip.hidden = true;
  };

  function nodeTip(d) {
    const rows = [
      h("div", { class: "t", text: `${d.id} · ${d.operator}${d.parent ? " ← " + d.parent : ""}` }),
      h("div", { class: "row" }, h("span", { class: `mark round bg-${d.status}` }), `${GLYPH[d.status]} ${STATUS_WORD[d.status]}`),
    ];
    if (d.cv != null)
      rows.push(
        h("div", { class: "row" }, h("span", { class: "val num", text: f4(d.cv) }), h("span", { class: "key", text: `± ${f4(d.std)} std over folds` })),
      );
    if (d.gain != null)
      rows.push(
        h(
          "div",
          { class: "row" },
          h("span", { class: `mark bg-${d.verdict}` }),
          h("span", { class: "val num", text: fgain(d.gain) }),
          h("span", { class: "key", text: `vs parent: ${VERDICT_WORD[d.verdict]} (noise ±${f4(d.floor)})` }),
        ),
      );
    if (d.stage) rows.push(h("div", { class: "key", text: `reached: ${STAGE_TEXT[d.stage] || d.stage}` }));
    if (d.error) rows.push(h("div", { class: "err", text: d.error }));
    else if (d.summary) rows.push(h("div", { class: "err", text: d.summary }));
    return rows;
  }

  // Attach hover, keyboard focus and click to an element standing for a node.
  function wireNode(el, d, anchor) {
    el.addEventListener("pointermove", (ev) => showTip(nodeTip(d), { x: ev.clientX, y: ev.clientY }));
    el.addEventListener("pointerleave", hideTip);
    el.addEventListener("focus", () => {
      const box = anchor ? anchor() : el.getBoundingClientRect();
      showTip(nodeTip(d), { x: box.left + box.width / 2, y: box.top + box.height / 2 });
    });
    el.addEventListener("blur", hideTip);
    el.addEventListener("click", () => openNode(d.id));
    el.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        openNode(d.id);
      }
    });
  }

  // ---- header --------------------------------------------------------------------------

  function renderHeader() {
    const { competition: c, run } = S.state;
    const page = c.page && c.page.kind ? ` · ${c.page.kind}` : "";
    $("#comp").textContent = `${c.slug} · ${c.metric} ${c.higher_is_better ? "(higher is better)" : "(lower is better)"}${page}`;
    const info = RUN_STATE[run.state] || RUN_STATE.never;
    $("#pill").replaceChildren(
      h("span", { class: `glyph ${run.state}${run.state === "running" ? " pulse" : ""}`, "aria-hidden": "true", text: info.glyph }),
      info.label,
    );
  }

  function nowText() {
    const run = S.state && S.state.run;
    if (!run) return "";
    if (run.state === "never") return RUN_ON ? "No search has started in this workspace. Use Start search." : "No search has started in this workspace. Run `kagent search`; this page updates by itself.";
    if (run.state === "starting") return "Launching the search process. Preflight checks run first; see the console.";
    if (run.state === "crashed") return "The last launch ended before the search began. The console below shows why.";
    if (run.state !== "running") return run.message || (run.state === "lost" ? "The search process is gone without having finished." : "");
    const t = nowSec();
    const node = run.node ? ` ${run.node}` : "";
    const label =
      {
        starting: "Starting",
        selecting: "Choosing the next step",
        writing: `Writing${node}${run.operator ? " (" + run.operator + ")" : ""}`,
        running: `Running${node} in the sandbox`,
        evaluating: `Evaluating${node}`,
      }[run.phase] || run.phase;
    let text = `${label} · ${dur(t - run.phase_since)}`;
    if (run.detail) text += ` · ${run.detail}`;
    if (run.llm) text += ` · waiting on the ${run.llm.role} model for ${dur(t - run.llm.since)}`;
    return text;
  }

  function renderClock() {
    if (!S.state) return;
    const run = S.state.run;
    $("#now").textContent = nowText();

    const box = $("#budget");
    if (run.budget_s) {
      box.hidden = false;
      const end = run.state === "running" ? nowSec() : run.updated_at;
      const used = Math.max(0, end - run.started_at);
      const fraction = Math.min(1, used / run.budget_s);
      const fill = $("#budget-fill");
      fill.style.width = `${(fraction * 100).toFixed(1)}%`;
      fill.className = "fill" + (fraction > 0.9 ? " crit" : fraction > 0.75 ? " warn" : "");
      $("#budget-text").textContent =
        `${dur(used)} of ${dur(run.budget_s)} budget` + (run.state === "running" ? ` · ${dur(Math.max(0, run.budget_s - used))} left` : "");
    } else {
      box.hidden = true;
    }

    const live = $("#live");
    if (S.error) live.replaceChildren(h("span", { class: "glyph dead", "aria-hidden": "true", text: "✕" }), ` offline: ${S.error}`);
    else if (S.lastOk) live.replaceChildren(h("span", { class: "glyph running pulse", "aria-hidden": "true", text: "●" }), ` live · updated ${dur((Date.now() - S.lastOk) / 1000)} ago`);
    $("#grid").classList.toggle("stale", Boolean(S.error));
  }

  // ---- filters and KPI row -------------------------------------------------------------

  function buildFilters() {
    const make = (boxSel, key, values, markClass) => {
      const box = $(boxSel);
      for (const value of ["all", ...values]) {
        const chip = h(
          "button",
          { class: "chip", type: "button", "aria-pressed": value === "all" ? "true" : "false", "data-value": value, onclick: () => {
            S.filter[key] = value;
            renderFilters();
            if (S.state) renderBody();
          } },
          value === "all" ? null : h("span", { class: `mark ${markClass(value)}` }),
          h("span", { text: value }),
          h("span", { class: "muted count" }),
        );
        chip.dataset.key = key;
        box.append(chip);
      }
    };
    make("#f-op", "op", OPERATORS, (v) => `bg-${v}`);
    make("#f-status", "status", STATUSES, (v) => `bg-${v} round`);
  }

  function renderFilters() {
    const nodes = S.state ? S.state.nodes : [];
    for (const chip of document.querySelectorAll(".chip[data-key]")) {
      const key = chip.dataset.key;
      const value = chip.dataset.value;
      chip.setAttribute("aria-pressed", String(S.filter[key] === value));
      const count = value === "all" ? nodes.length : nodes.filter((d) => (key === "op" ? d.operator : d.status) === value).length;
      chip.querySelector(".count").textContent = String(count);
    }
  }

  function tile(label, value, sub, hero) {
    return h(
      "div",
      { class: "tile" + (hero ? " hero" : "") },
      h("div", { class: "k", text: label }),
      h("div", { class: "v" + (hero ? "" : " num"), text: value }),
      sub ? h("div", { class: "sub", text: sub }) : null,
    );
  }

  function renderKpis() {
    const st = S.state;
    const sum = st.summary;
    const best = sum.best;
    const inc = sum.incumbent;
    const improve = st.operators.improve || { better: 0, noise: 0, worse: 0 };
    const judged = improve.better + improve.noise + improve.worse;
    const roles = Object.values(st.llm);
    const calls = roles.reduce((a, r) => a + r.calls, 0);
    const tokensOut = roles.reduce((a, r) => a + r.tokens_out, 0);
    const seconds = roles.reduce((a, r) => a + r.seconds, 0);
    const waiting = st.deps.filter((d) => d.status === "pending");

    $("#kpis").replaceChildren(
      tile("Best score (robust CV)", best ? f4(best.cv) : "–", best ? `± ${f4(best.std)} · ${best.id} · ${best.operator}` : "no scored node yet", true),
      tile(
        "Incumbent",
        inc ? inc.id : "–",
        inc ? `${f4(inc.cv)}. A new node must beat it by ${f4(st.noise_k * (inc.std || 0))} to replace it.` : "the node the search builds on",
      ),
      tile("Nodes", String(sum.total), `${sum.ok} ok · ${sum.buggy} failed · ${sum.dead} dead · ${sum.blocked} waiting`),
      tile("Improvements that beat noise", judged ? `${improve.better} of ${judged}` : "–", judged ? `${improve.noise} within noise · ${improve.worse} worse` : "no improve node scored yet"),
      tile("Pace", sum.nodes_per_hour ? `${sum.nodes_per_hour.toFixed(1)}/h` : "–", `mean run ${dur1(sum.mean_run_s)}`),
      tile("Model calls", String(calls), seconds ? `${compact(tokensOut)} tokens out, ${(tokensOut / seconds).toFixed(1)} tok/s` : "none yet"),
      tile("Package requests", String(waiting.length), waiting.length ? `waiting: ${waiting.map((d) => d.package).join(", ")}` : "none waiting"),
    );
  }

  // ---- score timeline ------------------------------------------------------------------

  function legendItem(icon, text) {
    return h("span", { class: "item" }, icon, text);
  }
  const keySvg = (...kids) => sv("svg", { viewBox: "0 0 14 14" }, ...kids);

  function renderTimeline() {
    const st = S.state;
    const nodes = st.nodes;
    const box = $("#timeline");
    $("#timeline-sub").textContent =
      `${st.competition.metric}, ${st.competition.higher_is_better ? "higher" : "lower"} is better. Dots are scored nodes with their fold spread; failed nodes sit in the lower row. Every value is also in the table below.`;
    if (!nodes.length) {
      box.replaceChildren(h("div", { class: "empty", text: "No node yet." }));
      return;
    }

    const W = Math.max(box.clientWidth, 320);
    const H = 340;
    const m = { l: 54, r: 74, t: 12, b: 28 };
    const lane = 34;
    const plotB = H - m.b - lane - 6;
    const n = nodes.length;
    const step = (W - m.l - m.r) / n;
    const X = (i) => m.l + (i + 0.5) * step;
    const scored = nodes.filter((d) => d.cv != null);

    let lo = Infinity;
    let hi = -Infinity;
    for (const d of scored) {
      lo = Math.min(lo, d.cv - (d.std || 0));
      hi = Math.max(hi, d.cv + (d.std || 0));
    }
    if (!scored.length) [lo, hi] = [0, 1];
    const pad = (hi - lo) * 0.08 || 0.01;
    lo -= pad;
    hi += pad;
    const Y = (v) => m.t + ((hi - v) / (hi - lo)) * (plotB - m.t);
    const { ticks, decimals } = niceTicks(lo, hi, 5);

    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Score of each node over the search" });
    for (const v of ticks) {
      svg.append(
        sv("line", { class: "grid-line", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }),
        sv("text", { class: "num", x: m.l - 8, y: Y(v), "text-anchor": "end", "dominant-baseline": "central", text: v.toFixed(decimals) }),
      );
    }
    svg.append(sv("line", { class: "axis-line", x1: m.l, x2: W - m.r, y1: plotB, y2: plotB }));

    const byId = new Map(nodes.map((d) => [d.id, d]));
    const index = new Map(nodes.map((d, i) => [d.id, i]));
    const trail = scored
      .map((d) => ({ i: index.get(d.id), inc: byId.get(d.incumbent_after) }))
      .filter((t) => t.inc);
    const incNow = st.summary.incumbent;

    if (trail.length) {
      trail.forEach((t, k) => {
        const x0 = X(t.i);
        const x1 = k + 1 < trail.length ? X(trail[k + 1].i) : X(n - 1);
        const half = st.noise_k * (t.inc.std || 0);
        svg.append(
          sv("rect", { class: "f-ink", opacity: 0.08, x: x0, y: Y(t.inc.cv + half), width: Math.max(x1 - x0, 0), height: Math.max(Y(t.inc.cv - half) - Y(t.inc.cv + half), 0) }),
        );
      });
      let path = "";
      trail.forEach((t, k) => {
        const x = X(t.i);
        const y = Y(t.inc.cv);
        path += k === 0 ? `M${x},${y}` : `H${x}V${y}`;
      });
      path += `H${X(n - 1)}`;
      svg.append(sv("path", { class: "s-ink", d: path, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }

    const laneY = plotB + 8 + lane / 2;
    svg.append(sv("text", { x: m.l - 8, y: laneY, "text-anchor": "end", "dominant-baseline": "central", text: "failed" }));
    nodes.forEach((d, i) => {
      const x = X(i);
      const g = sv("g", { opacity: matches(d) ? 1 : 0.18 });
      if (d.cv != null) {
        const o = op(d.operator);
        if (d.std) g.append(sv("line", { class: `s-${o}`, x1: x, x2: x, y1: Y(d.cv - d.std), y2: Y(d.cv + d.std), "stroke-width": 1.5, opacity: 0.55 }));
        g.append(marker(o, x, Y(d.cv), 7, "f-surface"), marker(o, x, Y(d.cv), 5, `f-${o}`));
        if (incNow && d.id === incNow.id) g.append(sv("circle", { class: "s-ink", cx: x, cy: Y(d.cv), r: 10, fill: "none", "stroke-width": 2 }));
      } else {
        g.append(
          sv("circle", { class: `f-${d.status}`, cx: x, cy: laneY, r: 7 }),
          sv("text", { class: "glyph-text" + (d.status === "blocked" ? " dark" : ""), x, y: laneY, text: GLYPH[d.status] }),
        );
      }
      svg.append(g);
    });

    const every = Math.max(1, Math.ceil(46 / step));
    nodes.forEach((d, i) => {
      if (i % every === 0) svg.append(sv("text", { class: "num", x: X(i), y: H - 8, "text-anchor": "middle", text: d.id }));
    });
    if (incNow) {
      svg.append(
        sv("text", { x: W - m.r + 10, y: Y(incNow.cv) - 8, text: "incumbent" }),
        sv("text", { class: "label-ink num", x: W - m.r + 10, y: Y(incNow.cv) + 6, text: f4(incNow.cv) }),
      );
    }

    // One crosshair and one tooltip for the nearest node, reachable by keyboard too.
    const cross = sv("line", { class: "crosshair", y1: m.t, y2: plotB + lane + 10, visibility: "hidden" });
    const overlay = sv("rect", {
      class: "overlay", x: m.l, y: 0, width: W - m.l - m.r, height: H - m.b + 4, tabindex: 0, role: "application",
      "aria-label": "Score chart. Left and right arrows move between nodes, Enter opens one.",
    });
    let cursor = n - 1;
    const nearest = (clientX) => {
      const r = svg.getBoundingClientRect();
      return Math.min(n - 1, Math.max(0, Math.floor(((clientX - r.left) * (W / r.width) - m.l) / step)));
    };
    const show = (i, point) => {
      cursor = i;
      cross.setAttribute("x1", X(i));
      cross.setAttribute("x2", X(i));
      cross.setAttribute("visibility", "visible");
      if (!point) {
        const r = svg.getBoundingClientRect();
        point = { x: r.left + (X(i) * r.width) / W, y: r.top + (Y(nodes[i].cv != null ? nodes[i].cv : lo) * r.height) / H };
      }
      showTip(nodeTip(nodes[i]), point);
    };
    const leave = () => {
      cross.setAttribute("visibility", "hidden");
      hideTip();
    };
    overlay.addEventListener("pointermove", (ev) => show(nearest(ev.clientX), { x: ev.clientX, y: ev.clientY }));
    overlay.addEventListener("pointerleave", leave);
    overlay.addEventListener("click", (ev) => openNode(nodes[nearest(ev.clientX)].id));
    overlay.addEventListener("focus", () => show(cursor));
    overlay.addEventListener("blur", leave);
    overlay.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowLeft") show(Math.max(0, cursor - 1));
      else if (ev.key === "ArrowRight") show(Math.min(n - 1, cursor + 1));
      else if (ev.key === "Enter") openNode(nodes[cursor].id);
      else return;
      ev.preventDefault();
    });
    svg.append(cross, overlay);
    box.replaceChildren(svg);

    $("#timeline-legend").replaceChildren(
      ...OPERATORS.map((o) => legendItem(keySvg(marker(o, 7, 7, 5, `f-${o}`)), o)),
      legendItem(keySvg(sv("line", { class: "s-ink", x1: 0, x2: 14, y1: 7, y2: 7, "stroke-width": 2 })), "node the search builds on"),
      legendItem(keySvg(sv("rect", { class: "f-ink", opacity: 0.12, x: 0, y: 3, width: 14, height: 8 })), `noise band (±${st.noise_k} × std)`),
      ...STATUSES.filter((s) => s !== "ok").map((s) => legendItem(keySvg(sv("circle", { class: `f-${s}`, cx: 7, cy: 7, r: 6 })), `${GLYPH[s]} ${STATUS_WORD[s]}`)),
    );
  }

  // ---- gain over parent ----------------------------------------------------------------

  function renderGains() {
    const box = $("#gains");
    const legend = $("#gains-legend");
    const items = S.state.nodes.filter((d) => d.gain != null);
    if (!items.length) {
      box.replaceChildren(h("div", { class: "empty", text: "No node has a scored parent yet." }));
      legend.replaceChildren();
      return;
    }
    const W = Math.max(box.clientWidth, 260);
    const H = 250;
    const m = { l: 50, r: 8, t: 10, b: 28 };
    const n = items.length;
    const slot = (W - m.l - m.r) / n;
    const bw = Math.min(24, Math.max(slot - 4, 3));
    const ext = Math.max(...items.map((d) => Math.max(Math.abs(d.gain), d.floor || 0))) * 1.2 || 0.01;
    const Y = (v) => m.t + ((ext - v) / (2 * ext)) * (H - m.t - m.b);
    const { ticks, decimals } = niceTicks(-ext, ext, 4);

    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Gain of each node over its parent" });
    for (const v of ticks) {
      svg.append(
        sv("line", { class: Math.abs(v) < 1e-12 ? "axis-line" : "grid-line", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }),
        sv("text", { class: "num", x: m.l - 8, y: Y(v), "text-anchor": "end", "dominant-baseline": "central", text: (v > 0 ? "+" : "") + v.toFixed(decimals) }),
      );
    }
    const every = Math.max(1, Math.ceil(40 / slot));
    items.forEach((d, i) => {
      const x = m.l + i * slot + (slot - bw) / 2;
      const g = sv("g", { opacity: matches(d) ? 1 : 0.18 });
      g.append(sv("path", { class: `f-${d.verdict}`, d: barPath(x, bw, Y(0), Y(d.gain), 4) }));
      for (const sign of [1, -1]) {
        g.append(sv("line", { class: "s-ink", x1: x - 2, x2: x + bw + 2, y1: Y(sign * d.floor), y2: Y(sign * d.floor), "stroke-width": 1, opacity: 0.35 }));
      }
      if (i % every === 0) g.append(sv("text", { class: "num", x: x + bw / 2, y: H - 8, "text-anchor": "middle", text: d.id }));
      if (i === n - 1) {
        g.append(sv("text", { class: "label-ink num", x: x + bw / 2, y: d.gain >= 0 ? Y(d.gain) - 6 : Y(d.gain) + 14, "text-anchor": "middle", text: fgain(d.gain) }));
      }
      const hit = sv("rect", { fill: "transparent", x: m.l + i * slot, y: 0, width: Math.max(slot, 6), height: H - m.b, tabindex: 0, role: "button", "aria-label": `${d.id}: ${VERDICT_WORD[d.verdict]}, gain ${fgain(d.gain)}` });
      wireNode(hit, d);
      g.append(hit);
      svg.append(g);
    });
    box.replaceChildren(svg);
    legend.replaceChildren(
      ...["better", "noise", "worse"].map((v) => legendItem(keySvg(sv("rect", { class: `f-${v}`, x: 1, y: 1, width: 12, height: 12, rx: 2 })), VERDICT_WORD[v])),
      legendItem(keySvg(sv("line", { class: "s-ink", x1: 0, x2: 14, y1: 7, y2: 7, "stroke-width": 1, opacity: 0.5 })), "noise floor of that pair"),
    );
  }

  // ---- how far runs got, and operator results ------------------------------------------

  function renderFunnel() {
    const st = S.state;
    const box = $("#funnel");
    const counts = st.stages;
    const total = STAGES.reduce((a, s) => a + counts[s], 0);
    if (!total) {
      box.replaceChildren(h("div", { class: "empty", text: "No run has reported a stage yet." }));
    } else {
      const W = Math.max(box.clientWidth, 260);
      const rowH = 30;
      const labelW = 128;
      const valueW = 78;
      const H = STAGES.length * rowH + 4;
      const max = Math.max(...STAGES.map((s) => counts[s]), 1);
      const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Runs by the last step they finished" });
      STAGES.forEach((stage, i) => {
        const y = i * rowH + 4;
        const len = (counts[stage] / max) * (W - labelW - valueW);
        svg.append(sv("text", { class: "label-ink", x: 0, y: y + 8, "dominant-baseline": "central", text: STAGE_TEXT[stage] }));
        const g = sv("g");
        if (counts[stage]) g.append(sv("path", { class: `f-ord-${i + 1}`, d: barPathH(labelW, y, labelW + len, 16, 4) }));
        g.append(
          sv("text", { class: "num label-ink", x: labelW + len + 8, y: y + 8, "dominant-baseline": "central", text: `${counts[stage]} (${Math.round((100 * counts[stage]) / total)}%)` }),
          sv("rect", { fill: "transparent", x: 0, y: y - 4, width: W, height: rowH }),
        );
        g.addEventListener("pointermove", (ev) =>
          showTip([h("div", { class: "t", text: STAGE_TEXT[stage] }), h("div", { class: "key", text: `${counts[stage]} of ${total} runs stopped here or finished here` })], { x: ev.clientX, y: ev.clientY }),
        );
        g.addEventListener("pointerleave", hideTip);
        svg.append(g);
      });
      box.replaceChildren(svg);
    }

    const rows = Object.entries(st.operators).map(([name, o]) =>
      h(
        "tr",
        {},
        h("td", {}, h("span", { class: "inline" }, h("span", { class: `mark bg-${op(name)}` }), name)),
        h("td", { class: "n", text: String(o.n) }),
        h("td", { class: "n", text: o.n ? `${Math.round((100 * o.ok) / o.n)}%` : "–" }),
        h("td", { class: "n", text: `${o.better}/${o.noise}/${o.worse}` }),
      ),
    );
    $("#operators").replaceChildren(
      rows.length
        ? h(
            "table",
            {},
            h("thead", {}, h("tr", {}, h("th", { text: "Operator" }), h("th", { class: "n", text: "Nodes" }), h("th", { class: "n", text: "Ran ok" }), h("th", { class: "n", title: "beats noise / within noise / worse", text: "B/N/W" }))),
            h("tbody", {}, ...rows),
          )
        : h("div", { class: "empty", text: "No node yet." }),
    );
  }

  // ---- solution tree -------------------------------------------------------------------

  function renderTree() {
    const st = S.state;
    const box = $("#tree");
    const nodes = st.nodes;
    $("#tree-legend").replaceChildren(
      ...STATUSES.map((s) => legendItem(keySvg(sv("circle", { class: `f-${s}`, cx: 7, cy: 7, r: 6 })), `${GLYPH[s]} ${s}`)),
      legendItem(keySvg(sv("circle", { class: "ring s-ink", cx: 7, cy: 7, r: 5, fill: "none", "stroke-width": 2 })), "incumbent"),
    );
    if (!nodes.length) {
      box.replaceChildren(h("div", { class: "empty", text: "No node yet." }));
      return;
    }
    const byId = new Map(nodes.map((d) => [d.id, d]));
    const kids = new Map();
    const roots = [];
    for (const d of nodes) {
      if (d.parent && byId.has(d.parent)) {
        if (!kids.has(d.parent)) kids.set(d.parent, []);
        kids.get(d.parent).push(d);
      } else roots.push(d);
    }
    const pos = new Map();
    let row = 0;
    const visit = (d, depth) => {
      pos.set(d.id, { depth, row: row++ });
      (kids.get(d.id) || []).forEach((c) => visit(c, depth + 1));
    };
    roots.forEach((r) => visit(r, 0));

    const colW = 184;
    const rowH = 38;
    const pad = 16;
    const r = 10;
    const maxDepth = Math.max(...[...pos.values()].map((p) => p.depth));
    const W = pad * 2 + (maxDepth + 1) * colW;
    const H = pad * 2 + row * rowH;
    const cx = (p) => pad + r + 4 + p.depth * colW;
    const cy = (p) => pad + rowH / 2 + p.row * rowH;
    const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Solution tree" });
    for (const d of nodes) {
      if (!(d.parent && pos.has(d.parent))) continue;
      const a = pos.get(d.parent);
      const b = pos.get(d.id);
      const x1 = cx(a) + r;
      const y1 = cy(a);
      const x2 = cx(b) - r;
      const y2 = cy(b);
      svg.append(sv("path", { class: "edge", d: `M${x1},${y1}C${x1 + 30},${y1} ${x2 - 30},${y2} ${x2},${y2}` }));
    }
    const incId = st.summary.incumbent ? st.summary.incumbent.id : null;
    for (const d of nodes) {
      const p = pos.get(d.id);
      const x = cx(p);
      const y = cy(p);
      const second = d.cv != null ? f4(d.cv) : d.status === "blocked" ? `waits on ${d.dep_request || "a package"}` : `reached ${d.stage || "?"}`;
      const g = sv(
        "g",
        { class: "node" + (d.id === S.selected ? " selected" : ""), tabindex: 0, role: "button", "aria-label": `${d.id}, ${d.operator}, ${d.status}`, opacity: matches(d) ? 1 : 0.25 },
        sv("rect", { class: "hit", x: x - r - 4, y: y - rowH / 2 + 2, width: colW - 12, height: rowH - 4, rx: 6 }),
        sv("circle", { class: `f-${d.status}`, cx: x, cy: y, r }),
        sv("text", { class: "glyph-text" + (d.status === "blocked" ? " dark" : ""), x, y, text: GLYPH[d.status] }),
        d.id === incId ? sv("circle", { class: "ring", cx: x, cy: y, r: r + 4 }) : null,
        sv("text", { x: x + r + 8, y: y - 3 }, sv("tspan", { class: "id", text: d.id }), sv("tspan", { dx: 6, text: d.operator })),
        sv("text", { class: "num", x: x + r + 8, y: y + 11, text: second }),
      );
      wireNode(g, d);
      svg.append(g);
    }
    box.replaceChildren(svg);

    if (S.follow() && nodes.length > S.seenNodes) {
      const last = pos.get(nodes[nodes.length - 1].id);
      box.scrollTop = Math.max(0, cy(last) - box.clientHeight + 60);
      box.scrollLeft = Math.max(0, cx(last) - box.clientWidth + 200);
    }
  }

  // ---- node table ----------------------------------------------------------------------

  const COLUMNS = [
    { key: "seq", label: "#", num: true },
    { key: "id", label: "Node" },
    { key: "operator", label: "Operator" },
    { key: "parent", label: "Parent" },
    { key: "status", label: "Status" },
    { key: "stage", label: "Reached" },
    { key: "cv", label: "Score", num: true },
    { key: "std", label: "± std", num: true },
    { key: "gain", label: "Gain vs parent", num: true },
    { key: "duration_s", label: "Run time", num: true },
    { key: "created_at", label: "Time" },
    { key: "result", label: "Result" },
  ];

  function renderTable() {
    const nodes = S.state.nodes;
    const rows = nodes.map((d, i) => ({ d, seq: i + 1 })).filter((r) => matches(r.d));
    const key = S.sort.key;
    const val = (r) => (key === "seq" ? r.seq : key === "result" ? r.d.error || r.d.summary : r.d[key]);
    rows.sort((a, b) => {
      const x = val(a);
      const y = val(b);
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * S.sort.dir;
    });
    $("#table-sub").textContent = `${rows.length} of ${nodes.length} nodes. Click a row for its code, logs and result.`;

    const head = h(
      "tr",
      {},
      ...COLUMNS.map((c) =>
        h(
          "th",
          { class: c.num ? "n" : "", "aria-sort": S.sort.key === c.key ? (S.sort.dir > 0 ? "ascending" : "descending") : "none" },
          h("button", {
            type: "button",
            text: c.label + (S.sort.key === c.key ? (S.sort.dir > 0 ? " ↑" : " ↓") : ""),
            onclick: () => {
              S.sort = { key: c.key, dir: S.sort.key === c.key ? -S.sort.dir : c.key === "id" || c.key === "operator" || c.key === "status" ? 1 : -1 };
              renderTable();
            },
          }),
        ),
      ),
    );
    const body = rows.slice(0, 300).map(({ d, seq }) => {
      const tr = h(
        "tr",
        { tabindex: 0, class: d.id === S.selected ? "selected" : "", "aria-label": `${d.id}, ${d.status}` },
        h("td", { class: "n", text: String(seq) }),
        h("td", { text: d.id }),
        h("td", {}, h("span", { class: "inline" }, h("span", { class: `mark bg-${op(d.operator)}` }), d.operator)),
        h("td", { text: d.parent || "–" }),
        h("td", {}, statusChip(d.status)),
        h("td", { text: d.stage ? STAGE_TEXT[d.stage] || d.stage : "–" }),
        h("td", { class: "n", text: f4(d.cv) }),
        h("td", { class: "n", text: f4(d.std) }),
        h("td", { class: "n" }, d.gain == null ? "–" : h("span", { class: "inline" }, h("span", { class: `mark bg-${d.verdict}` }), `${fgain(d.gain)} ${d.verdict}`)),
        h("td", { class: "n", text: dur1(d.duration_s) }),
        h("td", { text: clock(d.created_at) }),
        h("td", { class: "wrap", text: d.error || d.summary || "" }),
      );
      tr.addEventListener("click", () => openNode(d.id));
      tr.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") openNode(d.id);
      });
      return tr;
    });
    $("#nodes-table").replaceChildren(h("thead", {}, head), h("tbody", {}, ...body));
  }

  // ---- activity feed -------------------------------------------------------------------

  function feedItems(items) {
    if (!items.length) return [h("li", { class: "muted", text: "Nothing yet." })];
    return items.map((e) => {
      const found = /: (ok|buggy|dead|blocked)\b/.exec(e.text);
      const li = h(
        "li",
        e.node ? { tabindex: 0, role: "button", class: "clickable" } : {},
        h("span", { class: "time", text: clock(e.ts) }),
        found ? h("span", { class: `glyph ${found[1]}`, "aria-hidden": "true", text: GLYPH[found[1]] }) : h("span"),
        h("span", { class: "text", text: e.text }),
      );
      if (e.node) {
        li.addEventListener("click", () => openNode(e.node));
        li.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") openNode(e.node);
        });
      }
      return li;
    });
  }

  function renderFeed() {
    $("#feed").replaceChildren(...feedItems(S.events.slice(-200).reverse()));
    renderLiveFeed();
  }

  function renderLiveFeed() {
    $("#live-feed").replaceChildren(...feedItems(S.events.slice(-10).reverse()));
  }

  // ---- models --------------------------------------------------------------------------

  function spark(values) {
    if (values.length < 2) return h("div", { class: "hint", text: "Latency chart appears after two calls." });
    const W = 200;
    const H = 40;
    const max = Math.max(...values, 0.001);
    const points = values.map((v, i) => `${((i / (values.length - 1)) * W).toFixed(1)},${(H - 4 - (v / max) * (H - 8)).toFixed(1)}`).join(" ");
    return sv(
      "svg",
      { class: "spark", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img", "aria-label": `Seconds per call, last ${values.length} calls, up to ${max.toFixed(0)} s` },
      sv("line", { class: "base", x1: 0, x2: W, y1: H - 1, y2: H - 1 }),
      sv("polyline", { points }),
    );
  }

  function renderLlm() {
    const st = S.state;
    const params = st.run.params || {};
    $("#llm-sub").textContent = "Seconds per call, newest on the right.";
    const roles = Object.entries(st.llm);
    const box = $("#llm");
    if (!roles.length) {
      box.replaceChildren(h("div", { class: "empty", text: "No model call yet." }));
      return;
    }
    const flying = st.run.llm;
    const mini = (k, v) => h("div", {}, h("div", { class: "k", text: k }), h("div", { class: "v num", text: v }));
    const sections = roles.map(([role, s]) =>
      h(
        "div",
        { class: "role" },
        h(
          "h3",
          {},
          role,
          h("span", { class: "muted", text: params[`${role}_model`] || "" }),
          flying && flying.role === role
            ? h("span", { class: "inline" }, h("span", { class: "glyph running pulse", "aria-hidden": "true", text: "●" }), `waiting ${dur(nowSec() - flying.since)}`)
            : null,
        ),
        h("div", { class: "mini-tiles" }, mini("calls", String(s.calls)), mini("mean", dur1(s.mean_s)), mini("p95", dur1(s.p95_s)), mini("tokens out", compact(s.tokens_out)), mini("tok/s", s.tokens_per_s ? s.tokens_per_s.toFixed(1) : "–")),
        spark(s.latency),
      ),
    );

    const recent = S.calls.slice(-8).reverse();
    const table = recent.length
      ? h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            {},
            h("thead", {}, h("tr", {}, h("th", { text: "Time" }), h("th", { text: "Role" }), h("th", { class: "n", text: "Sec" }), h("th", { class: "n", text: "Out" }), h("th", { text: "End" }))),
            h(
              "tbody",
              {},
              ...recent.map((c) => {
                const tr = h("tr", { tabindex: 0, "aria-label": `${c.role} call ${c.i}` }, h("td", { text: clock(c.ts) }), h("td", { text: c.role }), h("td", { class: "n", text: dur1(c.seconds) }), h("td", { class: "n", text: compact(c.completion_tokens) }), h("td", { text: c.finish || "" }));
                tr.addEventListener("click", () => openCall(c.i));
                tr.addEventListener("keydown", (ev) => {
                  if (ev.key === "Enter") openCall(c.i);
                });
                return tr;
              }),
            ),
          ),
          h("p", { class: "hint", text: "Click a call to read its prompt and reply." }),
        )
      : null;
    box.replaceChildren(...[...sections, table].filter(Boolean));
  }

  // A prompt and, when there is one, the reply to it, in the modal.
  function showTranscript(title, meta, messages, reply) {
    const modal = $("#modal");
    modal.replaceChildren(
      h(
        "div",
        { class: "box", role: "dialog", "aria-modal": "true", "aria-label": title },
        h("header", { class: "inline" }, h("h2", { text: title }), h("span", { class: "muted", text: meta }), h("button", { class: "btn", type: "button", onclick: closeModal, text: "Close (Esc)" })),
        ...messages.flatMap((m) => [h("h3", { text: m.role }), h("pre", { class: "block", text: m.content })]),
        reply == null ? null : [h("h3", { text: "reply" }), h("pre", { class: "block", text: reply })],
      ),
    );
    modal.hidden = false;
  }

  async function openCall(index) {
    try {
      const { data } = await get(`/api/call/${index}`);
      showTranscript(
        `${data.role} call ${data.i}`,
        `${data.model} · ${dur1(data.seconds)} · ${compact(data.prompt_tokens)} in, ${compact(data.completion_tokens)} out · ${data.finish || ""}`,
        data.messages,
        data.response,
      );
    } catch (err) {
      S.error = err.message;
    }
  }
  const closeModal = () => {
    $("#modal").hidden = true;
  };

  // ---- package requests, do-not-repeat list ---------------------------------------------

  function renderSide() {
    const st = S.state;
    const slug = st.competition.slug;
    $("#deps").replaceChildren(
      ...(st.deps.length
        ? st.deps.map((d) =>
            h(
              "li",
              {},
              h("span", { class: "inline" }, h("span", { class: `glyph ${DEP_CLASS[d.status]}`, "aria-hidden": "true", text: DEP_GLYPH[d.status] }), h("strong", { text: `${d.package}${d.constraint}` }), ` · ${d.status} · ${d.node_id}`),
              d.flags.length ? h("div", { class: "flags", text: d.flags.join(" · ") }) : null,
              d.status === "pending" ? h("div", { class: "flags mono", text: `kagent deps approve -c ${slug} ${d.id}` }) : null,
              d.decision_reason ? h("div", { class: "flags", text: d.decision_reason }) : null,
            ),
          )
        : [h("li", { class: "muted", text: "None." })]),
    );
    $("#blocklist").replaceChildren(...(st.blocklist.length ? st.blocklist.map((b) => h("li", { text: b })) : [h("li", { class: "muted", text: "Nothing has failed or stalled yet." })]));
    const p = st.run.params || {};
    const bits = [`noise floor ${st.noise_k} × std`];
    if (p.num_drafts != null) bits.push(`${p.num_drafts} drafts`);
    if (p.max_debug != null) bits.push(`debug up to ${p.max_debug}`);
    if (p.max_nodes != null) bits.push(`stop after ${p.max_nodes} nodes`);
    if (p.planner_thinking) bits.push(`planner thinking: ${p.planner_thinking}`);
    $("#params").textContent = bits.join(" · ");
  }

  // ---- node drawer ---------------------------------------------------------------------

  async function openNode(id) {
    S.selected = id;
    hideTip();
    if (S.state) {
      renderTree();
      renderTable();
    }
    await loadDrawer();
  }

  function closeDrawer() {
    S.selected = null;
    $("#drawer").hidden = true;
    if (S.state) {
      renderTree();
      renderTable();
    }
  }

  async function loadDrawer() {
    const id = S.selected;
    if (!id) return;
    try {
      const { data } = await get(`/api/node/${id}`);
      if (S.selected === id) renderDrawer(data);
    } catch (err) {
      if (S.selected === id) {
        $("#drawer").hidden = false;
        $("#drawer").replaceChildren(h("header", {}, h("h2", { text: id }), h("span", { class: "spacer" }), h("button", { class: "btn", type: "button", onclick: closeDrawer, text: "Close" })), h("p", { class: "empty", text: err.message }));
      }
    }
  }

  function renderDrawer(d) {
    const drawer = $("#drawer");
    const link = (id) => h("button", { class: "link", type: "button", text: id, onclick: () => openNode(id) });
    const meta = (k, v) => h("div", {}, h("div", { class: "k", text: k }), h("div", { class: "v" }, v));
    const run = d.run || {};
    const section = (title, ...kids) => h("section", {}, h("h3", { text: title }), ...kids);
    const children = d.children.length ? d.children.flatMap((c, i) => (i ? [", ", link(c)] : [link(c)])) : ["none"];

    const tabs = ["stdout", "stderr"].map((t) =>
      h("button", { class: "chip", type: "button", "aria-pressed": String(S.logTab === t), text: t, onclick: () => { S.logTab = t; renderDrawer(d); } }),
    );
    const gutter = d.code ? Array.from({ length: d.code.split("\n").length }, (_, i) => i + 1).join("\n") : "";
    const dep = S.state.deps.find((x) => x.id === d.dep_request);

    const parts = [
      h("header", {}, h("h2", { text: d.id }), statusChip(d.status), h("span", { class: "muted", text: d.operator }), h("span", { class: "spacer" }), h("button", { class: "btn", type: "button", onclick: closeDrawer, text: "Close (Esc)" })),
      h(
        "div",
        { class: "meta" },
        meta("Parent", d.parent ? link(d.parent) : "none, a new draft"),
        meta("Children", children),
        meta("Score", d.cv != null ? `${f4(d.cv)} ± ${f4(d.std)}` : "–"),
        meta("Gain vs parent", d.gain != null ? `${fgain(d.gain)}, ${VERDICT_WORD[d.verdict]} (noise ±${f4(d.floor)})` : "–"),
        meta("Last step finished", d.stage ? STAGE_TEXT[d.stage] || d.stage : "–"),
        meta("Run time", dur1(run.duration_s)),
        meta("Exit code", run.exit_code != null ? String(run.exit_code) : "–"),
        meta("Created", d.created_at ? new Date(d.created_at).toLocaleString() : "–"),
        meta("Run", d.run_id || "–"),
        meta("Held-out score (display only)", run.heldout_score != null ? f4(run.heldout_score) : "none"),
      ),
      run.integrity_changes && run.integrity_changes.length ? section("Integrity", h("pre", { class: "block", text: run.integrity_changes.join("\n") })) : null,
      d.full_summary ? section("What it did", h("p", { text: d.full_summary })) : null,
      d.full_error ? section("Error", h("pre", { class: "block", text: d.full_error })) : null,
      dep ? section("Package request", h("p", { text: `${dep.package}${dep.constraint}: ${dep.status}${dep.flags.length ? " · " + dep.flags.join(" · ") : ""}` })) : null,
      d.plan.length ? section("Plan", h("p", { text: d.plan.join(" ") })) : null,
      d.code
        ? section(
            "Code",
            h("div", { class: "tabs" }, h("button", { class: "btn", type: "button", text: "Copy", onclick: () => navigator.clipboard && navigator.clipboard.writeText(d.code) })),
            h("div", { class: "code" }, h("pre", { class: "gutter", text: gutter }), h("pre", { text: d.code })),
          )
        : null,
      section("Logs", h("div", { class: "tabs" }, ...tabs), h("pre", { class: "block", text: (S.logTab === "stdout" ? d.stdout : d.stderr) || "(empty)" })),
    ];
    drawer.replaceChildren(...parts.filter(Boolean)); // optional sections are null when empty
    drawer.hidden = false;
  }

  // ---- run controls --------------------------------------------------------------------

  const metaContent = (name) => {
    const el = document.querySelector(`meta[name="${name}"]`);
    return el ? el.content : "";
  };
  const TOKEN = metaContent("kagent-token");
  const RUN_ON = metaContent("kagent-run") === "on" && TOKEN !== "";

  async function post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Kagent-Token": TOKEN },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(res.status === 403 ? "This page is out of date. Reload it." : data.error || `HTTP ${res.status}`);
      err.fields = data.fields || [];
      throw err;
    }
    return data;
  }

  let toastTimer = 0;
  function toast(text) {
    const el = $("#toast");
    el.textContent = text;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      el.hidden = true;
    }, 7000);
  }

  function renderRunButtons() {
    const run = S.state ? S.state.run : null;
    const busy = Boolean(run && (run.state === "running" || run.state === "starting"));
    if (!busy) S.stopping = false;
    const start = $("#run-btn");
    const stop = $("#stop-btn");
    start.hidden = !RUN_ON;
    start.disabled = busy;
    start.title = busy ? "A search is already running" : "Choose settings and start a search";
    stop.hidden = !(RUN_ON && busy);
    stop.textContent = S.stopping ? "Force stop" : "Stop";
  }

  async function stopSearch() {
    const force = Boolean(S.stopping);
    const ask = force
      ? "Force-stop the search now? The step in progress is lost."
      : "Stop the search? The step in progress is abandoned. The tree is kept, and a later run resumes from it.";
    if (!window.confirm(ask)) return;
    try {
      await post("/api/search/stop", { force });
      S.stopping = true;
      toast(force ? "Force-stopping…" : "Stopping. If it does not end soon, press Force stop.");
      pollNow();
    } catch (err) {
      toast(err.message);
    }
  }

  async function renderConsole() {
    const { data } = await get("/api/runlog");
    const box = $("#console");
    const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.textContent = data.text || "Nothing yet. This shows the output of searches started from this page.";
    S.consoleSize = data.size;
    if (atEnd) box.scrollTop = box.scrollHeight;
  }

  const NUMBER_KEYS = [
    "budget_hours", "max_nodes", "num_drafts", "max_debug", "noise_k", "step_timeout_s", "sandbox_cpus", "llm_timeout_s", "llm_max_tokens",
    "planner_temperature", "planner_top_p", "planner_top_k", "coder_temperature", "coder_top_p", "coder_top_k",
  ];
  const TEXT_KEYS = ["planner_base_url", "planner_model", "coder_base_url", "coder_model", "sandbox_memory", "planner_thinking"];
  const THINKING_CHOICES = [
    ["xhigh", "Full effort (slowest)"],
    ["medium", "Medium effort"],
    ["low", "Low effort (fastest)"],
    ["off", "Off"],
    ["default", "The model's own default (full effort)"],
  ];

  async function openRunForm() {
    let info;
    try {
      info = (await get("/api/launch")).data;
    } catch (err) {
      toast(err.message);
      return;
    }
    if (!info.enabled) return;
    const slug = S.state ? S.state.competition.slug : "";
    const defaults = info.defaults;
    const inputs = {};
    let ready = false;
    let seq = 0;
    let timer = 0;

    const field = (key, label, hint, attrs) => {
      const id = `f-${key}`;
      const input = h("input", { id, autocomplete: "off", spellcheck: "false", value: defaults[key] == null ? "" : String(defaults[key]), ...attrs });
      inputs[key] = input;
      input.addEventListener("input", changed);
      return h("div", { class: "field" }, h("label", { for: id, text: label }), input, hint ? h("div", { class: "hint", text: hint }) : null);
    };
    const num = (label, key, hint, min, max, step) => field(key, label, hint, { type: "number", min, max, step });
    const text = (label, key, hint) => field(key, label, hint, { type: "text" });
    const choice = (label, key, hint, options) => {
      const id = `f-${key}`;
      const select = h("select", { id }, ...options.map(([value, name]) => h("option", { value, text: name })));
      select.value = String(defaults[key]);
      inputs[key] = select;
      select.addEventListener("change", changed);
      return h("div", { class: "field" }, h("label", { for: id, text: label }), select, hint ? h("div", { class: "hint", text: hint }) : null);
    };

    const collect = () => {
      const body = {};
      for (const key of NUMBER_KEYS) {
        const raw = inputs[key].value.trim();
        body[key] = raw === "" ? null : Number(raw);
      }
      for (const key of TEXT_KEYS) body[key] = inputs[key].value.trim();
      return body;
    };

    const preview = h("pre", { class: "cmd" });
    const updatePreview = () => {
      const v = collect();
      const parts = ["kagent search", `-c ${slug}`, `--budget-hours ${v.budget_hours}`, `--num-drafts ${v.num_drafts}`, `--max-debug ${v.max_debug}`, `--noise-k ${v.noise_k}`];
      if (v.max_nodes != null) parts.push(`--max-nodes ${v.max_nodes}`);
      preview.textContent =
        parts.join(" ") +
        `\nplanner ${v.planner_model} at ${v.planner_base_url} (thinking ${v.planner_thinking}, temperature ${v.planner_temperature})` +
        `\ncoder   ${v.coder_model} at ${v.coder_base_url} (temperature ${v.coder_temperature})`;
    };

    const checks = h("ul", { class: "checks" });
    const problem = h("div", { class: "form-error", hidden: true });
    const showProblem = (err) => {
      const lines = err.fields && err.fields.length ? err.fields.map((f) => `${f.field}: ${f.message}`) : [err.message];
      problem.textContent = lines.join("\n");
      problem.hidden = false;
    };

    const startBtn = h("button", { class: "btn primary", type: "button", text: "Start search", disabled: true, onclick: submit });

    async function preflight() {
      const mine = ++seq;
      problem.hidden = true;
      startBtn.disabled = true;
      checks.replaceChildren(h("li", { class: "muted", text: "Checking Docker, images and the model servers…" }));
      try {
        const result = await post("/api/preflight", collect());
        if (mine !== seq) return;
        checks.replaceChildren(
          ...result.checks.map((c) =>
            h(
              "li",
              {},
              h("span", { class: `glyph ${c.status === "ok" ? "ok" : c.status === "warn" ? "blocked" : "dead"}`, "aria-hidden": "true", text: c.status === "ok" ? "✓" : c.status === "warn" ? "!" : "✕" }),
              h("strong", { text: c.label }),
              h("span", { class: "muted", text: c.message }),
            ),
          ),
        );
        ready = result.ready;
        startBtn.disabled = !ready;
      } catch (err) {
        if (mine !== seq) return;
        checks.replaceChildren();
        showProblem(err);
      }
    }

    function changed() {
      updatePreview();
      startBtn.disabled = true;
      clearTimeout(timer);
      timer = setTimeout(preflight, 700);
    }

    async function submit() {
      startBtn.disabled = true;
      problem.hidden = true;
      try {
        await post("/api/search/start", collect());
        closeModal();
        toast("Starting the search. This page follows it from here.");
        S.stopping = false;
        pollNow();
      } catch (err) {
        showProblem(err);
        startBtn.disabled = !ready;
      }
    }

    const presets = h(
      "div",
      { class: "group" },
      ...[0.5, 1, 2, 6].map((hours) =>
        h("button", { class: "chip", type: "button", text: `${hours} h`, onclick: () => { inputs.budget_hours.value = String(hours); changed(); } }),
      ),
    );

    const modal = $("#modal");
    modal.replaceChildren(
      h(
        "div",
        { class: "box form", role: "dialog", "aria-modal": "true", "aria-label": "Start a search" },
        h("header", { class: "inline" }, h("h2", { text: "Start a search" }), h("span", { class: "muted", text: slug })),
        h("p", { class: "hint", text: "The models write Python that runs in Docker with no network. Settings are remembered for next time." }),
        h("fieldset", {}, h("legend", { text: "How long and how wide" }),
          h("div", { class: "fields" },
            num("Time budget (hours)", "budget_hours", "The search stops when this is used up. Finished nodes are kept.", 0.05, 48, 0.05),
            num("Stop after this many new nodes", "max_nodes", "Leave empty for no limit.", 1, 1000, 1),
            num("Independent drafts first", "num_drafts", "Different starting approaches before improving.", 1, 10, 1),
            num("Debug attempts per failing node", "max_debug", "After this many, the node is marked dead.", 0, 10, 1),
            num("Noise floor (× fold std)", "noise_k", "A node must beat the best by this much to be built on.", 0, 5, 0.1),
          ),
          presets,
        ),
        h("fieldset", {}, h("legend", { text: "Models" }),
          h("div", { class: "fields" },
            text("Planner server URL", "planner_base_url", "OpenAI-compatible, ending in /v1"),
            text("Planner model", "planner_model"),
            text("Coder server URL", "coder_base_url", "OpenAI-compatible, ending in /v1"),
            text("Coder model", "coder_model"),
          ),
        ),
        h("fieldset", {}, h("legend", { text: "Sampling and thinking" }),
          h("div", { class: "fields" },
            choice("Planner thinking", "planner_thinking", "How long the planner may reason before it answers. The coder does not think.", THINKING_CHOICES),
            num("Planner temperature", "planner_temperature", "Values from the model card.", 0, 2, 0.05),
            num("Planner top-p", "planner_top_p", "", 0.05, 1, 0.05),
            num("Planner top-k", "planner_top_k", "", 0, 1000, 1),
            num("Coder temperature", "coder_temperature", "Lower repeats itself more; the card says 1.0.", 0, 2, 0.05),
            num("Coder top-p", "coder_top_p", "", 0.05, 1, 0.05),
            num("Coder top-k", "coder_top_k", "", 0, 1000, 1),
          ),
        ),
        h("fieldset", {}, h("legend", { text: "Sandbox for each run" }),
          h("div", { class: "fields" },
            num("Time limit per run (seconds)", "step_timeout_s", "", 30, 7200, 30),
            num("CPUs", "sandbox_cpus", "", 0.5, 64, 0.5),
            text("Memory", "sandbox_memory", "For example 8g or 512m"),
          ),
        ),
        h("details", {}, h("summary", { text: "Advanced" }),
          h("div", { class: "fields" },
            num("Model call timeout (seconds)", "llm_timeout_s", "", 30, 7200, 30),
            num("Max tokens per reply", "llm_max_tokens", "", 256, 65536, 256),
          ),
        ),
        h("h3", { text: "Before it starts" }),
        checks,
        problem,
        preview,
        h("div", { class: "actions" },
          h("button", { class: "btn", type: "button", text: "Recheck", onclick: preflight }),
          h("button", { class: "btn", type: "button", text: "Cancel", onclick: closeModal }),
          startBtn,
        ),
      ),
    );
    modal.hidden = false;
    updatePreview();
    preflight();
    inputs.budget_hours.focus();
  }

  // ---- live view: the models' replies as they are written ------------------------------

  const ROLES = ["planner", "coder"];
  const ROLE_TITLE = { planner: "Planner", coder: "Coder" };
  const ROLE_ABOUT = {
    planner: "Decides what to try next and summarizes each run.",
    coder: "Writes and repairs the Python script.",
  };
  const STEP_LABEL = { choose: "Choose", plan: "Plan", code: "Code", run: "Run", grade: "Grade" };
  const STEP_PHASES = new Set(["selecting", "writing", "running", "evaluating"]);

  const PY_KEYWORDS = new Set(
    "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "),
  );
  const PY_BUILTINS = new Set("print len range int float str list dict set tuple bool enumerate zip map sum min max abs sorted open isinstance super self cls".split(" "));
  const PY_WORD = /[A-Za-z_][A-Za-z0-9_]*/y;
  const PY_NUMBER = /\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?/y;
  const PY_PLAIN = /[^A-Za-z0-9_#"'@]+/y;
  const PY_DECORATOR = /@[A-Za-z_][\w.]*/y;
  const PY_PREFIX = /^(?:[rRbBuUfF]|[rR][bBfF]|[bBfF][rR])$/;
  const FENCE = /^\s*```(\w*)\s*$/;

  const P = {}; // one pane per role
  S.live = null;
  S.liveEtag = null;
  S.view = null;
  S.codeCache = new Map(); // node id -> script, "" if unavailable, null while loading

  const busy = () => {
    const run = S.state && S.state.run;
    return Boolean(run && (run.state === "running" || run.state === "starting"));
  };

  function matchAt(re, text, at) {
    re.lastIndex = at;
    const m = re.exec(text);
    return m ? m[0] : null;
  }

  // Colors one line of Python. `open` is a triple quote left unclosed by an earlier line.
  function pyTokens(line, open) {
    const out = [];
    const push = (cls, text) => {
      if (text) out.push([cls, text]);
    };
    let i = 0;
    let quote = open;
    if (quote) {
      const end = line.indexOf(quote);
      if (end < 0) return { tokens: [["str", line]], open: quote };
      push("str", line.slice(0, end + 3));
      i = end + 3;
      quote = "";
    }
    let afterDef = false;
    while (i < line.length) {
      const c = line[i];
      if (c === "#") {
        push("com", line.slice(i));
        break;
      }
      if (c === '"' || c === "'") {
        const triple = c.repeat(3);
        if (line.startsWith(triple, i)) {
          const end = line.indexOf(triple, i + 3);
          if (end < 0) {
            push("str", line.slice(i));
            quote = triple;
            break;
          }
          push("str", line.slice(i, end + 3));
          i = end + 3;
        } else {
          let j = i + 1;
          while (j < line.length && line[j] !== c) j += line[j] === "\\" ? 2 : 1;
          push("str", line.slice(i, j + 1));
          i = j + 1;
        }
      } else if (c >= "0" && c <= "9") {
        const num = matchAt(PY_NUMBER, line, i);
        push("num", num);
        i += num.length;
      } else if (/[A-Za-z_]/.test(c)) {
        const word = matchAt(PY_WORD, line, i);
        i += word.length;
        const next = line[i];
        if (PY_PREFIX.test(word) && (next === '"' || next === "'")) {
          push("str", word);
        } else if (afterDef) {
          push("fn", word);
          afterDef = false;
        } else if (PY_KEYWORDS.has(word)) {
          push("kw", word);
          afterDef = word === "def" || word === "class";
        } else {
          push(PY_BUILTINS.has(word) ? "bi" : "", word);
        }
      } else if (c === "@" && out.every((t) => !t[1].trim())) {
        const dec = matchAt(PY_DECORATOR, line, i) || "@";
        push("dec", dec);
        i += dec.length;
      } else {
        const plain = matchAt(PY_PLAIN, line, i) || c;
        push("", plain);
        i += plain.length;
      }
    }
    return { tokens: out, open: quote };
  }

  const tokenSpans = (text, open) =>
    pyTokens(text, open).tokens.map(([cls, t]) => (cls ? h("span", { class: `tk-${cls}`, text: t }) : t));

  // Splits a reply into display lines: prose, fenced code (colored as Python) and the fences.
  function parseLines(text) {
    const out = [];
    let fenced = false;
    let open = "";
    let no = 0;
    for (const line of text.split("\n")) {
      const fence = FENCE.exec(line);
      if (fence && !fenced) {
        fenced = true;
        open = "";
        no = 0;
        out.push({ kind: "fence", text: fence[1] || "code", open: "", no: 0 });
      } else if (fence) {
        fenced = false;
        out.push({ kind: "fence-end", text: "", open: "", no: 0 });
      } else if (fenced) {
        no += 1;
        out.push({ kind: "code", text: line, open, no });
        open = pyTokens(line, open).open;
      } else {
        out.push({ kind: "prose", text: line, open: "", no: 0 });
      }
    }
    return out;
  }

  function renderLine(el, m) {
    el.className = `ln ${m.kind}`;
    if (m.kind === "code") {
      el.replaceChildren(h("span", { class: "no", text: String(m.no) }), h("span", { class: "tx" }, ...tokenSpans(m.text, m.open)));
    } else if (m.kind === "fence-end") {
      el.replaceChildren();
    } else {
      el.replaceChildren(h("span", { class: "tx", text: m.text }));
    }
  }

  // Only the lines that changed are touched, so text can stream in without the page flickering.
  function paintLines(p, text, streaming) {
    const models = text ? parseLines(text) : [];
    const { els, keys } = p.lines;
    models.forEach((m, i) => {
      const key = `${m.kind}|${m.open}|${m.no}|${m.text}`;
      if (!els[i]) {
        els.push(h("div", {}));
        keys.push("");
        p.linesBox.append(els[i]);
      }
      if (keys[i] !== key) {
        keys[i] = key;
        renderLine(els[i], m);
      }
    });
    while (els.length > models.length) {
      els.pop().remove();
      keys.pop();
    }
    if (streaming) {
      const last = [...els].reverse().find((el) => !el.classList.contains("fence-end"));
      const tx = last ? last.querySelector(".tx") : null;
      (tx || p.linesBox).append(p.caret);
    } else {
      p.caret.remove();
    }
  }

  function resetLines(p) {
    p.lines.els.forEach((el) => el.remove());
    p.lines.els.length = 0;
    p.lines.keys.length = 0;
    p.caret.remove();
  }

  function extractCode(text) {
    const blocks = [];
    const re = /```(?:python|py)?[ \t]*\n([\s\S]*?)```/g;
    for (let m = re.exec(text); m; m = re.exec(text)) blocks.push(m[1]);
    if (!blocks.length) return "";
    return blocks.reduce((a, b) => (b.length > a.length ? b : a)).trim() + "\n";
  }

  const splitLines = (code) => code.replace(/\n$/, "").split("\n");

  // Line diff of two scripts: [{t: " " | "+" | "-", x: line}].
  function diffLines(a, b) {
    let head = 0;
    while (head < a.length && head < b.length && a[head] === b[head]) head += 1;
    let tail = 0;
    while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail += 1;
    const A = a.slice(head, a.length - tail);
    const B = b.slice(head, b.length - tail);
    const out = [];
    for (let k = 0; k < head; k++) out.push({ t: " ", x: a[k] });
    if (A.length * B.length > 4e6) {
      A.forEach((x) => out.push({ t: "-", x }));
      B.forEach((x) => out.push({ t: "+", x }));
    } else {
      const w = B.length + 1;
      const lcs = new Uint16Array((A.length + 1) * w);
      for (let i = A.length - 1; i >= 0; i--) {
        for (let j = B.length - 1; j >= 0; j--) {
          lcs[i * w + j] = A[i] === B[j] ? lcs[(i + 1) * w + j + 1] + 1 : Math.max(lcs[(i + 1) * w + j], lcs[i * w + j + 1]);
        }
      }
      let i = 0;
      let j = 0;
      while (i < A.length && j < B.length) {
        if (A[i] === B[j]) {
          out.push({ t: " ", x: A[i] });
          i += 1;
          j += 1;
        } else if (lcs[(i + 1) * w + j] >= lcs[i * w + j + 1]) {
          out.push({ t: "-", x: A[i] });
          i += 1;
        } else {
          out.push({ t: "+", x: B[j] });
          j += 1;
        }
      }
      for (; i < A.length; i++) out.push({ t: "-", x: A[i] });
      for (; j < B.length; j++) out.push({ t: "+", x: B[j] });
    }
    for (let k = a.length - tail; k < a.length; k++) out.push({ t: " ", x: a[k] });
    return out;
  }

  // The changes with three lines of context; long unchanged stretches fold into one line.
  function renderDiff(box, oldCode, newCode, parentId) {
    const rows = diffLines(splitLines(oldCode), splitLines(newCode));
    const keep = new Array(rows.length).fill(false);
    rows.forEach((r, i) => {
      if (r.t === " ") return;
      for (let k = Math.max(0, i - 3); k <= Math.min(rows.length - 1, i + 3); k++) keep[k] = true;
    });
    const added = rows.filter((r) => r.t === "+").length;
    const removed = rows.filter((r) => r.t === "-").length;
    const kids = [h("div", { class: "diff-sum", text: added || removed ? `Against ${parentId}: ${added} lines added, ${removed} removed.` : `Identical to ${parentId}.` })];
    let line = 0;
    let skipped = 0;
    const fold = () => {
      if (skipped) kids.push(h("div", { class: "ln gap", text: `… ${skipped} unchanged line${skipped === 1 ? "" : "s"}` }));
      skipped = 0;
    };
    rows.forEach((r, i) => {
      if (r.t !== "-") line += 1;
      if (!keep[i]) {
        skipped += 1;
        return;
      }
      fold();
      const kind = r.t === "+" ? "add" : r.t === "-" ? "del" : "ctx";
      kids.push(
        h(
          "div",
          { class: `ln diff ${kind}` },
          h("span", { class: "no", text: r.t === "-" ? "" : String(line) }),
          h("span", { class: "sign", text: r.t === "+" ? "+" : r.t === "-" ? "−" : "" }),
          h("span", { class: "tx" }, ...tokenSpans(r.x, "")),
        ),
      );
    });
    fold();
    box.replaceChildren(...kids);
  }

  async function loadParentCode(id) {
    if (S.codeCache.has(id)) return;
    S.codeCache.set(id, null);
    try {
      const { data } = await get(`/api/node/${id}`);
      S.codeCache.set(id, data.code || "");
    } catch (err) {
      S.codeCache.set(id, "");
    }
  }

  // What a pane shows: the call being written, or a finished call picked from the rail.
  function paneData(p) {
    if (p.replay) {
      const d = p.replay.data;
      return {
        id: `replay-${d.i}`, replay: true, index: d.i, ts: d.ts, model: d.model, purpose: "", node: null, operator: null,
        parent: null, thinking: "", answer: d.response || "", done: true, ok: true, chunks: d.completion_tokens || 0,
        started_at: null, first_at: null, updated_at: null, seconds: d.seconds,
      };
    }
    const rec = S.live && S.live.calls[p.role];
    return rec ? { ...rec, replay: false, seconds: null } : null;
  }

  function syncTools(p) {
    p.followBtn.setAttribute("aria-pressed", String(p.follow));
    p.wrapBtn.setAttribute("aria-pressed", String(p.wrap));
    p.diffBtn.setAttribute("aria-pressed", String(p.diff));
    p.root.classList.toggle("wrap", p.wrap);
    p.backBtn.hidden = !p.replay;
  }

  const scrollEnd = (p) => {
    p.body.scrollTop = p.body.scrollHeight;
  };

  function copyPane(p) {
    const d = paneData(p);
    if (!d) return;
    const text = (p.role === "coder" && extractCode(d.answer)) || d.answer;
    if (!navigator.clipboard) {
      toast("This browser does not allow copying from here.");
      return;
    }
    navigator.clipboard.writeText(text).then(
      () => toast("Copied."),
      () => toast("Copying was blocked by the browser."),
    );
  }

  async function openPrompt(p) {
    if (p.replay) {
      openCall(p.replay.i);
      return;
    }
    try {
      const { data } = await get(`/api/live/prompt/${p.role}`);
      showTranscript(`${ROLE_TITLE[p.role]} prompt`, `what the ${p.role} was sent for this call`, data.messages, null);
    } catch (err) {
      toast("No prompt has been recorded for this call yet.");
    }
  }

  function backToLive(p) {
    p.replay = null;
    renderPane(p.role);
    renderRail();
  }

  async function openReplay(index) {
    try {
      const { data } = await get(`/api/call/${index}`);
      const p = P[data.role];
      if (!p) {
        openCall(index);
        return;
      }
      p.replay = { i: index, data };
      renderPane(data.role);
      renderRail();
    } catch (err) {
      toast(err.message);
    }
  }

  function buildPane(role) {
    const p = {
      role,
      root: $(`#pane-${role}`),
      lines: { els: [], keys: [] },
      follow: true,
      wrap: role === "planner",
      diff: false,
      diffKey: "",
      replay: null,
      id: null,
      touched: false,
      caret: h("span", { class: "caret", "aria-hidden": "true" }),
    };
    P[role] = p;

    const tool = (label, onclick, toggle) => h("button", { class: "chip", type: "button", "aria-pressed": toggle ? "false" : null, text: label, onclick });
    p.model = h("span", { class: "muted" });
    p.badge = h("span", { class: "badge" });
    p.meta = h("div", { class: "pane-meta" });
    p.followBtn = tool(
      "Follow",
      () => {
        p.follow = !p.follow;
        if (p.follow) scrollEnd(p);
        syncTools(p);
      },
      true,
    );
    p.wrapBtn = tool(
      "Wrap lines",
      () => {
        p.wrap = !p.wrap;
        syncTools(p);
      },
      true,
    );
    p.diffBtn = tool(
      "Changes",
      () => {
        p.diff = !p.diff;
        p.diffKey = "";
        renderPane(role);
      },
      true,
    );
    p.copyBtn = tool("Copy", () => copyPane(p), false);
    p.promptBtn = tool("Prompt", () => openPrompt(p), false);
    p.backBtn = tool("Back to live", () => backToLive(p), false);
    p.diffBtn.hidden = true;
    p.backBtn.hidden = true;

    p.thinkSummary = h("summary", {
      onclick: () => {
        p.touched = true;
      },
    });
    p.thinkText = h("pre", {});
    p.think = h("details", { class: "think", hidden: true }, p.thinkSummary, p.thinkText);

    p.empty = h("div", { class: "empty-pane" });
    p.linesBox = h("div", {});
    p.diffBox = h("div", { hidden: true });
    p.body = h("div", { class: "pane-body", role: "log", "aria-live": "off", "aria-label": `${ROLE_TITLE[role]} output`, tabindex: 0 }, p.empty, p.linesBox, p.diffBox);
    p.body.addEventListener("scroll", () => {
      const atEnd = p.body.scrollHeight - p.body.scrollTop - p.body.clientHeight < 40;
      if (atEnd !== p.follow) {
        p.follow = atEnd;
        syncTools(p);
      }
    });

    p.root.replaceChildren(
      h(
        "header",
        { class: "pane-head" },
        h("div", { class: "pane-title" }, h("h2", { id: `h-pane-${role}`, text: ROLE_TITLE[role] }), p.model, p.badge),
        p.meta,
        h("div", { class: "pane-tools" }, p.followBtn, p.wrapBtn, p.diffBtn, p.copyBtn, p.promptBtn, p.backBtn),
      ),
      p.think,
      p.body,
    );
    syncTools(p);
  }

  // The header line of a pane: model, state, and how much has been written so far.
  function paneHead(p, d, streaming) {
    const run = S.state ? S.state.run : null;
    const params = (run && run.params) || {};
    p.model.textContent = (d && d.model) || params[`${p.role}_model`] || "";

    let glyph = "○";
    let cls = "never";
    let text = busy() ? "waiting" : "idle";
    if (d && d.replay) {
      [glyph, cls, text] = ["↺", "never", `replay of call #${d.index}`];
    } else if (streaming && !d.chunks) {
      [glyph, cls, text] = ["…", "starting", `waiting for the first token · ${dur1(nowSec() - d.started_at)}`];
    } else if (streaming) {
      [glyph, cls, text] = ["●", "running pulse", `writing · ${dur1(nowSec() - d.started_at)}`];
    } else if (d && d.done && d.ok) {
      [glyph, cls, text] = ["✓", "ok", `done in ${dur1(d.updated_at - d.started_at)}`];
    } else if (d && d.done) {
      [glyph, cls, text] = ["✕", "dead", "the call failed"];
    } else if (d) {
      [glyph, cls, text] = ["■", "stopped", "interrupted"];
    }
    p.badge.replaceChildren(h("span", { class: `glyph ${cls}`, "aria-hidden": "true", text: glyph }), text);

    const parts = [];
    if (!d) {
      parts.push(ROLE_ABOUT[p.role]);
    } else {
      if (d.replay) parts.push(`${clock(d.ts)} · ${d.model}`);
      if (d.purpose) parts.push(d.purpose);
      if (d.node) parts.push(`${d.node}${d.operator ? " · " + d.operator : ""}${d.parent ? " ← " + d.parent : ""}`);
      parts.push(`${compact(d.chunks)} tokens`);
      const took = d.replay ? d.seconds : d.first_at ? d.updated_at - d.first_at : 0;
      if (d.chunks && took > 0) parts.push(`${(d.chunks / took).toFixed(1)} tok/s`);
      const chars = (d.answer || "").length + (d.thinking || "").length;
      parts.push(`${(chars / 1024).toFixed(1)} KB`);
    }
    p.meta.replaceChildren(...parts.map((t) => h("span", { text: t })));
  }

  function renderPane(role) {
    const p = P[role];
    const d = paneData(p);
    const streaming = Boolean(d && !d.replay && !d.done && busy());

    if ((d ? d.id : null) !== p.id) {
      p.id = d ? d.id : null;
      p.touched = false;
      p.diff = false;
      p.diffKey = "";
      p.follow = !d || !d.done; // a finished reply is read from the top
      resetLines(p);
      p.body.scrollTop = 0;
    }
    p.root.classList.toggle("streaming", streaming);
    paneHead(p, d, streaming);

    const thinking = d ? d.thinking : "";
    p.think.hidden = !thinking;
    if (thinking) {
      p.thinkSummary.textContent = `Reasoning · ${compact(thinking.length)} characters`;
      p.thinkText.textContent = thinking;
      if (!p.touched) p.think.open = streaming && !d.answer;
      if (p.think.open) p.thinkText.scrollTop = p.thinkText.scrollHeight;
    }

    const answer = d ? d.answer : "";
    const code = role === "coder" && d && !d.replay && d.done ? extractCode(answer) : "";
    const canDiff = Boolean(code && d.parent);
    if (!canDiff) p.diff = false;
    p.diffBtn.hidden = !canDiff;
    if (canDiff) p.diffBtn.textContent = p.diff ? "Show full reply" : `Changes vs ${d.parent}`;
    const showDiff = p.diff && canDiff;
    p.diffBox.hidden = !showDiff;
    p.linesBox.hidden = showDiff;

    if (!answer && !(thinking && streaming)) {
      p.empty.textContent = !d
        ? busy()
          ? `Nothing from the ${role} yet. It appears here as the model writes.`
          : `No ${role} output in this workspace yet.`
        : streaming
          ? "Waiting for the model to start writing…"
          : "The model returned nothing.";
      p.empty.hidden = false;
    } else {
      p.empty.hidden = true;
    }

    if (showDiff) {
      const old = S.codeCache.get(d.parent);
      const key = `${d.id}|${d.parent}`;
      if (old == null) {
        p.diffBox.replaceChildren(h("div", { class: "empty-pane", text: `Loading ${d.parent}…` }));
        p.diffKey = "";
        if (old === undefined) loadParentCode(d.parent).then(() => renderPane(role));
      } else if (!old) {
        p.diffBox.replaceChildren(h("div", { class: "empty-pane", text: `The script of ${d.parent} is not available.` }));
      } else if (p.diffKey !== key) {
        p.diffKey = key;
        renderDiff(p.diffBox, old, code, d.parent);
      }
    } else {
      paintLines(p, answer, streaming);
      if (p.follow) scrollEnd(p);
    }
    syncTools(p);
  }

  // ---- the step in progress, the call rail, tabs ----------------------------------------

  function stepStages(run) {
    const steps = run.operator === "debug" ? ["choose", "code", "run", "grade"] : ["choose", "plan", "code", "run", "grade"];
    const coder = S.live && S.live.calls.coder;
    const coding = Boolean(coder && coder.node === run.node && coder.started_at >= run.phase_since - 1);
    const now = {
      selecting: "choose",
      writing: run.operator === "debug" || coding ? "code" : "plan",
      running: "run",
      evaluating: "grade",
    }[run.phase];
    return { steps, active: steps.indexOf(now) };
  }

  function stepTook(name, run) {
    const rec = S.live && S.live.calls[name === "plan" ? "planner" : "coder"];
    if (!["plan", "code"].includes(name) || !rec || rec.node !== run.node || !rec.done) return "";
    if (name === "plan" && !String(rec.purpose).startsWith("plan")) return "";
    return dur1(rec.updated_at - rec.started_at);
  }

  function renderPipeline() {
    const box = $("#pipeline");
    const run = S.state ? S.state.run : null;
    if (!run || run.state !== "running" || !STEP_PHASES.has(run.phase)) {
      box.hidden = true;
      box.replaceChildren();
      return;
    }
    const { steps, active } = stepStages(run);
    box.hidden = false;
    box.replaceChildren(
      ...steps.map((name, i) => {
        const state = i < active ? "done" : i === active ? "active" : "pending";
        const took = stepTook(name, run);
        return h(
          "li",
          { class: state, "aria-current": state === "active" ? "step" : null },
          h("span", { class: "ico" + (state === "active" ? " pulse" : ""), "aria-hidden": "true", text: state === "done" ? "✓" : state === "active" ? "●" : "" }),
          STEP_LABEL[name],
          took ? h("span", { class: "took", text: took }) : null,
        );
      }),
    );
  }

  function renderStepHead() {
    const box = $("#step-head");
    const run = S.state ? S.state.run : null;
    if (!(run && run.state === "running" && run.node)) {
      box.replaceChildren(h("span", { class: "muted", text: nowText() }));
      return;
    }
    const parent = run.parent ? S.state.nodes.find((d) => d.id === run.parent) : null;
    box.replaceChildren(
      h("span", { class: "inline big" }, h("span", { class: `mark bg-${op(run.operator)}` }), run.node),
      h("span", { class: "muted", text: run.operator }),
      parent
        ? h("span", { class: "muted", text: `builds on ${parent.id}${parent.cv != null ? ` (score ${f4(parent.cv)})` : ""}` })
        : run.operator === "draft"
          ? h("span", { class: "muted", text: "a new starting point" })
          : null,
    );
  }

  function allLive() {
    for (const role of ROLES) {
      P[role].replay = null;
      renderPane(role);
    }
    renderRail();
  }

  function renderRail() {
    const replaying = (c) => ROLES.some((r) => P[r].replay && P[r].replay.i === c.i);
    const recent = S.calls.slice(-16).reverse();
    $("#rail").replaceChildren(
      h("button", { class: "chip", type: "button", "aria-pressed": String(!ROLES.some((r) => P[r].replay)), text: "live", onclick: allLive }),
      ...recent.map((c) =>
        h(
          "button",
          { class: "chip", type: "button", "aria-pressed": String(replaying(c)), title: `${c.model} · ${compact(c.prompt_tokens)} tokens in`, onclick: () => openReplay(c.i) },
          h("span", { class: "muted", text: `#${c.i}` }),
          c.role,
          h("span", { class: "muted", text: `${dur1(c.seconds)} · ${compact(c.completion_tokens)} tok` }),
        ),
      ),
      recent.length ? null : h("span", { class: "muted", text: "No finished call yet." }),
    );
  }

  function renderTabs() {
    const run = S.state ? S.state.run : null;
    $("#tab-dot").hidden = !(run && run.state === "running" && run.llm);
  }

  function renderLive() {
    if (S.view !== "live" || !S.state || !P.planner) return;
    safely(renderStepHead);
    safely(renderPipeline);
    for (const role of ROLES) safely(() => renderPane(role));
    safely(renderRail);
    safely(renderLiveFeed);
  }

  // Once a second, so the elapsed times of a call in flight move between model updates.
  function renderLiveClock() {
    if (S.view !== "live" || !S.state || !P.planner) return;
    for (const role of ROLES) {
      const d = paneData(P[role]);
      if (d && !d.replay && !d.done && busy()) paneHead(P[role], d, true);
    }
  }

  // ---- live polling and the view switch ---------------------------------------------------

  let liveTimer = 0;
  let liveBusy = false;

  function liveWriting() {
    const run = S.state && S.state.run;
    if (!run || run.state !== "running") return false;
    return Boolean(run.llm || (S.live && ROLES.some((r) => S.live.calls[r] && !S.live.calls[r].done)));
  }

  async function liveTick() {
    if (liveBusy) return;
    liveBusy = true;
    let delay = 1000;
    if (S.view === "live" && !document.hidden) {
      try {
        const res = await get("/api/live", S.liveEtag);
        if (!res.same) {
          S.live = res.data;
          S.liveEtag = res.etag;
          safely(renderLive);
        }
        delay = liveWriting() ? 350 : 1500;
      } catch (err) {
        delay = 3000; // the main poll reports being offline
      }
    }
    liveBusy = false;
    liveTimer = setTimeout(liveTick, delay);
  }

  function liveNow() {
    S.liveEtag = null;
    clearTimeout(liveTimer);
    liveTick();
  }

  function applyView() {
    const live = S.view === "live";
    $("#view-live").hidden = !live;
    $("#view-overview").hidden = live;
    for (const name of ["live", "overview"]) {
      const tab = $(`#tab-${name}`);
      const on = (name === "live") === live;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
    }
  }

  function setView(view, remember) {
    S.view = view;
    applyView();
    if (remember) {
      try {
        localStorage.setItem("kagent-view", view);
      } catch (err) {
        /* not persisted */
      }
    }
    if (view === "live") {
      renderLive();
      liveNow();
    } else if (S.state) {
      renderBody(); // the charts measure their width, which is zero while the view is hidden
    }
  }

  function initLive() {
    ROLES.forEach(buildPane);
    $("#tab-live").addEventListener("click", () => setView("live", true));
    $("#tab-overview").addEventListener("click", () => setView("overview", true));
    $(".views").addEventListener("keydown", (ev) => {
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
      setView(S.view === "live" ? "overview" : "live", true);
      $(`#tab-${S.view}`).focus();
      ev.preventDefault();
    });
    try {
      const saved = localStorage.getItem("kagent-view");
      if (saved === "live" || saved === "overview") S.view = saved;
    } catch (err) {
      /* the first state decides */
    }
    applyView();
    liveTick();
  }

  // ---- polling and boot ----------------------------------------------------------------

  function safely(fn) {
    try {
      fn();
    } catch (err) {
      console.error(fn.name, err);
    }
  }

  function renderBody() {
    if (S.view === "live") return;
    for (const fn of [renderKpis, renderTimeline, renderGains, renderFunnel, renderTree, renderTable, renderLlm, renderSide]) safely(fn);
    S.seenNodes = S.state.nodes.length;
  }

  function renderAll() {
    safely(renderHeader);
    safely(renderRunButtons);
    safely(renderFilters);
    renderBody();
    safely(renderLive);
    safely(renderTabs);
    safely(renderClock);
  }

  async function pull() {
    const st = S.state;
    if (st.event_count < S.eventLast) {
      S.events = [];
      S.eventLast = 0;
    }
    if (st.event_count > S.eventLast) {
      const { data } = await get(`/api/events?after=${S.eventLast}`);
      S.events = S.events.concat(data.events).slice(-1000);
      S.eventLast = data.last;
      safely(renderFeed);
    }
    if (st.call_count < S.callLast + 1) {
      S.calls = [];
      S.callLast = -1;
    }
    if (st.call_count > S.callLast + 1) {
      const { data } = await get(`/api/calls?after=${S.callLast}`);
      S.calls = S.calls.concat(data.calls).slice(-200);
      S.callLast = data.last;
      safely(renderLlm);
      safely(renderRail);
    }
    if (st.run_log_size !== S.consoleSize) await renderConsole();
  }

  let polling = false;
  let pollTimer = 0;
  function pollNow() {
    S.etag = null;
    clearTimeout(pollTimer);
    tick();
  }

  async function tick() {
    if (polling) return;
    polling = true;
    try {
      const res = await get("/api/state", S.etag);
      if (!res.same) {
        S.state = res.data;
        S.etag = res.etag;
        S.skew = res.data.server_time - Date.now() / 1000;
        if (!S.view) {
          S.view = busy() ? "live" : "overview"; // a saved choice wins; otherwise follow the search
          applyView();
          if (S.view === "live") liveNow();
        }
        renderAll();
        await pull();
        if (S.selected) await loadDrawer();
      }
      S.lastOk = Date.now();
      S.error = null;
    } catch (err) {
      S.error = err.message;
      if (!S.state) $("#kpis").replaceChildren(h("div", { class: "empty", text: err.message }));
    }
    polling = false;
    safely(renderClock);
    safely(renderRunButtons);
    safely(renderTabs);
    pollTimer = setTimeout(tick, document.hidden ? 10 * POLL_MS : POLL_MS);
  }

  function initTheme() {
    const root = document.documentElement;
    try {
      const saved = localStorage.getItem("kagent-theme");
      if (saved === "light" || saved === "dark") root.dataset.theme = saved;
    } catch (err) {
      /* storage can be unavailable; the OS setting applies */
    }
    $("#theme").addEventListener("click", () => {
      const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = current === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem("kagent-theme", next);
      } catch (err) {
        /* not persisted */
      }
    });
  }

  S.follow = () => $("#follow").checked;

  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    if (!$("#modal").hidden) closeModal();
    else if (!$("#drawer").hidden) closeDrawer();
  });
  $("#modal").addEventListener("click", (ev) => {
    if (ev.target === $("#modal")) closeModal();
  });

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!S.state || S.view === "live") return;
      safely(renderTimeline);
      safely(renderGains);
      safely(renderFunnel);
    }, 150);
  });

  $("#run-btn").addEventListener("click", openRunForm);
  $("#stop-btn").addEventListener("click", stopSearch);
  safely(renderRunButtons);
  initTheme();
  buildFilters();
  initLive();
  tick();
  setInterval(() => {
    safely(renderClock);
    safely(renderLiveClock);
  }, 1000);
})();
