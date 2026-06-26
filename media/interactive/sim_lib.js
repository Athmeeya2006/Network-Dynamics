"use strict";
/* Shared helpers for the interactive simulations:
   RK4 integrator, slider/button builders, Plotly dark-theme layout. */
const SimLib = (() => {
  const COL = { bg:"#020617", card:"#0F172A", dim:"#334155", txt:"#E2E8F0",
                muted:"#94A3B8", teal:"#0E7490", red:"#DC2626", gold:"#D97706",
                purple:"#7C3AED", cyan:"#06B6D4", green:"#059669" };

  // One RK4 step for a vector field f(t, y[]) -> dy[].
  function rk4(f, t, y, dt) {
    const n = y.length;
    const k1 = f(t, y);
    const y2 = new Array(n), y3 = new Array(n), y4 = new Array(n);
    for (let i = 0; i < n; i++) y2[i] = y[i] + 0.5 * dt * k1[i];
    const k2 = f(t + 0.5 * dt, y2);
    for (let i = 0; i < n; i++) y3[i] = y[i] + 0.5 * dt * k2[i];
    const k3 = f(t + 0.5 * dt, y3);
    for (let i = 0; i < n; i++) y4[i] = y[i] + dt * k3[i];
    const k4 = f(t + dt, y4);
    const out = new Array(n);
    for (let i = 0; i < n; i++)
      out[i] = y[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return out;
  }

  // Rainbow hex colour i of n (HSL sweep).
  function rainbow(i, n) {
    const h = (360 * i / Math.max(n, 1)) | 0;
    return `hsl(${h},80%,62%)`;
  }

  // Create a labelled range slider; opts: {label,min,max,step,value,unit,fmt,onInput}.
  function slider(panel, opts) {
    const wrap = document.createElement("div"); wrap.className = "ctrl";
    const lab = document.createElement("label");
    const name = document.createElement("span"); name.textContent = opts.label;
    const val = document.createElement("span"); val.className = "val";
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = opts.min; inp.max = opts.max;
    inp.step = opts.step; inp.value = opts.value;
    const fmt = opts.fmt || (v => (+v).toFixed(2));
    const show = () => { val.textContent = fmt(inp.value) + (opts.unit || ""); };
    show();
    inp.addEventListener("input", () => { show(); if (opts.onInput) opts.onInput(+inp.value); });
    lab.appendChild(name); lab.appendChild(val);
    wrap.appendChild(lab); wrap.appendChild(inp);
    panel.appendChild(wrap);
    return inp;
  }

  function button(panel, label, onClick, cls) {
    const b = document.createElement("button");
    b.textContent = label; if (cls) b.className = cls;
    b.addEventListener("click", onClick);
    return b;
  }

  function row(panel, buttons) {
    const r = document.createElement("div"); r.className = "row";
    buttons.forEach(b => r.appendChild(b));
    panel.appendChild(r); return r;
  }

  function readout(panel) {
    const d = document.createElement("div"); d.className = "readout";
    panel.appendChild(d); return d;
  }

  function note(panel, html) {
    const d = document.createElement("div"); d.className = "note";
    d.innerHTML = html; panel.appendChild(d);
  }

  // Full-width explanation block under the plot. sections: [{title, html}, ...].
  // Pass {wide:true} to stack the boxes in a single column.
  function explain(sections, opts) {
    const sec = document.createElement("section");
    sec.className = "explain" + (opts && opts.wide ? " full" : "");
    sections.forEach(s => {
      const box = document.createElement("div"); box.className = "box";
      const h = document.createElement("h3"); h.textContent = s.title;
      const body = document.createElement("div"); body.innerHTML = s.html;
      box.appendChild(h); box.appendChild(body); sec.appendChild(box);
    });
    document.body.appendChild(sec);
    return sec;
  }

  function axis(extra) {
    return Object.assign({ gridcolor: COL.dim, zerolinecolor: COL.dim, color: COL.muted },
                         extra || {});
  }

  function layout(title, extra) {
    return Object.assign({
      paper_bgcolor: COL.bg, plot_bgcolor: COL.bg,
      font: { color: COL.txt, family: "Inter, Segoe UI, sans-serif" },
      margin: { l: 55, r: 30, t: 54, b: 50 }, showlegend: false,
      title: { text: title, x: 0.5, font: { color: COL.txt, size: 17 } }
    }, extra || {});
  }

  const config = { displayModeBar: true, displaylogo: false, scrollZoom: true,
                   responsive: true };

  return { COL, rk4, rainbow, slider, button, row, readout, note, explain,
           axis, layout, config };
})();
