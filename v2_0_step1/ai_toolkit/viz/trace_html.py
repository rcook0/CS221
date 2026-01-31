from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Optional, Sequence, Set, Tuple, TypeVar

from ..search import SearchResult, SearchTrace

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', r'\"')


def write_search_dot(
    res: SearchResult[S, A],
    out_path: str | Path,
    *,
    title: str = "search",
    max_nodes: int = 2000,
    include_cost_in_labels: bool = True,
) -> Path:
    """Write a Graphviz DOT file for the search tree.

    - Nodes are the discovered states in the final parent tree.
    - Edges are the chosen parent pointers (i.e., a tree/forest).
    - The solution path (if any) is emphasized.

    Tip:
      dot -Tpng search.dot -o search.png
    """
    if res.trace is None:
        raise ValueError("SearchResult has no trace. Run the algorithm with trace=True.")

    trace = res.trace
    parent = trace.parent

    # Ensure we always include the actual solution path.
    path_states: Sequence[S] = res.states
    keep: Set[S] = set(path_states)

    # Add other nodes up to max_nodes.
    for s in parent.keys():
        if len(keep) >= max_nodes:
            break
        keep.add(s)

    expanded_set = set(trace.expanded_order)

    # Compute path edges for emphasis
    path_edges: Set[Tuple[S, S]] = set()
    for u, v in zip(path_states, path_states[1:]):
        path_edges.add((u, v))

    def node_id(s: S) -> str:
        # deterministic-ish, but safe: use repr as stable label and hash for id
        return f"n{abs(hash(s))}"

    lines = []
    lines.append("digraph Search {")
    lines.append("  rankdir=LR;")
    lines.append(f"  labelloc=\"t\";")
    lines.append(
        "  label=\""
        + _dot_escape(
            f"{title} | cost={res.cost} | expanded={res.expanded} | generated={res.generated} | reopens={res.reopens} | runtime={res.runtime_sec:.6f}s"
        )
        + "\";"
    )

    # Nodes
    for s in keep:
        g = trace.g_score.get(s)
        lbl = repr(s)
        if include_cost_in_labels and g is not None:
            lbl = f"{lbl}\\ng={g:.4g}"
        shape = "box" if s in expanded_set else "ellipse"
        periph = "2" if s == path_states[-1] else "1"
        lines.append(f"  {node_id(s)} [label=\"{_dot_escape(lbl)}\", shape={shape}, peripheries={periph}];")

    # Edges from the parent tree.
    for child, (par, act) in parent.items():
        if par is None or act is None:
            continue
        if par not in keep or child not in keep:
            continue
        # emphasize solution path
        attrs = []
        if (par, child) in path_edges:
            attrs.append("penwidth=3")
        label = _dot_escape(str(act))
        attrs.append(f"label=\"{label}\"")
        attr_str = ", ".join(attrs)
        lines.append(f"  {node_id(par)} -> {node_id(child)} [{attr_str}];")

    lines.append("}")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_search_trace_jsonl(
    trace: SearchTrace[S, A],
    out_path: str | Path,
    *,
    max_edges: int = 1_000_000,
) -> Path:
    """Write a simple JSONL trace.

    Events:
      - {"type":"expand", "state":..., "idx": i}
      - {"type":"edge", "src":..., "dst":..., "action":..., "cost":...}

    Use with trace_edges=True to get edge events.
    """
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(trace.expanded_order):
            f.write(json.dumps({"type": "expand", "idx": i, "state": repr(s)}) + "\n")

        if trace.generated_edges is not None:
            n = 0
            for src, dst, act, cost in trace.generated_edges:
                f.write(
                    json.dumps(
                        {
                            "type": "edge",
                            "src": repr(src),
                            "dst": repr(dst),
                            "action": str(act),
                            "cost": float(cost),
                        }
                    )
                    + "\n"
                )
                n += 1
                if n >= max_edges:
                    break
    return out_path


def write_search_trace_html(
    res: SearchResult[S, A],
    out_path: str | Path,
    *,
    title: str = "search trace",
    max_nodes: int = 5000,
    max_edges: int = 20000,
) -> Path:
    """Write a self-contained HTML search visualizer.

    The output has no external dependencies (no Graphviz). It can visualize:
      - expanded order (timeline)
      - parent tree (always)
      - generated edges (optional; requires trace_edges=True)
      - solution path (if any)

    Notes:
      - For small integer state spaces (like the tram domain), the graph is drawn on a 1D axis.
      - For general state strings, it falls back to a list + parent-edge highlighting.
    """
    if res.trace is None:
        raise ValueError("SearchResult has no trace. Run the algorithm with trace=True.")

    tr = res.trace

    # Choose a bounded node set: always include solution path, then fill with expanded-order.
    path_states: Sequence[S] = res.states
    keep: Set[S] = set(path_states)
    for s in tr.expanded_order:
        if len(keep) >= max_nodes:
            break
        keep.add(s)

    # Materialize serializable structures with repr(state) keys.
    def r(s: S) -> str:
        return repr(s)

    keep_r = {r(s) for s in keep}

    parent_r: Dict[str, Dict[str, Any]] = {}
    for child, (par, act) in tr.parent.items():
        rc = r(child)
        if rc not in keep_r:
            continue
        parent_r[rc] = {
            "parent": r(par) if par is not None else None,
            "action": str(act) if act is not None else None,
        }

    g_r: Dict[str, float] = {}
    for s, g in tr.g_score.items():
        rs = r(s)
        if rs in keep_r:
            g_r[rs] = float(g)

    expanded_r = [r(s) for s in tr.expanded_order if r(s) in keep_r]
    solution_r = [r(s) for s in res.states if r(s) in keep_r]

    edges_r = []
    if tr.generated_edges is not None:
        n = 0
        for src, dst, act, cost in tr.generated_edges:
            rs, rd = r(src), r(dst)
            if rs not in keep_r or rd not in keep_r:
                continue
            edges_r.append({"src": rs, "dst": rd, "action": str(act), "cost": float(cost)})
            n += 1
            if n >= max_edges:
                break

    meta = {
        "title": title,
        "cost": float(res.cost),
        "expanded": int(res.expanded),
        "generated": int(res.generated),
        "reopens": int(res.reopens),
        "max_frontier": int(res.max_frontier),
        "runtime_sec": float(res.runtime_sec),
        "nodes_kept": int(len(keep_r)),
        "edges_kept": int(len(edges_r)),
    }

    data = {
        "meta": meta,
        "expanded_order": expanded_r,
        "solution": solution_r,
        "parent": parent_r,
        "g_score": g_r,
        "edges": edges_r,
    }

    # A single-file HTML app.
    html = f"""<!doctype html>
<html lang=\"en\">
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>{_html_escape(title)}</title>
<style>
  :root {{ --bg:#0b0f14; --panel:#101826; --ink:#e8eef5; --muted:#a7b6c6; --accent:#7bdcff; --warn:#ffcc66; --good:#b7ffb0; }}
  html, body {{ margin:0; height:100%; background:var(--bg); color:var(--ink); font:14px/1.35 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  .wrap {{ display:grid; grid-template-columns: 380px 1fr; height:100%; }}
  .side {{ background:var(--panel); border-right:1px solid rgba(255,255,255,.06); padding:16px; overflow:auto; }}
  .main {{ display:flex; flex-direction:column; overflow:hidden; }}
  h1 {{ font-size:16px; margin:0 0 8px 0; }}
  .meta {{ color:var(--muted); margin-bottom:12px; }}
  .row {{ display:flex; gap:8px; align-items:center; margin:10px 0; flex-wrap:wrap; }}
  button {{ background:rgba(255,255,255,.08); color:var(--ink); border:1px solid rgba(255,255,255,.12); padding:6px 10px; border-radius:10px; cursor:pointer; }}
  button:hover {{ border-color:rgba(255,255,255,.22); }}
  input[type=range] {{ width:100%; }}
  .kv {{ display:grid; grid-template-columns: 1fr 1fr; gap:6px 12px; }}
  .kv div {{ padding:6px 8px; background:rgba(255,255,255,.04); border-radius:10px; }}
  .kv b {{ color:var(--muted); font-weight:600; margin-right:6px; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:999px; background:rgba(123,220,255,.12); color:var(--accent); border:1px solid rgba(123,220,255,.18); font-size:12px; }}
  canvas {{ width:100%; height:100%; background:radial-gradient(1200px 600px at 10% 10%, rgba(123,220,255,.06), transparent), radial-gradient(900px 500px at 90% 90%, rgba(183,255,176,.05), transparent); }}
  .pane {{ flex:1; position:relative; }}
  .footer {{ padding:10px 12px; border-top:1px solid rgba(255,255,255,.06); color:var(--muted); display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .list {{ margin-top:10px; max-height:38vh; overflow:auto; border:1px solid rgba(255,255,255,.08); border-radius:12px; }}
  .list .item {{ padding:6px 10px; border-bottom:1px solid rgba(255,255,255,.06); }}
  .list .item:last-child {{ border-bottom:none; }}
  .item.current {{ background:rgba(123,220,255,.10); }}
  .item.solution {{ background:rgba(183,255,176,.10); }}
  .pill {{ display:inline-block; padding:1px 7px; border-radius:999px; border:1px solid rgba(255,255,255,.14); color:var(--muted); font-size:12px; }}
</style>

<div class=\"wrap\">
  <div class=\"side\">
    <h1 class=\"mono\" id=\"title\"></h1>
    <div class=\"meta\" id=\"meta\"></div>

    <div class=\"row\">
      <button id=\"btnFirst\">⏮</button>
      <button id=\"btnPrev\">◀</button>
      <button id=\"btnPlay\">▶</button>
      <button id=\"btnNext\">▶▶</button>
      <button id=\"btnLast\">⏭</button>
      <span class=\"pill\" id=\"stepLabel\"></span>
    </div>

    <div class=\"row\">
      <label style=\"flex:1\">Step</label>
      <span class=\"tag\" id=\"algoHint\"></span>
    </div>
    <input id=\"slider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\"/>

    <div class=\"row\">
      <label style=\"flex:1\">Speed</label>
      <span class=\"pill\" id=\"speedLabel\"></span>
    </div>
    <input id=\"speed\" type=\"range\" min=\"1\" max=\"120\" value=\"30\"/>

    <div class=\"row\">
      <button id=\"btnToggleEdges\">Toggle edges</button>
      <button id=\"btnToggleParents\">Toggle parents</button>
      <button id=\"btnToggleLabels\">Toggle labels</button>
    </div>

    <div class=\"kv\" id=\"kv\"></div>

    <div class=\"list\" id=\"list\"></div>
  </div>

  <div class=\"main\">
    <div class=\"pane\">
      <canvas id=\"cv\"></canvas>
    </div>
    <div class=\"footer\">
      <div>Legend: <span class=\"tag\">current</span> <span class=\"pill\">expanded</span> <span class=\"pill\">frontier</span> <span class=\"pill\">solution</span></div>
      <div class=\"mono\" id=\"status\"></div>
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(data)};

// --- small helpers
const $ = (id) => document.getElementById(id);
function clamp(x,a,b){{ return Math.max(a, Math.min(b,x)); }}
function tryParseInt(s){{
  // s is repr(state). For ints, repr is like "12" or "-3".
  if (/^-?\d+$/.test(s)) return parseInt(s,10);
  return null;
}}

// --- derived structures
const expanded = DATA.expanded_order;
const solution = new Set(DATA.solution);
const parent = DATA.parent;
const gScore = DATA.g_score;
const edges = DATA.edges;
const nodes = Object.keys(parent);

const allInt = nodes.length > 0 && nodes.every(n => tryParseInt(n) !== null);
const nodeNum = new Map(nodes.map(n => [n, tryParseInt(n)]));
const minN = allInt ? Math.min(...nodes.map(n => nodeNum.get(n))) : 0;
const maxN = allInt ? Math.max(...nodes.map(n => nodeNum.get(n))) : 1;

// Build parent edges (always)
const parentEdges = [];
for (const [child, info] of Object.entries(parent)) {{
  if (info.parent !== null && parent[info.parent] !== undefined) parentEdges.push([info.parent, child]);
}}

// Expanded prefix set is maintained incrementally by stepping.
let step = 0;
let playing = false;
let showEdges = true;
let showParents = true;
let showLabels = true;
let timer = null;

function updateKV(currentState) {{
  const m = DATA.meta;
  const kv = [
    ["total_expanded", expanded.length],
    ["final_cost", m.cost],
    ["generated", m.generated],
    ["reopens", m.reopens],
    ["max_frontier", m.max_frontier],
    ["runtime_sec", m.runtime_sec.toFixed(6)],
    ["nodes_kept", m.nodes_kept],
    ["edges_kept", m.edges_kept],
    ["current", currentState],
    ["g(current)", (gScore[currentState] ?? "-")],
    ["parent", (parent[currentState]?.parent ?? "-")],
    ["action", (parent[currentState]?.action ?? "-")],
  ];
  const el = $("kv");
  el.innerHTML = "";
  for (const [k,v] of kv) {{
    const d = document.createElement("div");
    d.innerHTML = `<b>${{k}}</b><span class=\"mono\">${{String(v)}}</span>`;
    el.appendChild(d);
  }}
}}

function updateList(currentState) {{
  const list = $("list");
  const start = Math.max(0, step - 200);
  const end = Math.min(expanded.length, step + 200);
  list.innerHTML = "";
  for (let i=start; i<end; i++) {{
    const s = expanded[i];
    const div = document.createElement("div");
    div.className = "item" + (s===currentState?" current":"") + (solution.has(s)?" solution":"");
    div.innerHTML = `<span class=\"pill\">${{i}}</span> <span class=\"mono\">${{s}}</span>`;
    list.appendChild(div);
  }}
}}

function layoutPoint(state, w, h) {{
  if (allInt) {{
    const n = nodeNum.get(state);
    const x = 40 + (w-80) * ((n - minN) / (maxN - minN || 1));
    // y encodes g-score, but keep it readable.
    const g = gScore[state];
    const yg = (g === undefined) ? 0.5 : clamp(1.0 - (Math.log10(1+g) / 3.0), 0.12, 0.88);
    const y = 40 + (h-80) * yg;
    return [x,y];
  }}
  // fallback: hash to a pseudo-grid.
  let hash = 0;
  for (let i=0; i<state.length; i++) hash = (hash*31 + state.charCodeAt(i)) >>> 0;
  const cols = 18;
  const r = Math.floor(hash / cols) % cols;
  const c = hash % cols;
  const x = 40 + (w-80) * (c / (cols-1));
  const y = 40 + (h-80) * (r / (cols-1));
  return [x,y];
}}

function draw() {{
  const cv = $("cv");
  const ctx = cv.getContext("2d");
  const rect = cv.getBoundingClientRect();
  const w = cv.width = Math.floor(rect.width * devicePixelRatio);
  const h = cv.height = Math.floor(rect.height * devicePixelRatio);
  ctx.scale(devicePixelRatio, devicePixelRatio);

  const cw = rect.width;
  const ch = rect.height;

  // Determine state sets
  const expandedSet = new Set(expanded.slice(0, step+1));
  const current = expanded[step] ?? expanded[expanded.length-1] ?? "";
  const frontier = new Set();
  // Approx frontier: nodes discovered (in parent) but not yet expanded at this step.
  for (const s of nodes) {{
    if (!expandedSet.has(s) && parent[s]?.parent !== null) frontier.add(s);
  }}

  // Edges
  function drawEdge(u,v, alpha, width, dash) {{
    const [x1,y1] = layoutPoint(u,cw,ch);
    const [x2,y2] = layoutPoint(v,cw,ch);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.strokeStyle = "rgba(232,238,245,0.55)";
    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x2,y2);
    ctx.stroke();
    ctx.restore();
  }}

  if (showParents) {{
    for (const [u,v] of parentEdges) {{
      // highlight solution edges
      const isSol = solution.has(u) && solution.has(v);
      drawEdge(u,v, isSol?0.75:0.20, isSol?2.5:1.0, null);
    }}
  }}

  if (showEdges && edges.length) {{
    // light overlay of generated edges
    const maxDraw = Math.min(edges.length, 4000);
    for (let i=0;i<maxDraw;i++) {{
      const e = edges[i];
      drawEdge(e.src, e.dst, 0.10, 1.0, [2,3]);
    }}
  }}

  // Nodes
  function nodeStyle(s) {{
    if (s === current) return {{r:8, fill:"rgba(123,220,255,0.95)", stroke:"rgba(123,220,255,0.95)", lw:2}};
    if (solution.has(s)) return {{r:6, fill:"rgba(183,255,176,0.75)", stroke:"rgba(183,255,176,0.85)", lw:1.5}};
    if (expandedSet.has(s)) return {{r:5, fill:"rgba(232,238,245,0.65)", stroke:"rgba(232,238,245,0.75)", lw:1.2}};
    if (frontier.has(s)) return {{r:4, fill:"rgba(255,204,102,0.65)", stroke:"rgba(255,204,102,0.75)", lw:1.0}};
    return {{r:3, fill:"rgba(167,182,198,0.35)", stroke:"rgba(167,182,198,0.35)", lw:1.0}};
  }}

  // draw nodes, but prioritize visible sets
  const drawOrder = [
    ...nodes.filter(s=>!expandedSet.has(s) && !frontier.has(s) && !solution.has(s) && s!==current),
    ...Array.from(frontier),
    ...Array.from(expandedSet),
    ...Array.from(solution),
    current,
  ];

  const seen = new Set();
  for (const s of drawOrder) {{
    if (!s || seen.has(s) || parent[s] === undefined) continue;
    seen.add(s);
    const [x,y] = layoutPoint(s,cw,ch);
    const st = nodeStyle(s);
    ctx.save();
    ctx.beginPath();
    ctx.arc(x,y,st.r,0,Math.PI*2);
    ctx.fillStyle = st.fill;
    ctx.fill();
    ctx.lineWidth = st.lw;
    ctx.strokeStyle = st.stroke;
    ctx.stroke();
    ctx.restore();

    if (showLabels && (s === current || solution.has(s) || (allInt && (nodeNum.get(s) % Math.ceil((maxN-minN+1)/12 || 1) === 0)))) {{
      ctx.save();
      ctx.fillStyle = "rgba(232,238,245,0.82)";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText(s, x + st.r + 4, y - st.r - 2);
      ctx.restore();
    }}
  }}

  // status
  $("status").textContent = `step ${{step}}/${{expanded.length-1}} | current=${{current}}`;
  updateKV(current);
  updateList(current);
  $("stepLabel").textContent = `step ${{step}}`;
}}

function setStep(newStep) {{
  step = clamp(newStep, 0, Math.max(0, expanded.length-1));
  $("slider").value = String(step);
  draw();
}}

function play() {{
  if (playing) return;
  playing = true;
  $("btnPlay").textContent = "⏸";
  const tick = () => {{
    const spd = parseInt($("speed").value, 10);
    $("speedLabel").textContent = `${{spd}} fps`;
    if (!playing) return;
    setStep(step + 1);
    if (step >= expanded.length - 1) {{ stop(); return; }}
    timer = setTimeout(tick, Math.floor(1000 / spd));
  }};
  tick();
}}

function stop() {{
  playing = false;
  $("btnPlay").textContent = "▶";
  if (timer) {{ clearTimeout(timer); timer = null; }}
}}

// init
$("title").textContent = DATA.meta.title;
$("meta").innerHTML = `<span class=\"mono\">cost=${{DATA.meta.cost}}</span> · expanded=${{DATA.meta.expanded}} · generated=${{DATA.meta.generated}} · reopens=${{DATA.meta.reopens}}`;
$("algoHint").textContent = edges.length ? "trace_edges:on" : "trace_edges:off";

$("slider").max = String(Math.max(0, expanded.length-1));
$("slider").addEventListener("input", (e) => setStep(parseInt(e.target.value,10)));
$("speed").addEventListener("input", () => $("speedLabel").textContent = `${{parseInt($("speed").value,10)}} fps`);

$("btnFirst").onclick = () => {{ stop(); setStep(0); }};
$("btnPrev").onclick = () => {{ stop(); setStep(step-1); }};
$("btnNext").onclick = () => {{ stop(); setStep(step+1); }};
$("btnLast").onclick = () => {{ stop(); setStep(expanded.length-1); }};
$("btnPlay").onclick = () => {{ playing ? stop() : play(); }};
$("btnToggleEdges").onclick = () => {{ showEdges = !showEdges; draw(); }};
$("btnToggleParents").onclick = () => {{ showParents = !showParents; draw(); }};
$("btnToggleLabels").onclick = () => {{ showLabels = !showLabels; draw(); }};

window.addEventListener("resize", () => draw());
$("speedLabel").textContent = `${{parseInt($("speed").value,10)}} fps`;
setStep(0);
</script>
</html>
"""

    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
