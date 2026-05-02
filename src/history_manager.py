"""
history_manager.py — Client-Side Interaction History Bridge

Generates JavaScript snippets (via Streamlit components.html) that
read/write interaction history from the browser's localStorage.

No server-side storage. No authentication. Persists across sessions.
"""

import json
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STORAGE_KEY = "smart_agri_xai_history"
MAX_ENTRIES = 50

# Fields that must NEVER appear in stored history logs
SENSITIVE_KEYS = {
    "GEMINI_API_KEY",
    "messages",
    "voice_response_audio",
    "voice_response_text",
    "voice_query",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _sanitize(state_data: Dict[str, Any]) -> Dict[str, Any]:
    """Strip sensitive keys before saving to localStorage."""
    return {k: v for k, v in state_data.items() if k not in SENSITIVE_KEYS}


def _to_js_str(data: Any) -> str:
    """Serialize Python data to a JS-safe JSON string (handles numpy types)."""
    raw = json.dumps(data, default=str)
    # Escape backslashes then single-quotes so string sits safely in JS literal
    return raw.replace("\\", "\\\\").replace("'", "\\'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_save_script(action_name: str, state_data: Dict[str, Any]) -> str:
    """
    Return an HTML <script> that saves one history entry to localStorage.
    """
    clean          = _sanitize(state_data)
    state_js_str   = _to_js_str(clean)
    action_escaped = action_name.replace("'", "\\'")

    return f"""<script>
(function() {{
    try {{
        var KEY  = '{STORAGE_KEY}';
        var MAX  = {MAX_ENTRIES};
        var data = JSON.parse('{state_js_str}');
        var item = {{ timestamp: Date.now(), action: '{action_escaped}', state: data }};
        var list = JSON.parse(localStorage.getItem(KEY) || '[]');
        list.unshift(item);
        if (list.length > MAX) {{ list = list.slice(0, MAX); }}
        localStorage.setItem(KEY, JSON.stringify(list));
    }} catch(e) {{
        console.warn('[SmartAgriXAI] History save failed:', e);
    }}
}})();
</script>"""


def build_render_script(height: int = 380) -> str:
    """
    Return a full self-contained HTML page for components.html() that renders the history panel.
    """
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  ::-webkit-scrollbar {{ width: 4px; }}
  ::-webkit-scrollbar-thumb {{ background: rgba(0,201,255,.3); border-radius: 4px; }}
  body {{
    background: transparent;
    font-family: 'Courier New', monospace;
    color: #e2e8f0;
    font-size: 12px;
    overflow-y: auto;
    max-height: {height}px;
    padding: 2px 4px;
  }}
  #empty-msg {{
    color: #475569; text-align: center;
    padding: 20px 10px; font-size: 11px; line-height: 1.7;
    border: 1px dashed rgba(0,201,255,.15); border-radius: 6px; margin-top: 6px;
  }}
  #empty-msg span {{ display:block; font-size: 22px; margin-bottom: 6px; }}
  #clear-btn {{
    width: 100%; background: transparent;
    border: 1px solid rgba(239,68,68,.35); color: #ef4444;
    border-radius: 5px; padding: 5px 0; font-size: 11px;
    cursor: pointer; font-family: 'Courier New', monospace;
    margin-bottom: 10px; letter-spacing: .6px;
    transition: background .2s, box-shadow .2s;
  }}
  #clear-btn:hover {{ background: rgba(239,68,68,.1); box-shadow: 0 0 8px rgba(239,68,68,.25); }}
  .entry {{
    background: rgba(15,23,42,.85);
    border: 1px solid rgba(0,201,255,.12);
    border-radius: 6px; padding: 8px 10px; margin-bottom: 7px;
    transition: border-color .2s, box-shadow .2s;
  }}
  .entry:hover {{ border-color: rgba(0,201,255,.4); box-shadow: 0 0 12px rgba(0,201,255,.1); }}
  .entry-top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }}
  .badge {{
    font-size: 9px; font-weight: 700; letter-spacing: .7px;
    border-radius: 4px; padding: 2px 6px; text-transform: uppercase; border: 1px solid;
  }}
  .ba {{ color:#00C9FF; border-color:rgba(0,201,255,.4); background:rgba(0,201,255,.08); }}
  .br {{ color:#a78bfa; border-color:rgba(167,139,250,.4); background:rgba(167,139,250,.08); }}
  .bd {{ color:#f87171; border-color:rgba(248,113,113,.4); background:rgba(248,113,113,.08); }}
  .ts {{ color:#64748b; font-size:10px; white-space:nowrap; }}
  .smry {{ color:#94a3b8; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .rbtn {{
    display:inline-block; margin-top:5px; background:transparent;
    border:1px solid rgba(0,201,255,.2); color:#00C9FF;
    border-radius:4px; padding:2px 8px; font-size:10px; cursor:pointer;
    font-family:'Courier New',monospace; letter-spacing:.3px;
    transition: background .2s, box-shadow .2s;
  }}
  .rbtn:hover {{ background:rgba(0,201,255,.1); box-shadow:0 0 8px rgba(0,201,255,.2); }}
</style>
</head><body><div id="root"></div>
<script>(function(){{
  var KEY = '{STORAGE_KEY}';

  function relTime(ts) {{
    var d = Math.floor((Date.now()-ts)/1000);
    if (d < 60)     return 'just now';
    if (d < 3600)   return Math.floor(d/60)   + ' min ago';
    if (d < 86400)  return Math.floor(d/3600) + ' hr ago';
    if (d < 604800) return Math.floor(d/86400) + ' day' + (Math.floor(d/86400)>1?'s':'') + ' ago';
    return Math.floor(d/604800) + ' wk ago';
  }}

  function badge(a) {{
    if (a==='Region Analysis')    return '<span class="badge ba">Region Analysis</span>';
    if (a==='Manual Recalculate') return '<span class="badge br">Manual Recalc</span>';
    if (a==='Disease Detection')  return '<span class="badge bd">Disease Detect</span>';
    return '<span class="badge ba">'+a+'</span>';
  }}

  function summary(e) {{
    var s = e.state || {{}};
    if (e.action==='Region Analysis'||e.action==='Manual Recalculate') {{
      var crop = Array.isArray(s.predicted_crop)?s.predicted_crop[0]:(s.predicted_crop||'?');
      return '📍 '+(s.location||'?')+' → 🌿 '+crop;
    }}
    if (e.action==='Disease Detection') {{
      var ok = (s.status||'').toUpperCase()==='HEALTHY';
      return '🔬 '+(s.detected_crop||'?')+' — '+(ok?'✅ HEALTHY':'⚠️ DISEASED');
    }}
    return JSON.stringify(s).substring(0,60);
  }}

  function encB64(state) {{
    try {{ return btoa(unescape(encodeURIComponent(JSON.stringify(state||{{}})))); }}
    catch(e) {{ return ''; }}
  }}

  window.clearHistory = function() {{ localStorage.removeItem(KEY); render(); }};

  window.restoreEntry = function(b64) {{
    try {{
      var base = window.parent.location.href.split('?')[0];
      window.parent.location.href = base + '?restore=' + encodeURIComponent(b64);
    }} catch(e) {{ console.warn('[SmartAgriXAI] Restore failed:', e); }}
  }};

  function render() {{
    var root = document.getElementById('root');
    var list = [];
    try {{ list = JSON.parse(localStorage.getItem(KEY)||'[]'); }} catch(e) {{}}

    if (!list.length) {{
      root.innerHTML = '<div id="empty-msg"><span>📊</span>No activity yet.<br>Analyze a region or detect a disease<br>to start building your history.</div>';
      return;
    }}

    var html = '<button id="clear-btn" onclick="clearHistory()">🗑️ Clear History ('+list.length+')</button>';
    list.forEach(function(entry) {{
      var canRestore = (entry.action==='Region Analysis'||entry.action==='Manual Recalculate');
      var b64 = encB64(entry.state);
      html += '<div class="entry">';
      html += '<div class="entry-top">'+badge(entry.action)+'<span class="ts">'+relTime(entry.timestamp||0)+'</span></div>';
      html += '<div class="smry">'+summary(entry)+'</div>';
      if (canRestore && b64) html += '<button class="rbtn" onclick="restoreEntry(\''+b64+'\')">&#8629; Restore</button>';
      html += '</div>';
    }});
    root.innerHTML = html;
  }}

  render();
}})();
</script></body></html>"""
