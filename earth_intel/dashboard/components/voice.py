"""
components/voice.py
Embeds the full voice/chat interface (from index.html) as a self-contained
Streamlit component using st.components.v1.html.

All API calls (/api/stt, /api/tts, /api/chat) are made from within the
embedded HTML/JS and target the FastAPI backend on the same origin
(http://localhost:8000).  The Streamlit widget is purely a UI host; no
Python-side processing is needed here.

This component is rendered on the "Voice Interface" page.
"""

import streamlit as st
import streamlit.components.v1 as components


VOICE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #30363d;
      --accent:   #1f6feb;
      --accent-h: #388bfd;
      --text:     #e6edf3;
      --muted:    #8b949e;
      --success:  #3fb950;
      --warn:     #d29922;
      --danger:   #f85149;
      --radius:   8px;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.6;
      display: flex;
      flex-direction: column;
      padding: 1rem;
      gap: 1rem;
      min-height: 100%;
    }

    /* ── Header ── */
    .voice-header {
      display: flex;
      align-items: center;
      gap: 10px;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border);
    }
    .voice-header .ei-badge {
      background: linear-gradient(135deg, #14b8a6, #2563eb);
      color: #fff;
      font-weight: 900;
      font-size: 12px;
      border-radius: 8px;
      padding: 4px 8px;
      letter-spacing: 0.05em;
    }
    .voice-header h2 { font-size: 1rem; font-weight: 600; }
    .voice-header p  { font-size: 0.78rem; color: var(--muted); }

    /* ── Thread ── */
    #thread {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      min-height: 120px;
      max-height: 380px;
      overflow-y: auto;
      padding: 4px 2px;
    }

    .bubble { display: flex; flex-direction: column; gap: 0.2rem; max-width: 95%; }
    .bubble.agent { align-self: flex-start; }
    .bubble.user  { align-self: flex-end; }

    .bubble-label {
      font-size: 0.65rem; font-weight: 700;
      letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
    }
    .bubble.agent .bubble-label { color: var(--accent-h); }

    .bubble-body {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.55rem 0.8rem;
      display: flex; align-items: flex-start; gap: 0.5rem;
    }
    .bubble.user .bubble-body { background: #1c2d3f; border-color: #2a4260; }
    .bubble-text { flex: 1; font-size: 0.875rem; }

    .speak-btn {
      background: none; border: none; cursor: pointer;
      color: var(--muted); font-size: 0.9rem; padding: 0; line-height: 1;
      transition: color 0.15s; flex-shrink: 0;
    }
    .speak-btn:hover { color: var(--accent-h); }

    /* ── Result card ── */
    .result-card {
      background: var(--surface);
      border: 1px solid var(--success);
      border-radius: var(--radius);
      padding: 0.85rem 1rem;
      font-size: 0.8rem;
      width: 100%;
    }
    .result-card h3 {
      font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.06em; color: var(--success); margin-bottom: 0.5rem;
    }
    .result-card table { width: 100%; border-collapse: collapse; }
    .result-card td {
      padding: 0.22rem 0.35rem; vertical-align: top;
      border-bottom: 1px solid var(--border);
      font-size: 0.8rem;
    }
    .result-card td:first-child { color: var(--muted); white-space: nowrap; width: 36%; }
    .result-card tr:last-child td { border-bottom: none; }

    /* ── Badges ── */
    .badge {
      font-size: 0.65rem; padding: 0.12rem 0.35rem;
      border-radius: 4px; display: inline-block; margin-top: 0.2rem;
    }
    .badge-translated { background: #2a2200; color: var(--warn); border: 1px solid #5a4500; }
    .badge-lang       { background: #1a2a1a; color: var(--success); border: 1px solid #2a5a2a; }

    /* ── Input area ── */
    .input-area {
      display: flex; flex-direction: column; gap: 0.6rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.85rem;
    }

    textarea {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      font-family: inherit; font-size: 0.85rem;
      padding: 0.6rem 0.75rem;
      resize: vertical; min-height: 72px;
      outline: none; transition: border-color 0.15s;
    }
    textarea:focus { border-color: var(--accent); }
    textarea::placeholder { color: var(--muted); }

    .input-controls { display: flex; gap: 0.45rem; align-items: center; flex-wrap: wrap; }

    button {
      cursor: pointer; border: none; border-radius: var(--radius);
      font-size: 0.8rem; font-weight: 500;
      padding: 0.45rem 0.9rem;
      transition: background 0.15s, opacity 0.15s;
    }
    button:disabled { opacity: 0.4; cursor: not-allowed; }

    .btn-primary { background: var(--accent); color: #fff; }
    .btn-primary:hover:not(:disabled) { background: var(--accent-h); }

    .btn-mic { background: #21262d; color: var(--text); border: 1px solid var(--border); }
    .btn-mic:hover:not(:disabled) { background: #2d333b; }
    .btn-mic.recording {
      background: #3d1a1a; color: var(--danger);
      border-color: var(--danger); animation: pulse 1s infinite;
    }

    .btn-ghost {
      background: none; color: var(--muted); border: 1px solid var(--border);
      margin-left: auto;
    }
    .btn-ghost:hover:not(:disabled) { color: var(--text); border-color: var(--text); }

    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.65; } }

    #mic-status { font-size: 0.72rem; color: var(--muted); min-height: 1.1em; }
    #mic-status.active { color: var(--danger); }
    #mic-status.ok     { color: var(--success); }

    #status-bar {
      font-size: 0.75rem; color: var(--muted);
      text-align: center; min-height: 1.2em;
      padding: 2px 0;
    }
    #status-bar.error { color: var(--danger); }

    hr { border: none; border-top: 1px solid var(--border); }

    /* ── Session info ── */
    .session-info {
      font-size: 0.68rem; color: var(--muted);
      text-align: right; padding-top: 2px;
    }
  </style>
</head>
<body>

  <div class="voice-header">
    <div class="ei-badge">EI</div>
    <div>
      <h2>🎙 Voice &amp; Chat Interface</h2>
      <p>Type or speak your scientific query — Agent 1 will parse intent, variables, location &amp; time period</p>
    </div>
  </div>

  <div id="thread"></div>

  <hr />

  <div class="input-area">
    <textarea
      id="user-input"
      placeholder="e.g. Sea surface temperature anomalies in the Bay of Bengal during the 2023 monsoon…"
      rows="3"
    ></textarea>

    <div class="input-controls">
      <button id="mic-btn" class="btn-mic">🎤 Record</button>
      <button id="send-btn" class="btn-primary">Analyse ↑</button>
      <button id="clear-btn" class="btn-ghost">Clear</button>
    </div>

    <div id="mic-status">Click "Record" to speak your query</div>
  </div>

  <div id="status-bar"></div>
  <div class="session-info" id="session-info"></div>

<script>
  // ── State ──────────────────────────────────────────────────────────────────
  let sessionId     = null;
  let mediaRecorder = null;
  let audioChunks   = [];
  let isRecording   = false;

  // ── DOM refs ───────────────────────────────────────────────────────────────
  const thread      = document.getElementById("thread");
  const input       = document.getElementById("user-input");
  const micBtn      = document.getElementById("mic-btn");
  const sendBtn     = document.getElementById("send-btn");
  const clearBtn    = document.getElementById("clear-btn");
  const micStatus   = document.getElementById("mic-status");
  const statusBar   = document.getElementById("status-bar");
  const sessionInfo = document.getElementById("session-info");

  // ── Helpers ────────────────────────────────────────────────────────────────

  function setStatus(msg, isError = false) {
    statusBar.textContent = msg;
    statusBar.className   = isError ? "error" : "";
  }

  function setMicStatus(msg, cls = "") {
    micStatus.textContent = msg;
    micStatus.className   = cls;
  }

  function updateSessionInfo() {
    sessionInfo.textContent = sessionId ? `Session: ${sessionId.slice(0, 12)}…` : "";
  }

  function addBubble(role, contentEl, meta = {}) {
    const wrap = document.createElement("div");
    wrap.className = `bubble ${role}`;

    const label = document.createElement("div");
    label.className   = "bubble-label";
    label.textContent = role === "agent" ? "Agent 1" : "You";
    wrap.appendChild(label);

    const body = document.createElement("div");
    body.className = "bubble-body";

    if (typeof contentEl === "string") {
      const t = document.createElement("div");
      t.className   = "bubble-text";
      t.textContent = contentEl;
      body.appendChild(t);
    } else {
      contentEl.style.width = "100%";
      body.appendChild(contentEl);
    }

    // 🔊 speak button — only on plain-text agent bubbles
    if (role === "agent" && typeof contentEl === "string") {
      const speakBtn       = document.createElement("button");
      speakBtn.className   = "speak-btn";
      speakBtn.title       = "Read aloud";
      speakBtn.textContent = "🔊";
      speakBtn.onclick     = () => speak(contentEl);
      body.appendChild(speakBtn);
    }

    wrap.appendChild(body);

    // Language / translation badges
    if (meta.language && meta.language !== "en") {
      const b       = document.createElement("span");
      b.className   = "badge badge-lang";
      b.textContent = `Detected: ${meta.language}`;
      wrap.appendChild(b);
    }
    if (meta.wasTranslated && meta.original) {
      const b       = document.createElement("span");
      b.className   = "badge badge-translated";
      b.textContent = `Translated from: "${meta.original.slice(0, 55)}${meta.original.length > 55 ? "…" : ""}"`;
      wrap.appendChild(b);
    }

    thread.appendChild(wrap);
    wrap.scrollIntoView({ behavior: "smooth", block: "end" });
    return wrap;
  }

  /**
   * Build the Agent 1 result card from the ScientificIntentOutput dict.
   */
  function buildResultCard(output) {
    const card    = document.createElement("div");
    card.className = "result-card";

    const heading       = document.createElement("h3");
    heading.textContent = "✅ Agent 1 — Query Analysis Complete";
    card.appendChild(heading);

    const domain     = output.domain_confidences?.[0]?.domain ?? "—";
    const intent     = output.scientific_intent?.primary_intent ?? "—";
    const location   = output.spatial_context?.primary_location
                     ?? output.spatial_context?.location ?? "—";
    const timePeriod = output.temporal_context?.period_description
                     ?? output.temporal_context?.date_range
                     ?? output.temporal_context?.start_date ?? "—";
    const variables  = (output.scientific_variables ?? [])
                       .map(v => v.variable).filter(Boolean).join(", ") || "—";
    const complexity = output.query_complexity ?? "—";
    const goal       = output.inferred_user_research_goal
                     ?? output.scientific_intent?.research_goal ?? "—";

    const rows = [
      ["Domain",       domain],
      ["Intent",       intent],
      ["Location",     location],
      ["Time period",  timePeriod],
      ["Variables",    variables],
      ["Complexity",   complexity],
      ["Goal",         goal.length > 120 ? goal.slice(0, 120) + "…" : goal],
    ];

    const table = document.createElement("table");
    for (const [key, val] of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${key}</td><td>${val}</td>`;
      table.appendChild(tr);
    }
    card.appendChild(table);
    return card;
  }

  // ── TTS ────────────────────────────────────────────────────────────────────

  async function speak(text) {
    setStatus("Loading audio…");
    try {
      const res = await fetch("/api/tts", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text, session_id: sessionId }),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      new Audio(URL.createObjectURL(blob)).play();
      setStatus("");
    } catch (err) {
      setStatus(`TTS error: ${err.message}`, true);
    }
  }

  // ── Send query to /api/chat (Agent 1) ─────────────────────────────────────

  async function sendMessage(text, meta = {}) {
    if (!text.trim()) return;

    addBubble("user", text, meta);
    input.value    = "";
    sendBtn.disabled = true;
    micBtn.disabled  = true;
    setStatus("⏳ Agent 1 is analysing your query…");

    try {
      const res = await fetch("/api/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detail.detail ?? res.statusText);
      }

      const data = await res.json();
      sessionId  = data.session_id;
      updateSessionInfo();

      // Summary text bubble with 🔊 button
      addBubble("agent", data.summary);

      // Full structured result card
      const card  = buildResultCard(data.agent1_output);
      const wrap  = document.createElement("div");
      wrap.className = "bubble agent";
      const lbl   = document.createElement("div");
      lbl.className   = "bubble-label";
      lbl.textContent = "Agent 1 — Full Output";
      wrap.appendChild(lbl);
      const bdy   = document.createElement("div");
      bdy.className  = "bubble-body";
      bdy.style.display = "block";
      bdy.appendChild(card);
      wrap.appendChild(bdy);
      thread.appendChild(wrap);
      wrap.scrollIntoView({ behavior: "smooth", block: "end" });

      setStatus("✅ Analysis complete. You can continue in the full dashboard pipeline.");
      sendBtn.disabled = false;
      micBtn.disabled  = false;

    } catch (err) {
      setStatus(`Error: ${err.message}`, true);
      sendBtn.disabled = false;
      micBtn.disabled  = false;
    }
  }

  // ── Recording (STT) ────────────────────────────────────────────────────────

  async function startRecording() {
    if (isRecording) return;
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setMicStatus("❌ Microphone access denied. Check browser permissions.");
      return;
    }

    audioChunks = [];
    const mimeType   = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
                       ? "audio/webm;codecs=opus" : "";
    mediaRecorder    = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => { stream.getTracks().forEach(t => t.stop()); uploadAudio(); };
    mediaRecorder.start(200);

    isRecording = true;
    micBtn.textContent = "⏹ Stop";
    micBtn.classList.add("recording");
    setMicStatus("🔴 Recording… click Stop when done", "active");
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    mediaRecorder.stop();
    isRecording = false;
    micBtn.textContent = "🎤 Record";
    micBtn.classList.remove("recording");
    setMicStatus("Processing audio…");
  }

  async function uploadAudio() {
    if (!audioChunks.length) { setMicStatus("No audio captured. Try again."); return; }
    setMicStatus("Transcribing…");
    micBtn.disabled = true;

    const blob = new Blob(audioChunks, { type: "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "recording.webm");
    if (sessionId) form.append("session_id", sessionId);

    try {
      const res = await fetch("/api/stt", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      sessionId    = data.session_id;
      input.value  = data.text;
      updateSessionInfo();

      const langInfo = data.language !== "en" ? ` (detected: ${data.language})` : "";
      setMicStatus(`✅ Transcribed${langInfo} — review and click Analyse`, "ok");

    } catch (err) {
      setMicStatus(`STT error: ${err.message}`);
    } finally {
      micBtn.disabled = false;
    }
  }

  // ── Event listeners ────────────────────────────────────────────────────────

  micBtn.addEventListener("click", () => isRecording ? stopRecording() : startRecording());

  sendBtn.addEventListener("click", () => sendMessage(input.value.trim()));

  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      sendMessage(input.value.trim());
    }
  });

  clearBtn.addEventListener("click", () => {
    thread.innerHTML  = "";
    sessionId         = null;
    input.value       = "";
    sendBtn.disabled  = false;
    micBtn.disabled   = false;
    setStatus("");
    setMicStatus("Click \"Record\" to speak your query");
    updateSessionInfo();
    addBubble("agent", "Session cleared. Enter a new scientific query above.");
  });

  // ── Greeting ───────────────────────────────────────────────────────────────
  addBubble(
    "agent",
    "Hello! Enter your scientific query above — type it or use the microphone. " +
    "Agent 1 will parse your intent, variables, location, and time period. " +
    "Results will appear here and feed into the full dashboard pipeline."
  );
</script>
</body>
</html>
"""


def render_voice_interface() -> None:
    """
    Render the voice/chat panel as an embedded HTML iframe inside Streamlit.

    The panel makes real API calls to /api/stt, /api/tts, and /api/chat on the
    FastAPI backend (same origin as the frontend served at /).  When the
    dashboard is run via the FastAPI+Streamlit setup described in app.py the
    backend is available at the same host.

    Height is set to 720px to show the full thread + input without a page
    scroll; adjust as needed.
    """
    st.markdown("### 🎙 Voice & Chat — Agent 1 Live Interface")
    st.caption(
        "Speak or type your query below. The panel calls the FastAPI backend directly "
        "(`/api/stt`, `/api/chat`, `/api/tts`). Make sure the FastAPI server is running "
        "(`uvicorn app:app --reload`) alongside this Streamlit dashboard."
    )

    components.html(VOICE_HTML, height=720, scrolling=False)

    st.divider()
    st.markdown(
        """
        **How this connects to the full pipeline:**
        1. Your query is sent to **`/api/chat`** → Agent 1 returns a `ScientificIntentOutput`.
        2. The parsed intent flows into the **Streamlit pipeline** — paste the same query
           in the sidebar and click **Run Analysis** to trigger Agents 2 & 3.
        3. Use the **🔊 speak** button on any agent bubble to hear the response via TTS
           (`/api/tts`).
        4. Switch to **Telugu / Hindi / any language** — the `/api/stt` endpoint detects
           it automatically and translates to English before passing to Agent 1.
        """
    )
