/**
 * FraudSentinel — SSE /stream, stats, transaction feed, SHAP bars, row-driven /explain.
 */

const MAX_ROWS = 50;
const RECONNECT_MS = 3000;

let totalCount = 0;
let fraudCount = 0;
let riskSum = 0;

/** @type {EventSource | null} */
let eventSource = null;
/** @type {ReturnType<typeof setTimeout> | null} */
let reconnectTimer = null;

/** @type {HTMLTableRowElement | null} */
let selectedRow = null;
let selectedTransaction = null;
const API_KEY = 'your_fraud_api_key_here';

function formatUsd(amount) {
  const n = Number(amount);
  if (!Number.isFinite(n)) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(n);
}

function formatTimeHMS(d = new Date()) {
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  const s = String(d.getSeconds()).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function formatClock() {
  const el = document.getElementById("live-clock");
  if (!el) return;
  const now = new Date();
  el.textContent = formatTimeHMS(now);
}

/**
 * @param {"live" | "reconnecting" | "idle"} mode
 */
function setConnectionStatus(mode) {
  const dot = document.getElementById("status-dot");
  if (!dot) return;
  const live = mode === "live";
  dot.classList.toggle("connected", live);
}

function riskClass(score) {
  const s = Number(score);
  if (!Number.isFinite(s)) return "";
  if (s < 40) return "risk-low";
  if (s <= 70) return "risk-mid";
  return "risk-high";
}

function getThreatLevel(score) {
  const s = Number(score);
  if (!Number.isFinite(s)) return "low";
  if (s < 20) return "low";
  if (s < 40) return "guarded";
  if (s < 60) return "elevated";
  if (s < 80) return "high";
  return "critical";
}

function updateThreatLevelBar(riskScore, containerId = "threat-level-bar") {
  const bar = document.getElementById(containerId);
  if (!bar) return;

  const segments = bar.querySelectorAll(".threat-segment");
  segments.forEach(seg => seg.classList.remove("active"));
  
  if (!Number.isFinite(Number(riskScore))) return;

  const s = Math.round(Number(riskScore) * 10) / 10;
  const level = getThreatLevel(s);
  
  // Map level name to data-level index
  const levelMap = { "low": 0, "guarded": 1, "elevated": 2, "high": 3, "critical": 4 };
  const levelIndex = levelMap[level] !== undefined ? levelMap[level] : 0;
  
  segments.forEach(seg => {
    const segLevel = parseInt(seg.getAttribute("data-level") || "0");
    if (segLevel <= levelIndex) {
      seg.classList.add("active");
    }
  });
}

function updateGauge(riskScore) {
  const bar = document.getElementById("threat-level-bar");
  const number = document.getElementById("gauge-risk");
  if (!bar || !number) return;

  if (!Number.isFinite(Number(riskScore))) {
    number.textContent = "—";
    return;
  }

  const s = Math.round(Number(riskScore) * 10) / 10;
  number.textContent = String(s);
  updateThreatLevelBar(s, "threat-level-bar");
}

function updateShapBars(shapFeatures) {
  const container = document.getElementById("shap-chart");
  if (!container) return;

  if (!Array.isArray(shapFeatures) || shapFeatures.length === 0) {
    container.innerHTML = "";
    return;
  }

  const items = shapFeatures
    .map((f) => ({
      name: String(f.feature || "").slice(0, 8).padEnd(8, " ").toUpperCase(),
      value: Number(f.value),
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10); // Top 10

  container.innerHTML = items.map(item => {
    const width = Math.abs(item.value) * 100; // Scale as needed
    const isNegative = item.value < 0;
    return `
      <div class="shap-feature">
        <div class="shap-name">${item.name}</div>
        <div class="shap-bar">
          <div class="shap-bar-fill ${isNegative ? 'negative' : ''}" style="width: ${width}%"></div>
        </div>
        <div class="shap-value">${item.value.toFixed(2)}</div>
      </div>
    `;
  }).join("");
}

function updateStats(payload) {
  totalCount += 1;
  const verdict = String(payload.verdict || "").toUpperCase();
  if (verdict === "FRAUD") fraudCount += 1;
  const risk = Number(payload.risk_score);
  if (Number.isFinite(risk)) riskSum += risk;

  const fraudPct = totalCount ? (100 * fraudCount) / totalCount : 0;
  const avgRisk = totalCount ? riskSum / totalCount : 0;

  const elTotal = document.getElementById("stat-total");
  const elFraud = document.getElementById("stat-fraud-count");
  const elPct = document.getElementById("stat-fraud-pct");
  const elAvg = document.getElementById("stat-avg-risk");
  const elLat = document.getElementById("stat-latency");
  const feedCount = document.getElementById("feed-count");

  if (elTotal) elTotal.textContent = String(totalCount);
  if (feedCount) feedCount.textContent = `(${totalCount})`;
  if (elFraud) elFraud.textContent = String(fraudCount);
  if (elPct) elPct.textContent = `(${fraudPct.toFixed(1)}%)`;
  if (elAvg) elAvg.textContent = avgRisk.toFixed(1);

  const ms = Number(payload.processing_ms);
  if (elLat) {
    elLat.textContent = Number.isFinite(ms) ? String(Math.round(ms)) : "—";
  }
}

function verdictCode(verdict) {
  const v = String(verdict || "").toUpperCase();
  if (v === "FRAUD") return "BLK";
  if (v === "REVIEW") return "OTP";
  return "APR";
}

function verdictClass(verdict) {
  const v = String(verdict || "").toUpperCase();
  if (v === "FRAUD") return "verdict-fraud";
  if (v === "REVIEW") return "verdict-review";
  return "verdict-approve";
}

function clearRowSelection() {
  if (selectedRow) {
    selectedRow.classList.remove("row-selected");
    selectedRow = null;
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function renderSimilarCases(cases) {
  const wrap = document.getElementById("explain-similar");
  if (!wrap) return;
  if (!Array.isArray(cases) || cases.length === 0) {
    wrap.innerHTML = "<div class=\"similar-empty\">No similar cases available.</div>";
    return;
  }
  const items = cases
    .filter((c) => c != null && String(c).trim() !== "")
    .slice(0, 3)
    .map((raw, index) => {
      const parts = String(raw).split("|").map((p) => p.trim());
      const caseId = parts[0] || `CASE-${String(index + 1).padStart(3, "0")}`;
      const caseType = (parts[1] || "RAG CASE").replace(/_/g, " ").toUpperCase();
      const description = parts.slice(2).join(" | ") || "No description available.";
      return `
        <div class="similar-case">
          <div class="similar-header">
            <span class="case-id">${escapeHtml(caseId)}</span>
            <span class="case-type">${escapeHtml(caseType)}</span>
          </div>
          <div class="case-desc">${escapeHtml(description)}</div>
        </div>
      `;
    })
    .join("");
  wrap.innerHTML = items;
}

function renderGraphInsights(insights) {
  const wrap = document.getElementById("explain-graph");
  if (!wrap) return;
  if (!Array.isArray(insights) || insights.length === 0) {
    wrap.innerHTML = "<div class=\"similar-empty\">No relational anomalies detected.</div>";
    return;
  }
  const items = insights
    .map((raw) => {
      return `
        <div class="graph-insight">
          <span class="insight-icon">🔗</span>
          <span class="insight-text">${escapeHtml(String(raw))}</span>
        </div>
      `;
    })
    .join("");
  wrap.innerHTML = items;
}

async function postExplainForPayload(payload) {
  const box = document.getElementById("explain-text");
  const similarEl = document.getElementById("explain-similar");
  const graphEl = document.getElementById("explain-graph");
  if (!box) return;
  box.classList.add("loading");
  currentExplanationText = "Loading explanation...";
  box.textContent = currentExplanationText;
  if (similarEl) similarEl.innerHTML = "";
  if (graphEl) graphEl.innerHTML = "";

  const body = {
    transaction_id: String(payload.transaction_id || ""),
    risk_score: Number(payload.risk_score),
    shap_features: Array.isArray(payload.shap_features)
      ? payload.shap_features
      : [],
    amount: Number(payload.amount),
    hour: Number(payload.hour),
    merchant_category:
      payload.merchant_category != null ? String(payload.merchant_category) : "",
    features: payload.features || {},
  };
  try {
    const res = await fetch("/api/investigate/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      box.classList.remove("loading");
      if (res.status === 429) {
        box.textContent = "Rate limit reached. Please wait 60 seconds.";
      } else {
        box.textContent = `Request failed with status ${res.status}`;
      }
      return;
    }
    box.textContent = "";
    box.classList.remove("loading");
    box.classList.add("streaming");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    currentExplanationText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'metadata') {
            renderSimilarCases(data.similar_cases);
            renderGraphInsights(data.graph_insights);
          }
          if (data.token) {
             currentExplanationText += data.token;
             box.textContent = currentExplanationText;
          }
          if (data.done) {
             box.classList.remove("streaming");
          }
        } catch (_) {}
      }
    }
  } catch {
    box.classList.remove("loading");
    box.classList.remove("streaming");
    currentExplanationText = "Explanation request failed. The API may be unreachable or the Gemini API may be offline.";
    box.textContent = currentExplanationText;
    renderSimilarCases([]);
    renderGraphInsights([]);
  }
}

function renderSelectedTransaction() {
  const defaultPanel = document.getElementById("default-panel");
  const analystPanel = document.getElementById("analyst-panel");
  const amountField = document.getElementById("alert-amount");
  const merchantField = document.getElementById("alert-merchant");
  const hourField = document.getElementById("alert-hour");
  const riskField = document.getElementById("alert-risk");
  const openButton = document.getElementById("open-analysis-btn");
  const verdictBadge = document.querySelector(".verdict-badge");
  const headerTxnId = document.querySelector(".header-txn-id");
  const explainText = document.getElementById("explain-text");

  if (!defaultPanel || !analystPanel) return;

  if (!selectedTransaction) {
    defaultPanel.classList.remove("hidden");
    analystPanel.classList.add("hidden");
    return;
  }

  defaultPanel.classList.add("hidden");
  analystPanel.classList.remove("hidden");

  const verdict = String(selectedTransaction.verdict || "").toUpperCase();
  
  // Update verdict badge
  if (verdictBadge) {
    verdictBadge.textContent = verdict === "FRAUD" ? "FRAUD" : verdict === "REVIEW" ? "REVIEW" : "APPROVE";
    verdictBadge.classList.remove("fraud", "review", "approve");
    if (verdict === "FRAUD") {
      verdictBadge.classList.add("fraud");
    } else if (verdict === "REVIEW") {
      verdictBadge.classList.add("review");
    } else {
      verdictBadge.classList.add("approve");
    }
  }

  // Update transaction ID
  if (headerTxnId) {
    headerTxnId.textContent = String(selectedTransaction.transaction_id);
  }

  // Format and display fields
  const amount = Number(selectedTransaction.amount);
  const formattedAmount = "$" + amount.toFixed(2);
  
  if (amountField) amountField.textContent = formattedAmount;
  if (merchantField) merchantField.textContent = String(selectedTransaction.merchant_category || "—");
  if (hourField) {
    const hour = String(selectedTransaction.hour).padStart(2, "0");
    hourField.textContent = `${hour}:00`;
  }
  if (riskField) {
    const riskVal = Number(selectedTransaction.risk_score);
    riskField.textContent = riskVal.toFixed(1) + " / 100";
  }

  // Update explanation
  if (explainText) {
    explainText.textContent = "Querying Google Gemini ...";
    explainText.classList.add("loading");
  }

  // Update gauge and SHAP bars
  updateGauge(selectedTransaction.risk_score);
  updateShapBars(selectedTransaction.shap_features);
  
  if (openButton) openButton.classList.remove("hidden");
}

let currentExplanationText = "";

function openExplanationModal() {
  const modal = document.getElementById("explanation-modal");
  const modalText = document.getElementById("modal-explanation-text");
  if (!modal || !modalText) return;
  
  if (currentExplanationText) {
    modalText.textContent = currentExplanationText;
  } else {
    modalText.textContent = "Explanation not available.";
  }
  
  modal.classList.remove("hidden");
}

function closeExplanationModal() {
  const modal = document.getElementById("explanation-modal");
  if (modal) {
    modal.classList.add("hidden");
  }
}

// Close modal on escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeExplanationModal();
});

function openFullAnalysis() {
  if (!selectedTransaction) return;
  const params = new URLSearchParams({
    txn_id: String(selectedTransaction.transaction_id || ""),
    score: String(selectedTransaction.risk_score || ""),
    amount: String(selectedTransaction.amount || ""),
    verdict: String(selectedTransaction.verdict || ""),
    merchant: String(selectedTransaction.merchant_category || ""),
    hour: String(selectedTransaction.hour || ""),
  });
  sessionStorage.setItem("fraudsentinel_selected_transaction", JSON.stringify(selectedTransaction));
  window.open(`/analyst?${params.toString()}`, "_blank");
}

function onFeedRowClick(tr) {
  /** @type {Record<string, unknown> | undefined} */
  const payload = tr._fraudPayload;
  if (!payload) return;
  const v = String(payload.verdict || "").toUpperCase();
  if (v !== "FRAUD" && v !== "REVIEW") return;

  clearRowSelection();
  selectedRow = tr;
  selectedTransaction = payload;
  tr.classList.add("row-selected");
  renderSelectedTransaction();
  void postExplainForPayload(payload);
}

function wireFeedClickDelegation() {
  const tbody = document.getElementById("feed-body");
  if (!tbody) return;
  tbody.addEventListener("click", (ev) => {
    const tr = ev.target.closest("tr");
    if (!tr || tr.parentElement !== tbody) return;
    onFeedRowClick(tr);
  });
}

function prependFeedRow(payload) {
  const tbody = document.getElementById("feed-body");
  if (!tbody) return;
  const tr = document.createElement("tr");
  const timeStr = formatTimeHMS(new Date());
  const merchant =
    payload.merchant_category != null
      ? String(payload.merchant_category)
      : "—";
  const hour =
    payload.hour != null && payload.hour !== ""
      ? String(payload.hour)
      : "—";
  const risk =
    payload.risk_score != null && Number.isFinite(Number(payload.risk_score))
      ? Number(payload.risk_score).toFixed(1)
      : "—";
  const verdict = String(payload.verdict || "—");
  const vUpper = verdict.toUpperCase();

  tr._fraudPayload = payload;
  tr.classList.add("row-enter");
  if (vUpper === "FRAUD") {
    tr.classList.add("row-fraud");
  } else if (vUpper === "REVIEW") {
    tr.classList.add("row-review");
  } else {
    tr.classList.add("row-approve");
  }
  if (vUpper === "FRAUD" || vUpper === "REVIEW") {
    tr.setAttribute("role", "button");
    tr.setAttribute("tabindex", "0");
    tr.style.cursor = "pointer";
    tr.title = "Click to load analyst explanation and similar cases";
  } else {
    tr.style.cursor = "default";
  }

  const riskBarWidth = Number.isFinite(Number(payload.risk_score)) ? Math.min(Number(payload.risk_score), 100) : 0;

  tr.innerHTML = `
    <td>${escapeHtml(String(payload.transaction_id || "—"))}</td>
    <td>${escapeHtml(formatUsd(payload.amount))}</td>
    <td>${escapeHtml(merchant)}</td>
    <td>${escapeHtml(hour)}</td>
    <td>
      <div class="risk-bar">
        <div class="risk-bar-fill" style="width: ${riskBarWidth}%"></div>
      </div>
      ${escapeHtml(risk)}
    </td>
    <td><span class="verdict-code ${verdictClass(verdict)}">${verdictCode(verdict)}</span></td>
    <td>${escapeHtml(timeStr)}</td>
  `;
  tbody.insertBefore(tr, tbody.firstChild);
  tr.addEventListener(
    "animationend",
    () => {
      tr.classList.remove("row-enter");
    },
    { once: true },
  );
  if (vUpper === "FRAUD" || vUpper === "REVIEW") {
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onFeedRowClick(tr);
      }
    });
  }
  while (tbody.rows.length > MAX_ROWS) {
    const last = tbody.lastChild;
    if (last === selectedRow) clearRowSelection();
    tbody.removeChild(last);
  }
}

function handleStreamPayload(payload) {
  updateStats(payload);
  prependFeedRow(payload);
  if (!selectedTransaction) {
    updateGauge(payload.risk_score);
    updateShapBars(payload.shap_features);
  }
}

function scheduleReconnect() {
  if (reconnectTimer != null) return;
  setConnectionStatus("reconnecting");
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    openEventSource();
  }, RECONNECT_MS);
}

function openEventSource() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  const controller = new AbortController();
  const url = `${window.location.origin}/stream`;
  const signal = controller.signal;
  let buffer = "";

  eventSource = {
    close: () => controller.abort(),
  };

  fetch(url, {
    method: "GET",
    headers: {
      "X-API-Key": API_KEY,
    },
    signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        if (response.status === 429) {
          throw new Error("Rate limit reached. Please wait 60 seconds.");
        }
        throw new Error(`Stream request failed: ${response.status}`);
      }
      if (reconnectTimer != null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      setConnectionStatus("live");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.error) {
              console.warn("Stream payload error:", payload.error);
              continue;
            }
            handleStreamPayload(payload);
          } catch (e) {
            console.warn("Invalid SSE JSON", e);
          }
        }
      }
      scheduleReconnect();
    })
    .catch((error) => {
      if (signal.aborted) return;
      console.warn("Stream fetch error:", error);
      scheduleReconnect();
    });
}

function bootstrapDashboard() {
  formatClock();
  setInterval(formatClock, 1000);
  setConnectionStatus("idle");

  wireFeedClickDelegation();
  openEventSource();
}

window.addEventListener("DOMContentLoaded", bootstrapDashboard);
