function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  document.querySelector(`[data-tab="${name}"]`)?.classList.add("active");
  document.getElementById(`panel-${name}`)?.classList.add("active");
}

async function loadMetrics() {
  try {
    const data = await apiFetch("/metrics");
    document.getElementById("thresholds-body").innerHTML = data.thresholds
      .map(
        (r) =>
          `<tr><td>${r.threshold}</td><td>${r.sensitivity}</td><td>${r.specificity}</td><td>${r.fn}</td><td>${r.fp}</td></tr>`
      )
      .join("");
    document.getElementById("roc-auc").textContent = data.roc_auc;
    document.getElementById("accuracy").textContent =
      (data.test_accuracy * 100).toFixed(1) + "%";
  } catch (err) {
    console.error(err);
  }
}

async function runPrediction(e) {
  e.preventDefault();
  const resultEl = document.getElementById("result");
  resultEl.className = "result-box";
  resultEl.innerHTML = "<p style='color:var(--text-muted)'>Analyzing…</p>";

  const body = {
    concave_points_worst: parseFloat(document.getElementById("concave_points_worst").value),
    radius_worst: parseFloat(document.getElementById("radius_worst").value),
    area_worst: parseFloat(document.getElementById("area_worst").value),
    concave_points_mean: parseFloat(document.getElementById("concave_points_mean").value),
    perimeter_worst: parseFloat(document.getElementById("perimeter_worst").value),
    concavity_mean: parseFloat(document.getElementById("concavity_mean").value),
    texture_worst: parseFloat(document.getElementById("texture_worst").value),
    smoothness_worst: parseFloat(document.getElementById("smoothness_worst").value),
    symmetry_worst: parseFloat(document.getElementById("symmetry_worst").value),
  };

  try {
    const data = await apiFetch("/predict", { method: "POST", body: JSON.stringify(body) });
    resultEl.className = `result-box analyzed ${data.level}`;
    resultEl.innerHTML = `
      <div class="result-pct">${data.probability_percent}%</div>
      <div class="result-level ${data.level}">${data.level}</div>
      <p style="margin-top:0.75rem;color:var(--text-muted);font-size:0.9rem">${data.message}</p>
      <p style="margin-top:0.5rem;font-size:0.8rem;color:var(--text-muted)">Not a clinical diagnosis.</p>
    `;
  } catch (err) {
    resultEl.innerHTML = `<p class="error">${err.message}</p>`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!requireAuth()) return;
  document.getElementById("user-label").textContent = getUsername() || "User";
  document.getElementById("logout-btn").addEventListener("click", handleLogout);
  document.getElementById("predict-form").addEventListener("submit", runPrediction);
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => showTab(tab.dataset.tab));
  });
  loadMetrics();
});
