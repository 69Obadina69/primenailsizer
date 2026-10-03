function readHandResult(hand) {
  const errRaw = sessionStorage.getItem(`ns_error_${hand}`);
  const resRaw = sessionStorage.getItem(`ns_result_${hand}`);
  if (errRaw) return { hand, error: JSON.parse(errRaw) };
  if (resRaw) return { hand, data: JSON.parse(resRaw) };
  return { hand, error: { code: "MISSING", message: `No ${hand} hand photo was processed.` } };
}

function fingerLabel(f) {
  return f.charAt(0).toUpperCase() + f.slice(1);
}

function renderHandSection(result) {
  let html = `<div class="hand-section-title">${result.hand === "left" ? "Left Hand" : "Right Hand"}</div>`;

  if (result.error) {
    html += `<div class="error-banner">
      <strong>We couldn't measure this hand</strong>
      ${escapeHtml(result.error.message)}
    </div>`;
    return html;
  }

  const order = ["thumb", "index", "middle", "ring", "pinky"];
  const nailsByFinger = {};
  result.data.nails.forEach(n => { nailsByFinger[n.finger] = n; });

  order.forEach(f => {
    const n = nailsByFinger[f];
    if (!n) return;
    const lowConf = n.confidence < 0.6;
    html += `
      <div class="nail-row ${lowConf ? "conf-low" : ""}">
        <div>
          <div class="finger-name">${fingerLabel(f)}</div>
          <div class="finger-meta">${n.width_mm.toFixed(1)} mm width &middot; ${n.length_mm.toFixed(1)} mm length &middot; ±${n.uncertainty_mm} mm &middot; ${Math.round(n.confidence * 100)}% confidence</div>
        </div>
        <div class="size-pill">${n.recommended_size ? n.recommended_size.label : "—"}</div>
      </div>`;
  });

  return html;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function buildCopyText(left, right) {
  const lines = ["NailSizer — Your Recommended Nail Sizes", ""];
  [["Left Hand", left], ["Right Hand", right]].forEach(([label, result]) => {
    lines.push(label + ":");
    if (result.error) {
      lines.push(`  Could not measure — ${result.error.message}`);
    } else {
      ["thumb", "index", "middle", "ring", "pinky"].forEach(f => {
        const n = result.data.nails.find(x => x.finger === f);
        if (n) lines.push(`  ${fingerLabel(f)}: ${n.recommended_size.label} (${n.width_mm.toFixed(1)}mm ± ${n.uncertainty_mm}mm)`);
      });
    }
    lines.push("");
  });
  return lines.join("\n");
}

function init() {
  const left = readHandResult("left");
  const right = readHandResult("right");

  document.getElementById("trial-banner").textContent =
    `You have ${NailSizeTrial.remaining()} free scans remaining.`;

  const content = document.getElementById("results-content");
  content.innerHTML = renderHandSection(left) + renderHandSection(right);

  const actions = document.getElementById("results-actions");
  actions.innerHTML = `
    <button class="btn btn-primary" id="btn-copy">Copy Results</button>
    <button class="btn btn-secondary" id="btn-pdf">Download PDF</button>
    <button class="btn btn-secondary" id="btn-share">Share</button>
    <button class="btn btn-ghost" id="btn-again">Measure Again</button>
  `;

  document.getElementById("btn-copy").addEventListener("click", async () => {
    const text = buildCopyText(left, right);
    try {
      await navigator.clipboard.writeText(text);
      flashButton("btn-copy", "Copied!");
    } catch {
      alert(text);
    }
  });

  document.getElementById("btn-pdf").addEventListener("click", () => {
    // Uses the browser's native print-to-PDF rather than bundling a PDF
    // library, so it works offline with zero extra dependencies.
    window.print();
  });

  document.getElementById("btn-share").addEventListener("click", async () => {
    const text = buildCopyText(left, right);
    if (navigator.share) {
      try { await navigator.share({ title: "My NailSizer Results", text }); } catch {}
    } else {
      await navigator.clipboard.writeText(text);
      flashButton("btn-share", "Copied to share!");
    }
  });

  document.getElementById("btn-again").addEventListener("click", () => {
    sessionStorage.removeItem("ns_reference_type");
    sessionStorage.removeItem("ns_reference_id");
    sessionStorage.removeItem("ns_custom_diameter_mm");
    sessionStorage.removeItem("ns_custom_width_mm");
    sessionStorage.removeItem("ns_custom_height_mm");
    location.href = "camera.html?hand=left";
  });
}

function flashButton(id, text) {
  const btn = document.getElementById(id);
  const original = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = original; }, 1500);
}

init();
