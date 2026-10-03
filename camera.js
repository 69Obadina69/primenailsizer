// Reads which hand we're capturing from the query string (?hand=left|right),
// walks through: choose reference -> instructions -> capture -> processing
// -> then either the other hand or results.html.

const params = new URLSearchParams(location.search);
const HAND = params.get("hand") || "left";
const HAND_LABEL = HAND === "left" ? "Left" : "Right";

const state = {
  referenceType: sessionStorage.getItem("ns_reference_type") || null,
  referenceId: sessionStorage.getItem("ns_reference_id") || null,
  customDiameterMm: sessionStorage.getItem("ns_custom_diameter_mm") || null,
  customWidthMm: sessionStorage.getItem("ns_custom_width_mm") || null,
  customHeightMm: sessionStorage.getItem("ns_custom_height_mm") || null,
  file: null,
};

document.getElementById("ref-hand-title").textContent = `Choose Your Reference`;
document.getElementById("instr-title").textContent = `Upload ${HAND_LABEL} Hand`;
document.getElementById("capture-title").textContent = `Upload ${HAND_LABEL} Hand`;

function goToPanel(id) {
  document.querySelectorAll(".screen").forEach(el => el.classList.add("hidden"));
  document.getElementById(id).classList.remove("hidden");
  window.scrollTo(0, 0);
}

// If reference was already chosen (e.g. we're on the right-hand pass),
// skip straight to instructions.
if (state.referenceType) {
  goToPanel("panel-instructions");
} else {
  goToPanel("panel-reference");
}

// ---- Reference selection ----
const refOptions = document.querySelectorAll(".reference-option");
const refExtraFields = document.getElementById("ref-extra-fields");
const refContinueBtn = document.getElementById("ref-continue");

function renderExtraFields(refType) {
  refExtraFields.innerHTML = "";
  if (refType === "coin") {
    refExtraFields.innerHTML = `
      <select id="coin-select">
        <option value="">Loading coins…</option>
      </select>`;
    NailSizeAPI.references().then(data => {
      const sel = document.getElementById("coin-select");
      if (!sel) return;
      sel.innerHTML = data.coins.map(c =>
        `<option value="${c.id}">${c.name}${c.country !== "Any" ? " — " + c.country : ""}</option>`
      ).join("");
      sel.addEventListener("change", checkRefReady);
      checkRefReady();
    }).catch(() => {
      refExtraFields.innerHTML = `<p class="subtitle">Couldn't load coin list. Check your connection.</p>`;
    });
  } else if (refType === "bottle_cap") {
    refExtraFields.innerHTML = `
      <input type="number" step="0.1" id="cap-diameter" placeholder="Bottle cap diameter (mm)">`;
    document.getElementById("cap-diameter").addEventListener("input", checkRefReady);
  } else if (refType === "custom") {
    refExtraFields.innerHTML = `
      <input type="number" step="0.1" id="custom-width" placeholder="Reference width (mm)">
      <input type="number" step="0.1" id="custom-height" placeholder="Reference height (mm)">`;
    document.getElementById("custom-width").addEventListener("input", checkRefReady);
    document.getElementById("custom-height").addEventListener("input", checkRefReady);
  }
}

function checkRefReady() {
  const selected = document.querySelector(".reference-option.selected");
  if (!selected) { refContinueBtn.disabled = true; return; }
  const refType = selected.dataset.ref;
  let ready = true;
  if (refType === "coin") {
    const sel = document.getElementById("coin-select");
    ready = !!(sel && sel.value);
  } else if (refType === "bottle_cap") {
    const el = document.getElementById("cap-diameter");
    ready = !!(el && parseFloat(el.value) > 0);
  } else if (refType === "custom") {
    const w = document.getElementById("custom-width");
    const h = document.getElementById("custom-height");
    ready = !!(w && h && parseFloat(w.value) > 0 && parseFloat(h.value) > 0);
  }
  refContinueBtn.disabled = !ready;
}

refOptions.forEach(opt => {
  opt.addEventListener("click", () => {
    refOptions.forEach(o => o.classList.remove("selected"));
    opt.classList.add("selected");
    renderExtraFields(opt.dataset.ref);
    checkRefReady();
  });
});

refContinueBtn.addEventListener("click", () => {
  const selected = document.querySelector(".reference-option.selected");
  const refType = selected.dataset.ref;
  sessionStorage.setItem("ns_reference_type", refType);
  if (refType === "coin") {
    sessionStorage.setItem("ns_reference_id", document.getElementById("coin-select").value);
  } else if (refType === "bottle_cap") {
    sessionStorage.setItem("ns_custom_diameter_mm", document.getElementById("cap-diameter").value);
  } else if (refType === "custom") {
    sessionStorage.setItem("ns_custom_width_mm", document.getElementById("custom-width").value);
    sessionStorage.setItem("ns_custom_height_mm", document.getElementById("custom-height").value);
  }
  goToPanel("panel-instructions");
});

// ---- Capture / upload ----
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const uploadBox = document.getElementById("upload-box");
const captureContinueBtn = document.getElementById("capture-continue");
const captureError = document.getElementById("capture-error");

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  captureError.classList.add("hidden");

  const result = await NailSizeValidation.checkFile(file);
  if (!result.valid) {
    captureError.textContent = result.message;
    captureError.classList.remove("hidden");
    captureContinueBtn.disabled = true;
    return;
  }

  state.file = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.remove("hidden");
  captureContinueBtn.disabled = false;
});

const PROCESSING_MESSAGES = [
  "Uploading…", "Detecting hand…", "Finding nails…", "Detecting reference…",
  "Calibrating measurements…", "Calculating nail widths…", "Matching nail sizes…", "Almost done…",
];

function runProcessingAnimation() {
  const msgEl = document.getElementById("processing-message");
  const fillEl = document.getElementById("progress-fill");
  let i = 0;
  msgEl.textContent = PROCESSING_MESSAGES[0];
  fillEl.style.width = "8%";
  const interval = setInterval(() => {
    i = Math.min(i + 1, PROCESSING_MESSAGES.length - 1);
    msgEl.textContent = PROCESSING_MESSAGES[i];
    fillEl.style.width = `${Math.min(100, (i + 1) / PROCESSING_MESSAGES.length * 100)}%`;
  }, 550);
  return () => clearInterval(interval);
}

captureContinueBtn.addEventListener("click", async () => {
  if (!state.file) return;
  goToPanel("panel-processing");
  const stopAnim = runProcessingAnimation();

  const { ok, body } = await NailSizeAPI.measure({
    imageBlob: state.file,
    referenceType: sessionStorage.getItem("ns_reference_type"),
    referenceId: sessionStorage.getItem("ns_reference_id"),
    customDiameterMm: sessionStorage.getItem("ns_custom_diameter_mm"),
    customWidthMm: sessionStorage.getItem("ns_custom_width_mm"),
    customHeightMm: sessionStorage.getItem("ns_custom_height_mm"),
  });

  stopAnim();

  if (!ok || body.success === false) {
    sessionStorage.setItem(`ns_error_${HAND}`, JSON.stringify(body.error || { code: "UNKNOWN", message: "Something went wrong." }));
    sessionStorage.removeItem(`ns_result_${HAND}`);
  } else {
    NailSizeTrial.increment();
    sessionStorage.setItem(`ns_result_${HAND}`, JSON.stringify(body));
    sessionStorage.removeItem(`ns_error_${HAND}`);
  }

  if (HAND === "left") {
    location.href = "camera.html?hand=right";
  } else {
    location.href = "results.html";
  }
});
