   // Thin wrapper around the backend API. Centralizing this makes it a
// one-line change to point at a different host in production.
const API_BASE = window.location.origin.includes(":") && window.location.port
  ? "" // same-origin, backend served alongside frontend or via proxy
  : "";

const NailSizeAPI = {
  async health() {
    const r = await fetch(`${API_BASE}/api/v1/health`);
    return r.json();
  },

  async references() {
    const r = await fetch(`${API_BASE}/api/v1/references`);
    return r.json();
  },

  async measure({ imageBlob, referenceType, referenceId, customDiameterMm, customWidthMm, customHeightMm }) {
    const fd = new FormData();
    fd.append("image", imageBlob, "photo.jpg");
    fd.append("reference_type", referenceType);
    if (referenceId) fd.append("reference_id", referenceId);
    if (customDiameterMm) fd.append("custom_reference_diameter_mm", customDiameterMm);
    if (customWidthMm) fd.append("custom_reference_width_mm", customWidthMm);
    if (customHeightMm) fd.append("custom_reference_height_mm", customHeightMm);

    const r = await fetch(`${API_BASE}/api/v1/measure`, { method: "POST", body: fd });
    const body = await r.json();
    return { ok: r.ok, status: r.status, body };
  },
};

// Trial-count tracking (local for MVP; structured so it can move server-side
// once accounts exist — see NailSizeTrial.migrateToAccount()).
const NailSizeTrial = {
  KEY: "nailsize_trial_scans_used",
  LIMIT: 10,

  used() {
    return parseInt(localStorage.getItem(this.KEY) || "0", 10);
  },
  remaining() {
    return Math.max(0, this.LIMIT - this.used());
  },
  increment() {
    localStorage.setItem(this.KEY, String(this.used() + 1));
  },
  isLocked() {
    return this.remaining() <= 0;
  },
};
