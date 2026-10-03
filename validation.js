// Lightweight client-side pre-checks. These are a fast first pass only —
// the authoritative validation always happens on the backend
// (backend/vision/quality.py), which has the real blur/exposure/reference
// detection logic. This just avoids uploading an obviously-unusable photo.
const NailSizeValidation = {
  MIN_WIDTH: 600,
  MIN_HEIGHT: 600,
  MAX_FILE_MB: 10,

  async checkFile(file) {
    if (!file.type.startsWith("image/")) {
      return { valid: false, message: "Please choose an image file." };
    }
    if (file.size > this.MAX_FILE_MB * 1024 * 1024) {
      return { valid: false, message: `Image is too large (max ${this.MAX_FILE_MB}MB).` };
    }
    const dims = await this._dimensions(file);
    if (!dims) {
      return { valid: false, message: "Couldn't read that image. Try another photo." };
    }
    if (dims.width < this.MIN_WIDTH || dims.height < this.MIN_HEIGHT) {
      return { valid: false, message: "Please use a higher-resolution photo." };
    }
    return { valid: true, dims };
  },

  _dimensions(file) {
    return new Promise((resolve) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        resolve({ width: img.naturalWidth, height: img.naturalHeight });
        URL.revokeObjectURL(url);
      };
      img.onerror = () => { resolve(null); URL.revokeObjectURL(url); };
      img.src = url;
    });
  },
};
