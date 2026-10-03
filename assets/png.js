/*
 * Minimal 8-bit greyscale PNG encoder for mask downloads.
 *
 * canvas.toBlob always writes RGB(A), so a mask would come out three or four
 * times larger than it needs to be and not as a true single-channel image.
 * CompressionStream('deflate') emits zlib-wrapped deflate, which is exactly
 * what a PNG IDAT chunk holds, so the browser does the heavy lifting.
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes, crc = 0xffffffff) {
  for (let i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return crc;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, (crc32(out.subarray(4, 8 + data.length)) ^ 0xffffffff) >>> 0);
  return out;
}

/** Encode `grey` (width × height bytes) as a colour-type-0, 8-bit PNG Blob. */
export async function encodeGreyPng(grey, width, height) {
  // Each scanline gets filter type 2 ("Up"): masks are mostly runs of 0 and
  // 255 that repeat row to row, which Up turns into long runs of zeros.
  const raw = new Uint8Array((width + 1) * height);
  for (let y = 0; y < height; y++) {
    const o = y * (width + 1);
    const s = y * width;
    raw[o] = 2;
    for (let x = 0; x < width; x++) {
      raw[o + 1 + x] = grey[s + x] - (y ? grey[s - width + x] : 0);
    }
  }
  const compressed = new Uint8Array(await new Response(
    new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate')),
  ).arrayBuffer());

  const header = new Uint8Array(13);
  const view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header[8] = 8;    // bit depth
  header[9] = 0;    // colour type: greyscale
  const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob([signature, chunk('IHDR', header), chunk('IDAT', compressed), chunk('IEND', new Uint8Array())],
    { type: 'image/png' });
}
