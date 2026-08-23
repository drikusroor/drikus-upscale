/*
 * End-to-end smoke test: serves the repository, drives headless Chromium
 * through real upscales and checks output size, alpha handling and tile seams.
 *
 *   npm install && node tools/smoke-test.mjs
 *
 * Set CHROME_PATH to use a specific Chromium build.
 */
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURES = path.join(ROOT, 'tools', 'fixtures');
const PORT = Number(process.env.PORT || 8099);
const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
  '.jpg': 'image/jpeg', '.png': 'image/png', '.json': 'application/json',
};

const server = http.createServer((req, res) => {
  let file = decodeURIComponent(req.url.split('?')[0]);
  if (file === '/') file = '/index.html';
  const full = path.join(ROOT, file);
  if (!full.startsWith(ROOT) || !fs.existsSync(full) || fs.statSync(full).isDirectory()) {
    res.writeHead(404);
    return res.end('not found');
  }
  res.writeHead(200, {
    'Content-Type': TYPES[path.extname(full)] || 'application/octet-stream',
    'Content-Length': fs.statSync(full).size,
  });
  fs.createReadStream(full).pipe(res);
});
await new Promise((resolve) => server.listen(PORT, resolve));

const browser = await chromium.launch(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {});
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
page.on('pageerror', (err) => console.log('  [pageerror]', err.message));
page.on('console', (msg) => { if (msg.type() === 'error') console.log('  [console]', msg.text()); });

await page.goto(`http://localhost:${PORT}/`);
await page.waitForTimeout(2500);   // the COI service worker reloads the page once
console.log('cross-origin isolated:', await page.evaluate(() => window.crossOriginIsolated));
console.log('backend chip:', (await page.textContent('#caps-chip')).trim());

const failures = [];
const check = (name, ok, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(name);
};

async function runCase({ fixture, model, scale, tile, label, expect }) {
  console.log(`\n${label}`);
  await page.setInputFiles('#file-input', path.join(FIXTURES, fixture));
  await page.waitForSelector('#workspace:not([hidden])');
  await page.selectOption('#model-select', model);
  await page.selectOption('#scale-select', scale);
  await page.evaluate((value) => {
    const slider = document.getElementById('tile-size');
    slider.value = String(value);
    slider.dispatchEvent(new Event('input'));
    slider.dispatchEvent(new Event('change'));
  }, tile);

  const started = Date.now();
  await page.click('#run');
  await page.waitForFunction(() => {
    const status = document.getElementById('status').textContent || '';
    const error = document.getElementById('error');
    return status.startsWith('Done in') || (error && !error.hidden);
  }, null, { timeout: 900000 });

  const error = await page.evaluate(() => {
    const node = document.getElementById('error');
    return node.hidden ? null : node.textContent;
  });
  check(`${label}: completed without error`, !error, error || undefined);
  if (error) return;

  const stats = await page.evaluate((tileSize) => {
    const canvas = document.getElementById('canvas-after');
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    const at = (x, y, channel) => data[(y * canvas.width + x) * 4 + channel];
    let boundarySum = 0, boundaryCount = 0, allSum = 0, allCount = 0;
    let minAlpha = 255, maxAlpha = 0;
    for (let y = 1; y < canvas.height - 1; y++) {
      for (let x = 1; x < canvas.width - 1; x++) {
        const gradient = Math.abs(at(x, y, 0) - at(x - 1, y, 0));
        allSum += gradient; allCount++;
        if (x % (tileSize * 4) === 0) { boundarySum += gradient; boundaryCount++; }
        const alpha = at(x, y, 3);
        if (alpha < minAlpha) minAlpha = alpha;
        if (alpha > maxAlpha) maxAlpha = alpha;
      }
    }
    return {
      width: canvas.width, height: canvas.height,
      seam: boundaryCount ? boundarySum / boundaryCount : 0,
      mean: allSum / allCount,
      minAlpha, maxAlpha,
    };
  }, tile);

  console.log(`  ${((Date.now() - started) / 1000).toFixed(1)} s · ${stats.width}×${stats.height} · seam ${stats.seam.toFixed(2)} vs mean ${stats.mean.toFixed(2)}`);
  check(`${label}: output dimensions`, stats.width === expect.width && stats.height === expect.height,
    `${stats.width}×${stats.height}, wanted ${expect.width}×${expect.height}`);
  check(`${label}: produced non-blank pixels`, stats.mean > 0.5, `mean gradient ${stats.mean.toFixed(2)}`);
  check(`${label}: no visible tile seams`, stats.seam <= stats.mean * 1.5,
    `seam ${stats.seam.toFixed(2)} vs mean ${stats.mean.toFixed(2)}`);
  if (expect.alpha) {
    check(`${label}: alpha channel preserved`, stats.minAlpha === 0 && stats.maxAlpha === 255,
      `alpha range ${stats.minAlpha}..${stats.maxAlpha}`);
  }
}

await runCase({
  fixture: 'lowres-160x120.jpg', model: 'realesr-general-x4v3-balanced', scale: '2', tile: 96,
  label: 'general/balanced, 2× output', expect: { width: 320, height: 240 },
});
await runCase({
  fixture: 'alpha-400x260.png', model: 'realesrgan-x4plus-anime-6b', scale: '4', tile: 128,
  label: 'anime 6B, transparency, 12 tiles', expect: { width: 1600, height: 1040, alpha: true },
});
await runCase({
  fixture: 'lowres-160x120.jpg', model: 'realesrgan-x4plus', scale: '1', tile: 160,
  label: 'x4plus heavy, 1× cleanup', expect: { width: 160, height: 120 },
});

await browser.close();
server.close();
console.log(failures.length ? `\n${failures.length} check(s) failed.` : '\nAll checks passed.');
process.exit(failures.length ? 1 : 0);
