// Pruebas de lib/api.js con un fetch falso, con el ejecutor de pruebas de Node (npm test).
import test from 'node:test';
import assert from 'node:assert/strict';
import { getJson, postJson } from './api.js';

// Sustituye fetch por `fake` durante la prueba y lo restaura al terminar
async function withFetch(fake, run) {
  const original = globalThis.fetch;
  globalThis.fetch = fake;
  try {
    await run();
  } finally {
    globalThis.fetch = original;
  }
}

const answer = (status, body) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => {
    if (body === undefined) throw new SyntaxError("no es JSON");
    return body;
  }
});

test('la dirección se forma con la base y la ruta', async () => {
  let url;
  await withFetch(async (u) => { url = u; return answer(200, {}); }, async () => {
    await getJson('/api/stats');
  });
  assert.equal(url, 'http://localhost:8000/api/stats');
});

test('GET devuelve el JSON de la respuesta', async () => {
  await withFetch(async () => answer(200, { fps: 12 }), async () => {
    assert.deepEqual(await getJson('/api/stats'), { fps: 12 });
  });
});

test('GET lanza un error con un estado 500', async () => {
  await withFetch(async () => answer(500, { detail: "fallo" }), async () => {
    await assert.rejects(getJson('/api/stats'), /500/);
  });
});

test('POST envía el método, el tipo de contenido y el cuerpo en JSON', async () => {
  let call;
  await withFetch(async (u, options) => { call = { u, options }; return answer(200, { conf_threshold: 0.4 }); }, async () => {
    const result = await postJson('/api/settings', { conf: 0.4 });
    assert.deepEqual(result, { ok: true, status: 200, data: { conf_threshold: 0.4 } });
  });
  assert.equal(call.u, 'http://localhost:8000/api/settings');
  assert.equal(call.options.method, 'POST');
  assert.equal(call.options.headers['Content-Type'], 'application/json');
  assert.equal(call.options.body, JSON.stringify({ conf: 0.4 }));
});

test('POST con un 400 devuelve ok falso y el detalle del servidor, sin lanzar', async () => {
  await withFetch(async () => answer(400, { detail: "Cámara no válida" }), async () => {
    const result = await postJson('/api/cameras/select', { type: 'ip' });
    assert.equal(result.ok, false);
    assert.equal(result.status, 400);
    assert.equal(result.data.detail, "Cámara no válida");
  });
});

test('una respuesta que no es JSON da data vacío', async () => {
  await withFetch(async () => answer(502, undefined), async () => {
    const result = await postJson('/api/profiles/select', { id: 'x' });
    assert.deepEqual(result, { ok: false, status: 502, data: {} });
  });
});

test('GET se cancela con la señal que se le pasa', async () => {
  const controller = new AbortController();
  await withFetch(
    (u, { signal }) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason));
    }),
    async () => {
      const pending = getJson('/api/stats', { signal: controller.signal });
      controller.abort();
      await assert.rejects(pending);
    }
  );
});

test('GET lanza un error si la petición tarda más que timeoutMs', async () => {
  await withFetch(
    (u, { signal }) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason));
    }),
    async () => {
      await assert.rejects(getJson('/api/stats', { timeoutMs: 30 }), { name: 'TimeoutError' });
    }
  );
});

test('POST pasa una señal a fetch', async () => {
  let options;
  await withFetch(async (u, o) => { options = o; return answer(200, {}); }, async () => {
    await postJson('/api/settings', { conf: 0.4 });
  });
  assert.ok(options.signal instanceof AbortSignal);
});
