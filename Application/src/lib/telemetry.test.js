// Pruebas de las funciones puras del panel con el ejecutor de pruebas de Node (npm test), sin dependencias.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  hasAlarm, getSafetyColor, getCameraDisplayName, localDate, infractionTime, galleryCount
} from './telemetry.js';

test('la alarma sigue a la infracción confirmada, no a un fotograma suelto', () => {
  assert.equal(hasAlarm({ violations: 1, violation_confirmed: false }), false);
  assert.equal(hasAlarm({ violations: 1, violation_confirmed: true }), true);
  assert.equal(hasAlarm({ violations: 0, violation_confirmed: false }), false);
});

test('con un backend anterior, sin violation_confirmed, vale cualquier infracción', () => {
  assert.equal(hasAlarm({ violations: 2 }), true);
  assert.equal(hasAlarm({ violations: 0 }), false);
});

test('el color del cumplimiento cambia en 90 y en 60', () => {
  assert.equal(getSafetyColor(100), getSafetyColor(90));
  assert.equal(getSafetyColor(89), getSafetyColor(60));
  assert.equal(getSafetyColor(59), getSafetyColor(0));
  assert.equal(new Set([getSafetyColor(90), getSafetyColor(60), getSafetyColor(59)]).size, 3);
});

test('nombre de la cámara', () => {
  assert.equal(getCameraDisplayName(null), 'Cámara no seleccionada');
  assert.equal(getCameraDisplayName({ type: 'local', index: 2 }), 'Cámara Local 2');
  assert.equal(getCameraDisplayName({ type: 'local', index: 0, name: 'Portátil' }), 'Portátil');
  assert.equal(getCameraDisplayName({ type: 'ip', url: 'http://***@host/video' }), 'http://***@host/video');
  assert.equal(getCameraDisplayName({ type: 'ip', url: 'http://host/video', name: 'Mi Móvil' }), 'Mi Móvil');
});

test('la fecha local tiene el formato del backend', () => {
  assert.equal(localDate(new Date(2026, 0, 5, 23, 59)), '2026-01-05');
  assert.equal(localDate(new Date(2026, 10, 11, 0, 0)), '2026-11-11');
});

test('una infracción de otro día muestra la fecha', () => {
  const today = '2026-11-11';
  assert.equal(infractionTime({ date: '2026-11-11', timestamp: '12:30:05' }, today), '12:30:05');
  assert.equal(infractionTime({ date: '2026-11-10', timestamp: '12:30:05' }, today), '2026-11-10 12:30:05');
  assert.equal(infractionTime({ timestamp: '12:30:05' }, today), '12:30:05');  // Registro sin fecha
});

test('contador de la galería', () => {
  assert.equal(galleryCount(0), '');
  assert.equal(galleryCount(1), 'La más reciente');
  assert.equal(galleryCount(12), 'Las 12 más recientes');
});
