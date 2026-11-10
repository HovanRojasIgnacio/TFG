# GuardWatch AI — panel de control

Panel web (React 19 + Vite 8) del sistema de vigilancia de EPIs. Muestra el vídeo anotado, los contadores por elemento de protección y las infracciones guardadas, y permite cambiar el modo de detección, el umbral de confianza y la cámara. La instalación completa y el backend están descritos en el [README de la raíz](../README.md).

## Comandos

Desde esta carpeta, después de `npm install`:

| Comando | Qué hace |
|---|---|
| `npm run dev` | Servidor de desarrollo en `http://localhost:5173/TFG/`. |
| `npm run lint` | Comprueba el código con ESLint. |
| `npm run build` | Genera la versión de producción en `dist/`. |
| `npm run preview` | Sirve `dist/` en `http://localhost:4173/TFG/`. |
| `npm run deploy` | Compila y publica `dist/` en GitHub Pages. |

## Configuración

- **Backend:** por defecto `http://localhost:8000`. Para usar otro, crea `.env` con `VITE_API_BASE=http://direccion:8000` antes de `npm run dev` o `npm run build`.
- **Tiempos:** el intervalo de sondeo, el de la alarma y los tiempos máximos de las peticiones están en `src/config.js`.
- **Ruta base:** `/TFG/`, definida en `vite.config.js` para GitHub Pages.

## Estructura

- `src/App.jsx`: composición del panel.
- `src/components/`: vídeo, consola de control, métricas, galería de infracciones y diálogos.
- `src/hooks/`: telemetría (`/api/stats`), cámaras, modos de detección, umbral, alarma y foco de los diálogos.
- `src/config.js`: dirección del backend y constantes de tiempo.
