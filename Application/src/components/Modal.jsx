import { useEffect, useRef } from 'react';

/** Diálogo modal accesible basado en <dialog>: el navegador gestiona el foco, Escape y el fondo. */
export default function Modal({ label, onClose, className = '', children }) {
  const dialogRef = useRef(null);
  // Solo cierra un clic que empieza y termina en el fondo. Si se arrastra desde dentro del diálogo y se suelta
  // fuera, o desde el fondo y se suelta dentro, el navegador envía el clic al fondo o al diálogo y, sin esta
  // comprobación, se cerraba.
  const pressedOnBackdrop = useRef(false);
  const releasedOnBackdrop = useRef(false);

  useEffect(() => {
    const dialog = dialogRef.current;
    const opener = document.activeElement;
    if (!dialog.open) dialog.showModal();
    // React quita el diálogo de la página antes de ejecutar esta limpieza, así que el navegador ya no devuelve
    // el foco por sí solo: se devuelve aquí al elemento que lo abrió.
    return () => {
      if (dialog.open) dialog.close();
      opener?.focus?.();
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className={`modal-content ${className}`.trim()}
      aria-label={label}
      onCancel={(e) => { e.preventDefault(); onClose(); }}
      onMouseDown={(e) => { pressedOnBackdrop.current = e.target === e.currentTarget; }}
      onMouseUp={(e) => { releasedOnBackdrop.current = e.target === e.currentTarget; }}
      onClick={(e) => {
        if (pressedOnBackdrop.current && releasedOnBackdrop.current && e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-body">
        <button type="button" className="modal-close" aria-label="Cerrar" onClick={onClose}>×</button>
        {children}
      </div>
    </dialog>
  );
}
