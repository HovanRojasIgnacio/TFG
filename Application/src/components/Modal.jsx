import { useRef } from 'react';
import { useModalFocus } from '../hooks/useModalFocus';

/** Diálogo modal accesible: foco atrapado, Escape y clic fuera lo cierran. */
export default function Modal({ label, onClose, className = '', children }) {
  const modalRef = useModalFocus(onClose);
  // Solo cierra un clic que empieza y termina en el fondo. Si se arrastra para seleccionar texto dentro del
  // diálogo y se suelta fuera, el navegador envía el clic al fondo y, sin esta comprobación, se cerraba.
  const pressedOnOverlay = useRef(false);

  return (
    <div
      className="modal-overlay"
      onMouseDown={(e) => { pressedOnOverlay.current = e.target === e.currentTarget; }}
      onClick={(e) => {
        if (pressedOnOverlay.current && e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={`modal-content ${className}`.trim()}
        ref={modalRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={label}
      >
        <button type="button" className="modal-close" aria-label="Cerrar" onClick={onClose}>×</button>
        {children}
      </div>
    </div>
  );
}
