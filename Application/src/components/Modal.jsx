import { useModalFocus } from '../hooks/useModalFocus';

/** Diálogo modal accesible: foco atrapado, Escape y clic fuera lo cierran. */
export default function Modal({ label, onClose, className = '', children }) {
  const modalRef = useModalFocus(onClose);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className={`modal-content ${className}`.trim()}
        ref={modalRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        onClick={(e) => e.stopPropagation()}
      >
        <button type="button" className="modal-close" aria-label="Cerrar" onClick={onClose}>×</button>
        {children}
      </div>
    </div>
  );
}
