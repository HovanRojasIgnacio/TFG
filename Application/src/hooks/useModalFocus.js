import { useEffect, useRef } from 'react';

/**
 * Gestión de foco para un diálogo modal montado solo mientras está abierto:
 * el foco entra en el diálogo, Tab queda atrapado dentro, Escape lo cierra
 * y al desmontarse el foco vuelve al elemento que lo abrió.
 * Devuelve la ref que debe ponerse en el contenedor del diálogo.
 */
export function useModalFocus(onClose) {
  const modalRef = useRef(null);

  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  });

  useEffect(() => {
    const opener = document.activeElement;
    const modal = modalRef.current;
    modal?.focus();

    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        onCloseRef.current();
        return;
      }
      if (e.key !== 'Tab' || !modal) return;

      const focusable = modal.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]'
      );
      if (focusable.length === 0) {
        e.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && (document.activeElement === first || document.activeElement === modal)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      opener?.focus?.();
    };
  }, []);

  return modalRef;
}
