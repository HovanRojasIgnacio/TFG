import Modal from './Modal';

export default function EvidenceModal({ imageUrl, onClose }) {
  return (
    <Modal label="Evidencia de infracción" onClose={onClose}>
      <img src={imageUrl} alt="Evidencia Detallada" />
      <div className="modal-meta">Evidencia Forense — Captura de fotograma procesada por el motor de inferencia</div>
    </Modal>
  );
}
