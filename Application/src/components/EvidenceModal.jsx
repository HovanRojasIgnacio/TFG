import Modal from './Modal';

export default function EvidenceModal({ imageUrl, onClose }) {
  return (
    <Modal label="Evidencia de infracción" onClose={onClose}>
      <img src={imageUrl} alt="Imagen de la infracción, con los recuadros de la detección" />
      <div className="modal-meta">Imagen guardada por el sistema al detectar la infracción.</div>
    </Modal>
  );
}
