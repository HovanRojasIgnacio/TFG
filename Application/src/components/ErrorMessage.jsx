// Mensaje de error con el icono decorativo; `role` es "alert" (urgente) o "status" (informativo)
export default function ErrorMessage({ children, role = "alert", className = "camera-error" }) {
  return (
    <div className={className} role={role}>
      <span aria-hidden="true">⚠ </span>{children}
    </div>
  );
}
