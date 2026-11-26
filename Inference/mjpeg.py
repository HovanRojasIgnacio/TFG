"""Formato del vídeo de /api/stream (MJPEG, multipart/x-mixed-replace); lo usa también el backend simulado.

Cada frame se envía seguido del separador y las cabeceras de la parte siguiente: Chromium solo muestra una parte
cuando los recibe, así que con ellos delante cada frame se veía con un frame de retraso y el último antes de una
pausa (por ejemplo, «SIN IMAGEN» al perder la cámara) no se veía nunca.
"""
BOUNDARY = "frame"
MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
PART_HEADER = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n\r\n".encode()  # Lo primero que se envía


def part(jpeg: bytes) -> bytes:
    """Un frame seguido del separador y las cabeceras de la parte siguiente."""
    return jpeg + b"\r\n" + PART_HEADER
