"""Relanza el script con el Python de Inference/.venv si existe y no se está usando ya.

El Python global suele tener PyTorch solo-CPU; el entorno virtual lleva la versión con CUDA.
Así `python execute.py` funciona con cualquier Python. Se desactiva con GUARDWATCH_NO_VENV=1.
"""
import os
import subprocess
import sys


def ensure_venv() -> None:
    if os.environ.get("GUARDWATCH_NO_VENV") == "1":
        return
    here = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(here, ".venv")
    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
    if not os.path.exists(venv_python):
        return
    if os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(venv_dir):
        return  # ya estamos dentro del entorno virtual

    env = dict(os.environ, GUARDWATCH_NO_VENV="1")
    # Un entorno copiado de otro equipo apunta a un Python que aquí no existe y no llega a arrancar
    if subprocess.call([venv_python, "-c", ""], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise SystemExit(
            f"[GuardWatch] El entorno virtual {venv_dir} no funciona en este equipo (¿se creó en otro o se "
            "desinstaló el Python con el que se hizo?). Bórralo y créalo de nuevo como indica el README, o usa el "
            "Python actual con GUARDWATCH_NO_VENV=1.")
    print(f"[GuardWatch] Usando el entorno virtual: {venv_python}", flush=True)
    raise SystemExit(subprocess.call([venv_python] + sys.argv, env=env))



def setup(is_main: bool) -> None:
    """Prepara el proceso antes de importar torch, numpy, OpenCV o ultralytics; is_main: se lanzó como script."""
    # Evita el error de Windows "OpenMP runtime already initialized" cuando torch, numpy y OpenCV cargan cada uno
    # su propia copia de libiomp (workaround habitual). Debe fijarse antes de importar cualquiera de ellos.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if is_main:
        ensure_venv()  # Usa Inference/.venv (PyTorch con CUDA) aunque se lance con otro Python
    # Ultralytics no debe instalar paquetes por su cuenta con pip (puede modificar el Python global)
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
