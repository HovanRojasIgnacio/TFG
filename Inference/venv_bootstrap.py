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

    print(f"[GuardWatch] Usando el entorno virtual: {venv_python}", flush=True)
    env = dict(os.environ, GUARDWATCH_NO_VENV="1")
    raise SystemExit(subprocess.call([venv_python] + sys.argv, env=env))
