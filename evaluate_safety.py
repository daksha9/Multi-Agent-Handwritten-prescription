# Forwarding runner to data/prescriptions/prescriptions_step1/evaluate_safety.py
import sys
from pathlib import Path

target_script = Path(__file__).resolve().parent / "data" / "prescriptions" / "prescriptions_step1" / "evaluate_safety.py"
with open(target_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(target_script), "exec")
exec(code, globals())

