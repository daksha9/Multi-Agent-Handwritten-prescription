"""Initialize git repo, stage files (respecting .gitignore), and commit using dulwich."""
import os
import stat
import time
from pathlib import Path

import pathspec
from dulwich import porcelain
from dulwich.repo import Repo
from dulwich.objects import Blob, Tree, Commit, parse_timezone
from dulwich.index import build_index_from_tree

REPO_ROOT = Path(r"c:\Users\HP\OneDrive\Desktop\Handwritten\med_agent")
GIT_DIR = REPO_ROOT / ".git"

AUTHOR_NAME = b"Med Agent Developer"
AUTHOR_EMAIL = b"developer@medagent.local"
COMMIT_MSG = b"feat: Initial commit - Multi-Intelligence Agent for Handwritten Prescription and Predictive Health Analytics\n\n- Streamlit UI with 3 tabs (Prescription Extraction, Reports Comparison, Report Analysis)\n- Gemini Vision AI for handwritten prescription OCR\n- Longitudinal lab report comparison with medication impact analysis\n- Power BI-ready relational analytics export\n- Blank patient profile default sidebar\n- Composite (date, medication) report grouping to prevent report merging\n- Data isolation: prescription vs report analysis"

# ── 1. Initialize repo ──
if not GIT_DIR.exists():
    repo = porcelain.init(str(REPO_ROOT))
    print(f"[INIT] Initialized git repo at {REPO_ROOT}")
else:
    repo = Repo(str(REPO_ROOT))
    print(f"[INIT] Repo already exists at {REPO_ROOT}")

# ── 2. Load .gitignore ──
gitignore_path = REPO_ROOT / ".gitignore"
gitignore_lines = []
if gitignore_path.exists():
    gitignore_lines = gitignore_path.read_text(encoding="utf-8").splitlines()
spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_lines)

# ── 3. Walk files (skip ignored dirs early) and stage respecting .gitignore ──
staged_files = []
skipped_files = []

SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".pyright", "node_modules", ".idea"}

for dirpath, dirnames, filenames in os.walk(str(REPO_ROOT)):
    # Prune directories we never want to descend into
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

    for fname in filenames:
        full = Path(dirpath) / fname
        rel_path = full.relative_to(REPO_ROOT)
        rel_str = str(rel_path).replace("\\", "/")

        # Check .gitignore
        if spec.match_file(rel_str):
            skipped_files.append(rel_str)
            continue

        staged_files.append(rel_str)

print(f"[STAGE] Will stage {len(staged_files)} files (skipped {len(skipped_files)} via .gitignore)")

# Show first 10 skipped for sanity
if skipped_files:
    print(f"[SKIP] Sample ignored files: {skipped_files[:10]}")

# ── 4. Stage using porcelain.add ──
# porcelain.add respects .gitignore? No — it stages whatever paths you give it.
# But we've already filtered, so pass the filtered list.
porcelain.add(repo, paths=staged_files)

# ── 5. Create commit ──
commit_id = porcelain.commit(
    repo,
    message=COMMIT_MSG,
    author=f"{AUTHOR_NAME.decode()} <{AUTHOR_EMAIL.decode()}>".encode(),
    committer=f"{AUTHOR_NAME.decode()} <{AUTHOR_EMAIL.decode()}>".encode(),
)
print(f"[COMMIT] Created commit {commit_id.decode()[:12]}")
print(f"[COMMIT] Staged {len(staged_files)} files total.")
