"""
src/git_sync.py
===============
Git synchronisation helpers for Google Colab.

Handles cloning the repository at notebook startup and pushing
results back to GitHub when a notebook completes.

Designed to work with a GitHub deploy key added to the repository
(read/write access). The deploy key private key must be stored in
Google Drive and its path passed to configure_ssh().

Typical usage in every notebook
--------------------------------
# ── TOP OF NOTEBOOK (after Drive mount) ──────────────────────────
from src.git_sync import configure_ssh, clone_or_pull
configure_ssh()
clone_or_pull()

# ... all notebook work ...

# ── BOTTOM OF NOTEBOOK ────────────────────────────────────────────
from src.git_sync import push_to_github
push_to_github("NB01: EDA complete — cleaned.pkl saved")
"""

import os
import subprocess
import sys
import textwrap

# ── Project constants — edit these once, inherited by all notebooks ────────────
GITHUB_USERNAME = "<your-username>"           # e.g. "kirunda-jeremy"
REPO_NAME       = "diabetes-hypertension-medication-adherence"
GITHUB_EMAIL    = "<your-email>"              # used for git config
GITHUB_NAME     = "Kirunda Jeremy Menya"      # used for git config

# SSH URL — uses the deploy key for authentication
REPO_SSH_URL    = f"git@github.com:{GITHUB_USERNAME}/{REPO_NAME}.git"

# Where the deploy key private key lives in Google Drive
# Upload your deploy key private key to Drive and set this path.
DEPLOY_KEY_DRIVE_PATH = f"/content/drive/MyDrive/{REPO_NAME}/deploy_key"

# Where the key will be installed inside the Colab VM
DEPLOY_KEY_LOCAL_PATH = "/root/.ssh/deploy_key"

# Local clone path inside the Colab VM
COLAB_CLONE_PATH = f"/content/{REPO_NAME}"


# ── SSH configuration ─────────────────────────────────────────────────────────
def configure_ssh(
    deploy_key_drive_path: str = DEPLOY_KEY_DRIVE_PATH,
    deploy_key_local_path: str = DEPLOY_KEY_LOCAL_PATH,
) -> None:
    """
    Install the deploy key and configure SSH so that git commands
    authenticate to GitHub without a password prompt.

    Must be called once per Colab session, before clone_or_pull()
    or push_to_github().

    Steps performed:
        1. Copy the deploy key from Drive to /root/.ssh/
        2. Set correct permissions (chmod 600) — SSH refuses keys
           that are world-readable
        3. Write ~/.ssh/config to route github.com through the key
        4. Add github.com to known_hosts to suppress the host
           verification prompt that would block non-interactive runs

    Parameters
    ----------
    deploy_key_drive_path : str
        Path to the private key file in Google Drive.
    deploy_key_local_path : str
        Where to install the key inside the Colab VM.
    """
    if not _in_colab():
        print("[git_sync] Not running in Colab — SSH configuration skipped.")
        return

    # 1. Copy key from Drive
    if not os.path.exists(deploy_key_drive_path):
        raise FileNotFoundError(
            f"\n[git_sync] Deploy key not found at:\n"
            f"  {deploy_key_drive_path}\n\n"
            f"To fix this:\n"
            f"  1. Export your deploy key private key from GitHub:\n"
            f"     GitHub → repo → Settings → Deploy keys\n"
            f"  2. Upload the private key file to Google Drive at:\n"
            f"     {deploy_key_drive_path}\n"
            f"  3. Re-run this cell."
        )

    os.makedirs("/root/.ssh", exist_ok=True)
    _run(f"cp '{deploy_key_drive_path}' '{deploy_key_local_path}'")

    # 2. Correct permissions — SSH will refuse to use a key that
    #    is group- or world-readable (security enforcement).
    _run(f"chmod 600 '{deploy_key_local_path}'")

    # 3. SSH config: use this key for all github.com connections
    ssh_config = textwrap.dedent(f"""\
        Host github.com
            HostName github.com
            User git
            IdentityFile {deploy_key_local_path}
            StrictHostKeyChecking no
    """)
    config_path = "/root/.ssh/config"
    with open(config_path, "w") as f:
        f.write(ssh_config)
    _run(f"chmod 600 '{config_path}'")

    # 4. Add github.com to known_hosts to prevent interactive prompt
    _run("ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts 2>/dev/null")

    # 5. Set git identity (required for commits)
    _run(f'git config --global user.email "{GITHUB_EMAIL}"')
    _run(f'git config --global user.name "{GITHUB_NAME}"')

    print("[git_sync] SSH configured successfully.")
    print(f"           Key : {deploy_key_local_path}")
    print(f"           User: {GITHUB_NAME} <{GITHUB_EMAIL}>")


# ── Clone or pull ─────────────────────────────────────────────────────────────
def clone_or_pull(
    clone_path: str = COLAB_CLONE_PATH,
    ssh_url: str    = REPO_SSH_URL,
) -> None:
    """
    Clone the repository if it does not exist in the Colab VM,
    or pull the latest changes if it already does.

    After cloning/pulling, changes the working directory to the
    repository root so all subsequent relative paths (data/raw/,
    figures/, src/) resolve correctly.

    Must be called after configure_ssh().

    Parameters
    ----------
    clone_path : str
        Local path to clone the repo into.
    ssh_url : str
        SSH URL of the GitHub repository.
    """
    if not _in_colab():
        print("[git_sync] Not running in Colab — clone/pull skipped.")
        print(f"           Expected working directory: project root")
        return

    if os.path.exists(os.path.join(clone_path, ".git")):
        # Repository already cloned — pull latest changes
        print(f"[git_sync] Repository found at {clone_path}")
        print("[git_sync] Pulling latest changes from origin/main ...")
        result = _run(f"git -C '{clone_path}' pull origin main", capture=True)
        print(f"           {result.strip()}")
    else:
        # Fresh clone
        print(f"[git_sync] Cloning {ssh_url} → {clone_path} ...")
        _run(f"git clone '{ssh_url}' '{clone_path}'")
        print("[git_sync] Clone complete.")

    # Change working directory to the repo root
    os.chdir(clone_path)
    if clone_path not in sys.path:
        sys.path.insert(0, clone_path)

    print(f"[git_sync] Working directory set to: {os.getcwd()}")
    _verify_structure()


# ── Push to GitHub ────────────────────────────────────────────────────────────
def push_to_github(
    commit_message: str,
    branch: str      = "main",
    clone_path: str  = COLAB_CLONE_PATH,
) -> None:
    """
    Stage all changes, commit with the provided message, and push
    to the remote repository.

    Call this as the final cell of every notebook after all outputs
    (figures, pickled dataframes, model files) have been saved.

    What gets committed:
        - Everything not excluded by .gitignore
        - This means: notebooks, src/ files, figures/, reports/
        - data/raw/, data/processed/, and models/ are gitignored
          and will NOT be committed (patient data stays local)

    Parameters
    ----------
    commit_message : str
        The commit message. Be specific — include the notebook
        number and what was produced.
        Example: "NB01: EDA complete — cleaned.pkl saved, 8 figures"

    branch : str
        Branch to push to. Default 'main'.

    clone_path : str
        Path to the local git repository.

    Example
    -------
    push_to_github("NB03: Baseline models — results table and ROC curves")
    """
    if not _in_colab():
        print("[git_sync] Not running in Colab — push skipped.")
        print(f"           Commit message would have been: '{commit_message}'")
        return

    print(f"[git_sync] Staging all changes ...")
    _run(f"git -C '{clone_path}' add -A")

    # Check if there is anything to commit
    status = _run(f"git -C '{clone_path}' status --porcelain", capture=True)
    if not status.strip():
        print("[git_sync] Nothing to commit — working tree is clean.")
        return

    print(f"[git_sync] Committing: '{commit_message}'")
    _run(f"git -C '{clone_path}' commit -m \"{commit_message}\"")

    print(f"[git_sync] Pushing to origin/{branch} ...")
    _run(f"git -C '{clone_path}' push origin {branch}")

    # Confirm with the latest commit hash
    commit_hash = _run(
        f"git -C '{clone_path}' rev-parse --short HEAD", capture=True
    ).strip()
    print(f"[git_sync] Push successful. Commit: {commit_hash}")
    print(f"           Message: {commit_message}")


# ── Status helper ─────────────────────────────────────────────────────────────
def git_status(clone_path: str = COLAB_CLONE_PATH) -> None:
    """
    Print the current git status. Useful for checking what has changed
    before calling push_to_github().

    Parameters
    ----------
    clone_path : str
        Path to the local git repository.
    """
    print("[git_sync] Current git status:")
    result = _run(f"git -C '{clone_path}' status", capture=True)
    print(result)

    print("[git_sync] Recent commits:")
    log = _run(
        f"git -C '{clone_path}' log --oneline -5", capture=True
    )
    print(log)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _run(cmd: str, capture: bool = False) -> str:
    """
    Run a shell command. Raises RuntimeError on non-zero exit code.
    If capture=True, returns stdout as a string instead of printing.
    """
    result = subprocess.run(
        cmd, shell=True, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"\n[git_sync] Command failed (exit {result.returncode}):\n"
            f"  Command : {cmd}\n"
            f"  Error   : {error_msg}"
        )
    return result.stdout if capture else ""


def _in_colab() -> bool:
    """Return True if running inside Google Colab."""
    return "google.colab" in sys.modules


def _verify_structure() -> None:
    """
    Warn if expected directories are missing after clone.
    This catches the case where the repo was cloned but the
    directory structure has not been committed yet.
    """
    expected = ["src", "notebooks", "data", "figures", "models", "reports"]
    missing  = [d for d in expected if not os.path.exists(d)]
    if missing:
        print(f"[git_sync] WARNING: Expected directories not found: {missing}")
        print("           Create them and push before running notebooks.")
    else:
        print("[git_sync] Repository structure verified.")
