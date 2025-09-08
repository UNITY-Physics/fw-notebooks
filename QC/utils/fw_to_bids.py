import os
import re
import shutil
from pathlib import Path

# Map keywords in folder names to BIDS suffixes
MODALITY_MAP = {
    "T1": "T1w",
    "T2": "T2w",
    "FLAIR": "FLAIR",
    "dwi": "dwi",
    "bold": "bold"
}

ACQ_MAP = {
    "AXI": "axi",
    "SAG": "sag",
    "COR": "cor"
}

def sanitize_session(session_name):
    # Case 1: session is a date or date+time
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})(?:[ T_-](\d{2})[:_]?(\d{2})[:_]?(\d{2}))?", session_name)
    if match:
        y, m, d, H, M, S = match.groups()
        if H and M and S:
            return f"ses-{y}{m}{d}T{H}{M}{S}"
        else:
            return f"ses-{y}{m}{d}"
    
    # Case 2: general label (e.g. "Type-visit-14")
    clean = re.sub(r"[^a-zA-Z0-9]", "", session_name)
    return f"ses-{clean}"


def convert_to_bids(root_dir, bids_root, copy=True):
    """
    Convert custom folder structure to BIDS.
    
    Parameters
    ----------
    root_dir : str
        Path to the root directory containing project folders.
    bids_root : str
        Destination BIDS root directory.
    copy : bool
        If True, copy files; if False, symlink.
    """
    root_dir = Path(root_dir)
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    for project in root_dir.iterdir():
        if not project.is_dir():
            continue

        for subject in project.iterdir():
            if not subject.is_dir():
                continue
            
            subj_id = subject.name.replace("_", "").replace("-", "")
            subj_label = f"sub-{subj_id}"

            for session in subject.iterdir():
                if not session.is_dir():
                    continue

                ses_label = sanitize_session(session.name)
                anat_dir = bids_root / project.name / subj_label / ses_label / "anat"
                anat_dir.mkdir(parents=True, exist_ok=True)

                for scan in session.iterdir():
                    if not scan.is_dir():
                        continue
                    
                    scan_name = scan.name
                    modality = None
                    acq_label = None
                    
                    # Find modality
                    for key, suffix in MODALITY_MAP.items():
                        if key.lower() in scan_name.lower():
                            modality = suffix
                            break

                    # Find acquisition plane
                    for key, label in ACQ_MAP.items():
                        if key.lower() in scan_name.lower():
                            acq_label = f"acq-{label}"
                            break

                    if modality is None:
                        print(f"⚠️ Skipping (unknown modality): {scan_name}")
                        continue

                    for f in scan.glob("*"):
                        if f.suffix.lower() in [".nii", ".nii.gz"]:
                            parts = [subj_label, ses_label]
                            if acq_label:
                                parts.append(acq_label)
                            parts.append(modality)

                            bids_name = "_".join(parts) + f.suffix
                            dest = anat_dir / bids_name
                            if copy:
                                shutil.copy2(f, dest)
                            else:
                                if dest.exists():
                                    dest.unlink()
                                os.symlink(f.resolve(), dest)
                            print(f"✅ {f} -> {dest}")

if __name__ == "__main__":
    # Example usage
    convert_to_bids(
        root_dir="/home/k2480041/QC/Images/Train",
        bids_root="/home/k2480041/QC/BIDS_out",
        copy=True
    )
