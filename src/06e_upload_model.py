"""Upload the shipped detector and its model card to Hugging Face.

Ship candidate: A_f100 seed 2 (best locked-test mAP@0.5). Uploads best.pt,
best.onnx (from 06f) and MODEL_CARD.md as the Hub README.md. Run from the
repository root with ``gprdet`` active and a write token configured.
"""

from huggingface_hub import HfApi

REPO_ID = "hectorvh/gpradar-hyperbola-yolo"
WEIGHTS_DIR = "runs/detect/A_f100_seed2/weights"
FILES = {
    f"{WEIGHTS_DIR}/best.pt": "best.pt",
    f"{WEIGHTS_DIR}/best.onnx": "best.onnx",
    "MODEL_CARD.md": "README.md",
}

api = HfApi()
api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
for local, remote in FILES.items():
    info = api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO_ID)
    print(f"{local} -> {remote}: commit {info.oid}")
