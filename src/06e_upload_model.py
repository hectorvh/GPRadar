"""Upload the shipped YOLO weights to Hugging Face.

Ship candidate: A_f100 seed 2 (best locked-test mAP@0.5). Run from the
repository root with ``gprdet`` active and a write token configured.
"""

from huggingface_hub import HfApi

WEIGHTS = "runs/detect/A_f100_seed2/weights/best.pt"

api = HfApi()
api.create_repo("hectorvh/gpradar-hyperbola-yolo", repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj=WEIGHTS,
    path_in_repo="best.pt",
    repo_id="hectorvh/gpradar-hyperbola-yolo",
)
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id="hectorvh/gpradar-hyperbola-yolo",
)
