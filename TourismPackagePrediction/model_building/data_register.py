from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

token = os.getenv("HF_MLOps")

api = HfApi(token=token)

repo_id = "mesan21ster/TourismPackagePrediction"
repo_type = "dataset"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"{repo_id} already exists")
except RepositoryNotFoundError:
    print(f"{repo_id} not found. Creating...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        private=False,
        exist_ok=True
    )

api.upload_folder(
    folder_path="TourismPackagePrediction/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
