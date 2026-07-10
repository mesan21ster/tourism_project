from huggingface_hub import HfApi
import os

token=os.getenv("HF_MLOps")   # please use your token

api = HfApi(token=os.getenv("HF_MLOps"))
api.upload_folder(
    folder_path="TourismPackagePrediction/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id="mesan21ster/TourismPackagePrediction",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
