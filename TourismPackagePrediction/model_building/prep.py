
import pandas as pd
import sklearn
import os
from huggingface_hub import hf_hub_download
from google.colab import userdata
from huggingface_hub import HfApi



# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths


token = os.getenv("HF_MLOps")
api = HfApi(token=os.getenv("HF_MLOps"))

# please create your dataset as you create your space
DATASET_PATH = hf_hub_download(
    repo_id="mesan21ster/TourismPackagePrediction",
    filename="tourism.csv",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
)
#"hf://datasets/mesan21ster/TourismPackagePrediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=["Unnamed: 0","CustomerID"], inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())


# Remove leading/trailing spaces from all string columns
object_columns = df.select_dtypes(include="object").columns

for col in object_columns:
    df[col] = df[col].str.strip()

# Fix inconsistent values

# Marital Status
df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})

# Gender
df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female"
})

# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()

categorical_columns = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="mesan21ster/TourismPackagePrediction",
        repo_type="dataset",
    )
