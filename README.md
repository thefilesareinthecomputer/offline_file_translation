# Text File Translation App

## Capabilities
- Translates text files from one language to another using Helsinki-NLP's OPUS-MT models from Hugging Face.
- Supports multiple file formats: .txt, .csv, .xlsx.
- Batch processing for handling large volumes of text.
- Token chunking for efficient translation of lengthy text while avoiding memory issues.

## Usage
- Place your text file in an accessible location.
- Select the desired source and target languages from the lists in the UI.
- Select the file you want to translate in the UI file explorer window.
- The translated file will be saved alongside the original, with the file name appended with the target language.

## Benefits
- Flexibility: Works with multiple languages.
- Privacy: Performs translations locally and offline, ensuring data privacy.
- Convenience: Easy to set up and run, making translations hassle-free.

## Specifications
- Language Support: Dependent on available Helsinki-NLP/OPUS-MT models.
- File Format Support: Text (.txt), Comma-Separated Values (.csv), and Excel (.xlsx).
- Dependencies: Python, pandas, torch, transformers, tqdm, openpyxl, pyinstaller.

## Installation & Setup
- Ensure Python 3.11+ is installed, then:
- Clone the github repo from: https://github.com/thefilesareinthecomputer/offline_file_translation
- Set up a virtual environment and install the required dependencies:
```bash
git clone {REPO_URL} {REPO_FOLDER}

cd {REPO_FOLDER}

python3.11 -m venv {VENV_NAME}

source {VENV_NAME}/bin/activate
 
pip install --upgrade pip pip-check-reqs wheel python-dotenv

pip install -r requirements.txt

pip install {ADDITIONAL_PACKAGES}

pip freeze > requirements.txt

echo "{VENV_NAME}/
_archive/
_notes/
_notes.txt
generated_data/
venv/
__pycache__/
*.pyc
*/migrations/*
db.sqlite3
.env
staticfiles/" > .gitignore

cat .gitignore

git init

git add .

git commit -m "Initial commit"

optionally, set a remote repository and push the new code to it
```
