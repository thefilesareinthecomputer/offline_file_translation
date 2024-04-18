## Text Translation Script

# Capabilities
Translates text between languages using Helsinki-NLP's OPUS-MT models from Hugging Face.
Supports multiple file formats: .txt, .csv, .xlsx.
Batch processing for handling large volumes of text.

# Benefits
Flexibility: Works with various languages by changing the OPUS-MT model.
Privacy: Performs translations locally, ensuring data privacy.
Convenience: Easy to set up and run, making translations hassle-free.

# Specifications
Language Support: Dependent on available Helsinki-NLP/OPUS-MT models.
File Format Support: Text (.txt), Comma-Separated Values (.csv), and Excel (.xlsx).
Dependencies: Python, pandas, torch, transformers, tqdm, openpyxl, pyinstaller.

# Installation
Ensure Python is installed, then run:
pip install pandas torch transformers tqdm openpyxl sentencepiece sacremoses

# Usage
Place your file in an accessible location.
Set MODEL_NAME in the script to your desired OPUS-MT model (e.g., "Helsinki-NLP/opus-mt-en-es").
Update FILE_NAME in the script to point to your file.
Execute the script.
The translated file will be saved alongside the original, marked with the target language.

# Language Customization
Change MODEL_NAME to the desired Helsinki-NLP/OPUS-MT model for your specific language pair. Visit Hugging Face for a list of available models.

# Comparison with Google Translate
Offline Capability: Unlike Google Translate, this script does not require an internet connection.
Data Privacy: Translations are done locally, offering better privacy.
Customizability: Easily switch between languages by changing models.
License
This script is open-source. Feel free to modify and distribute as needed.