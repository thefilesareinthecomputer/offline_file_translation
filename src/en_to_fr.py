import os
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm
import re

FILE_NAME = "article_summary.txt"  # File to translate
MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"  

model = MarianMTModel.from_pretrained(MODEL_NAME)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def split_into_batches(text, max_tokens=400):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return [text]
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def custom_translator(text):
    batches = split_into_batches(text)
    translated_text_batches = []
    for batch in batches:
        if batch.strip() == '':
            continue

        tokenized_text = tokenizer([batch], return_tensors="pt", padding=True, truncation=False).input_ids.to(device)
        translated = model.generate(tokenized_text, max_length=512)
        translated_ids = translated[0].tolist()
        tgt_text = tokenizer.decode(translated_ids, skip_special_tokens=True)
        translated_text_batches.append(tgt_text)

    return ' '.join(translated_text_batches)

def translate_dataframe(df):
    translated_df = df.copy()
    translated_df = translated_df.map(lambda x: x.lower() if isinstance(x, str) else x)
    with tqdm(total=translated_df.size, desc="Translating Cells") as pbar:
        for col in translated_df.columns:
            translated_texts = []
            for text in translated_df[col]:
                try:
                    if pd.isna(text):
                        translated_texts.append(text)
                    else:
                        translated_texts.append(custom_translator(str(text)))
                    pbar.update(1)
                except Exception as e:
                    print(f"An error occurred during translation: {e}")
                    pbar.update(1)
            translated_df[col] = translated_texts
    return translated_df

def main():
    downloads_folder_file_name = os.path.join(os.path.expanduser('~'), 'Downloads', FILE_NAME)
    file_extension = os.path.splitext(downloads_folder_file_name)[1]
    
    if file_extension == '.csv':
        df = pd.read_csv(downloads_folder_file_name)
    elif file_extension == '.xlsx':
        df = pd.read_excel(downloads_folder_file_name, engine='openpyxl')
    elif file_extension == '.txt':
        with open(downloads_folder_file_name, 'r') as f:
            data = f.readlines()
        processed_data = [line.strip() for line in data]
        df = pd.DataFrame(processed_data, columns=['Text'])
    else:
        raise ValueError("Unsupported file type")

    translated_df = translate_dataframe(df)
    base_name = os.path.splitext(downloads_folder_file_name)[0]
    translated_file_path = f"{base_name} [French Translation]{file_extension}"
    
    if file_extension == '.csv':
        translated_df.to_csv(translated_file_path, index=False)
    elif file_extension == '.xlsx':
        translated_df.to_excel(translated_file_path, index=False, engine='openpyxl')
    elif file_extension == '.txt':
        with open(translated_file_path, 'w') as f:
            for row in translated_df.itertuples(index=False):
                f.write(f"{row[0]}\n")

if __name__ == "__main__":
    main()

'''
Key differences between OPUS-MT and Google Translate:

Feature	                        Helsinki-NLP/opus-mt-en-es	         Google Translate
Requires internet connection	No	                                 Yes
Machine translation type    	Statistical	                         Neural
Accuracy	                    Higher	                             Lower
Speed	                        Slower	                             Faster
Best use cases	                High-accuracy tasks	                 Speed or simplicity tasks
License                         Free / Open source	                 Proprietary
Processing                      On local machine	                 On Google servers
'''