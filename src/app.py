'''
functional
will need to be extended for docx, pdf, etc.
'''

import os
import pandas as pd
import re
import tkinter as tk
import torch
from tkinter import filedialog, messagebox, ttk
from transformers import MarianMTModel, MarianTokenizer

# ---------- Translator Code ---------- #

# Constants for model selection based on language
LANGUAGE_OPTIONS = ['en', 'de', 'es', 'fr', 'it', 'pt', 'ru']
MODEL_MAP = {
    'en-de': "Helsinki-NLP/opus-mt-en-de",
    'de-en': "Helsinki-NLP/opus-mt-de-en",
    'en-es': "Helsinki-NLP/opus-mt-en-es",
    'es-en': "Helsinki-NLP/opus-mt-es-en",
    'en-fr': "Helsinki-NLP/opus-mt-en-fr",
    'fr-en': "Helsinki-NLP/opus-mt-fr-en",
    'en-it': "Helsinki-NLP/opus-mt-en-it",
    'it-en': "Helsinki-NLP/opus-mt-it-en",
    'en-pt': "Helsinki-NLP/opus-mt-en-pt",
    'pt-en': "Helsinki-NLP/opus-mt-pt-en",
    'en-ru': "Helsinki-NLP/opus-mt-en-ru",
    'ru-en': "Helsinki-NLP/opus-mt-ru-en",
    # add other language pairs as needed
}

def load_model(source_lang, target_lang):
    model_name = MODEL_MAP.get(f"{source_lang}-{target_lang}")
    if not model_name:
        messagebox.showerror("Error", "No model available for selected language pair.")
        return None, None, None
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name, model_max_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def split_into_batches(text, tokenizer, max_tokens=400):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return [text]
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def custom_translator(text, model, tokenizer, device):
    batches = split_into_batches(text, tokenizer)
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

def translate_dataframe(app, df, model, tokenizer, device, progress_bar):
    total_size = df.size
    progress_bar["maximum"] = total_size
    translated_df = df.copy()
    progress = 0
    for col in translated_df.columns:
        translated_texts = []
        for text in translated_df[col]:
            if pd.isna(text):
                translated_texts.append(text)
            else:
                translated_texts.append(custom_translator(text, model, tokenizer, device))
            progress += 1
            progress_bar["value"] = progress
            app.update()  # This now works because `app` is passed to the function
        translated_df[col] = translated_texts
    return translated_df

def translate_file(app, filepath, source_lang, target_lang):
    model, tokenizer, device = load_model(source_lang, target_lang)
    file_extension = os.path.splitext(filepath)[1]
    
    if file_extension in ['.csv', '.xlsx', '.txt']:
        if file_extension == '.csv':
            df = pd.read_csv(filepath)
        elif file_extension == '.xlsx':
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            with open(filepath, 'r') as f:
                data = f.readlines()
            df = pd.DataFrame(data, columns=['Text'])

        progress_bar = ttk.Progressbar(app, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(padx=10, pady=20)
        translated_df = translate_dataframe(app, df, model, tokenizer, device, progress_bar)  # Pass `app` here
        
        base_name = os.path.splitext(filepath)[0]
        translated_file_path = f"{base_name}[{source_lang}_{target_lang}]{file_extension}"
        if file_extension == '.csv':
            translated_df.to_csv(translated_file_path, index=False)
        elif file_extension == '.xlsx':
            translated_df.to_excel(translated_file_path, index=False, engine='openpyxl')
        else:
            with open(translated_file_path, 'w') as f:
                for row in translated_df.itertuples(index=False):
                    f.write(f"{row[0]}\n")
        
        messagebox.showinfo("Success", f"File translated successfully and saved as {translated_file_path}")
        app.quit()

# ---------- GUI Code ---------- #

def run_app():
    app = tk.Tk()
    app.title("Language Translator")

    # Variables to store the selected languages
    source_vars = [tk.StringVar() for _ in LANGUAGE_OPTIONS]
    target_vars = [tk.StringVar() for _ in LANGUAGE_OPTIONS]

    # Create source language selection checkboxes
    tk.Label(app, text="Select Source Language:").pack()
    for lang, var in zip(LANGUAGE_OPTIONS, source_vars):
        chk = ttk.Checkbutton(app, text=lang, variable=var, onvalue=lang, offvalue='', command=lambda: update_button_state())
        chk.pack(anchor=tk.W)

    # Create target language selection checkboxes
    tk.Label(app, text="Select Target Language:").pack()
    for lang, var in zip(LANGUAGE_OPTIONS, target_vars):
        chk = ttk.Checkbutton(app, text=lang, variable=var, onvalue=lang, offvalue='', command=lambda: update_button_state())
        chk.pack(anchor=tk.W)

    # Button to proceed with file selection
    select_button = tk.Button(app, text="Select File to Translate", state='disabled')
    select_button.pack(padx=10, pady=10)

    # Function to update the state of the select button
    def update_button_state():
        selected_source = [var.get() for var in source_vars if var.get()]
        selected_target = [var.get() for var in target_vars if var.get()]
        if selected_source and selected_target and selected_source != selected_target:
            select_button.config(state='normal', command=select_file)
        else:
            select_button.config(state='disabled')

    # Function to handle file selection and translation
    def select_file():
        filepath = filedialog.askopenfilename()
        if filepath:
            selected_source_lang = next((var.get() for var in source_vars if var.get()), None)
            selected_target_lang = next((var.get() for var in target_vars if var.get()), None)
            
            if selected_source_lang and selected_target_lang and selected_source_lang != selected_target_lang:
                # Load model and start translation
                model, tokenizer, device = load_model(selected_source_lang, selected_target_lang)
                if model and tokenizer and device:
                    translate_file(app, filepath, selected_source_lang, selected_target_lang)
                else:
                    messagebox.showerror("Error", "Failed to load the translation model.")
            else:
                messagebox.showerror("Error", "Please ensure distinct source and target languages are selected.")

    app.mainloop()

if __name__ == "__main__":
    run_app()