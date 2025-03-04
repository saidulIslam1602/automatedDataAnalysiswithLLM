# PDF Document Analysis with LLM

This project provides an automated pipeline for extracting, processing, and analyzing PDF documents using Large Language Models (LLMs). It includes features for text extraction, preprocessing, semantic search, and fine-tuning capabilities.

## Features

- PDF text extraction with PyPDF2
- Text preprocessing with NLTK (tokenization, lemmatization, stopword removal)
- Document embedding generation using Sentence Transformers
- Semantic search across documents
- LLM fine-tuning capabilities using Hugging Face Transformers
- Automated data processing pipeline

## Project Structure

```
.
├── src/
│   ├── data_processing/
│   │   ├── pdf_extractor.py
│   │   └── text_preprocessor.py
│   ├── models/
│   │   └── llm_handler.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── tests/
├── userManual/
│   └── [PDF files]
└── requirements.txt
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your PDF documents in the `userManual` directory.

2. Run the analysis pipeline:
```bash
python src/main.py
```

This will:
- Extract text from all PDFs
- Preprocess the extracted text
- Generate document embeddings
- Enable semantic search capabilities

3. The processed data will be saved in the `data/processed` directory:
- `*_processed.json`: Contains original and processed text
- `*_embeddings.npy`: Contains document embeddings

## Customization

### Modifying Text Preprocessing
Edit `src/data_processing/text_preprocessor.py` to adjust:
- Stopword removal
- Lemmatization
- Text cleaning rules

### Adjusting LLM Settings
Edit `src/models/llm_handler.py` to:
- Change the base model
- Modify fine-tuning parameters
- Adjust semantic search settings

## Requirements

- Python 3.8+
- PyPDF2
- transformers
- torch
- pandas
- numpy
- scikit-learn
- spacy
- nltk
- datasets
- sentence-transformers

## License

MIT License 