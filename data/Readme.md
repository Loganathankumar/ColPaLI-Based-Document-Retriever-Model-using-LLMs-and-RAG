# ColPaLI-Based Document Retriever for Scanned PDFs

A comprehensive implementation of a semantic search system for scanned PDF documents using **ColPaLI** (Collection of Patches for Language and Image) - a vision-language retriever that treats each page as an image.

## What This Project Does

This project demonstrates how to build an end-to-end document retrieval system that can:

- **Search scanned PDFs without OCR** - No need to extract text first
- **Understand visual elements** - Captures tables, charts, diagrams, and layout
- **Perform semantic search** - Finds relevant pages based on meaning, not just keywords
- **Answer questions** - Connects to vision QA models for question-answering

## Key Features

### Vision-Based Document Retrieval
- Converts PDF pages to images for processing
- Uses ColPaLI's multi-vector embeddings (128-dim patches per page)
- Implements "late interaction" scoring (ColBERT-style) for better matching

### Semantic Search Capabilities
- Indexes document embeddings using FAISS for fast similarity search
- Supports natural language queries
- Returns relevant page images based on semantic similarity

### Question Answering Integration
- Connects to Pix2Struct (DocVQA model) for answer extraction
- Processes retrieved page images to generate answers
- Handles complex layouts, tables, and visual information

## Technical Stack

- **ColPaLI v1.3** - Vision-language retriever model (~3B parameters)
- **FAISS** - Vector similarity search and indexing
- **Pix2Struct** - Document visual question-answering model
- **Transformers** - Hugging Face model loading and processing
- **PDF2Image** - PDF to image conversion
- **NumPy/PyTorch** - Numerical computations

## Use Cases

This system is particularly useful for:

- **Research Papers** - Finding relevant sections in academic documents
- **Technical Manuals** - Locating specific procedures or diagrams
- **Financial Reports** - Searching through charts and tables
- **Legal Documents** - Finding relevant clauses or sections
- **Medical Records** - Locating specific test results or images
- **Any scanned document** - Where OCR might miss visual context

## Why This Approach is Powerful

### Traditional OCR Limitations:
- Loses visual layout information
- Struggles with tables, charts, and diagrams
- Requires perfect text extraction
- Misses visual context

### ColPaLI Advantages:
- Preserves visual information
- Understands tables, charts, and layout
- Works with imperfect scans
- Captures semantic meaning from visuals
- No OCR preprocessing required

## Project Structure

```
colpali-retriever/
├── colpali-based-retriever.ipynb    # Main implementation notebook
├── README.md                        # This file
└── .venv/                           # Virtual environment
```

## Getting Started

### Prerequisites
- Python 3.8+
- 12GB+ RAM (recommended for model loading)
- GPU with 12GB+ VRAM (optional but recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/markuskuehnle/colpali-document-retriever
cd colpali-retriever

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install transformers accelerate huggingface-hub pillow pdf2image faiss-cpu torch
```

### Usage
1. Open `colpali-based-retriever.ipynb` in Jupyter
2. Run cells sequentially to:
   - Load the ColPaLI model
   - Convert your PDF to images
   - Generate embeddings
   - Create search index
   - Perform semantic search
   - Answer questions using retrieved pages

## Implementation Steps

The notebook implements a complete pipeline:

1. **Model Setup** - Load ColPaLI model and processor
2. **PDF Processing** - Convert PDF pages to images
3. **Embedding Generation** - Create multi-vector embeddings for each page
4. **Indexing** - Build FAISS index for fast search
5. **Query Processing** - Encode search queries
6. **Retrieval** - Find relevant pages using semantic search
7. **Question Answering** - Extract answers from retrieved pages

## Example Workflow

```python
# 1. Load model
model = ColPaliForRetrieval.from_pretrained("vidore/colpali-v1.3-hf")

# 2. Convert PDF to images
page_images = convert_pdf_to_images("document.pdf")

# 3. Generate embeddings
page_embeddings = [model.encode_image(img) for img in page_images]

# 4. Create search index
index = build_faiss_index(page_embeddings)

# 5. Search for relevant pages
query = "correlation between samples tested and positivity rate"
relevant_pages = search_document(query, index)

# 6. Answer questions
answer = answer_question(query, relevant_pages[0])
```

## Search Quality

The system excels at finding:
- **Visual content** - Charts, graphs, diagrams
- **Layout information** - Tables, forms, structured data
- **Semantic relationships** - Related concepts across visual elements
- **Contextual information** - Information that depends on visual arrangement

## Known Issues & Solutions

### Memory Usage
- **Issue**: Large models require significant RAM
- **Solution**: Use GPU offloading or model quantization

### Model Loading
- **Issue**: First-time model download can be slow
- **Solution**: Models are cached locally after first download

## References

- [ColPaLI Paper](https://arxiv.org/abs/2407.01449) - Original research paper
- [Hugging Face ColPaLI](https://huggingface.co/vidore/colpali-v1.3-hf) - Model repository
- [Pix2Struct](https://huggingface.co/google/pix2struct-docvqa-large) - Document QA model
- [FAISS Documentation](https://github.com/facebookresearch/faiss) - Vector search library

## License

This project is for educational and research purposes. Please check the licenses of the underlying models (ColPaLI, Pix2Struct) for commercial use.

---

**Happy document searching! 🎉** 
