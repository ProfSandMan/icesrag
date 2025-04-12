# ICES RAG Search Tool

A Streamlit-based search application for exploring NASA ICES research papers using advanced retrieval methods.

## 🚀 Quick Start Guide

### Prerequisites

Before you begin, make sure you have:
- Python 3.12.0 installed
- Poetry package manager installed (see [Installation](#installation) section)

### Installation

1. **Install Poetry** (if you haven't already):
   - Windows (PowerShell):
     ```powershell
     (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
     ```
   - macOS/Linux:
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/icesrag.git
   cd icesrag
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Download NLTK data**:
   ```bash
   poetry run python nltkdownload.py
   ```

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   poetry run streamlit run app.py
   ```

2. The application will open in your default web browser. If it doesn't, you can access it at:
   ```
   http://localhost:8501
   ```

## 🔍 Features

- **Multiple Retrieval Methods**:
  - Dense Retrieval (Semantic Search)
  - Sparse Retrieval (Keyword-based)
  - Composite Retrieval (Combined approach)

- **User-Friendly Interface**:
  - Clean, modern design
  - Easy-to-use search functionality
  - Detailed paper information display
  - Keyboard shortcuts (Ctrl+Enter to search)

## 🛠️ Technical Details

### Dependencies

The project uses several key Python packages with exact versions:
- Streamlit (1.40.0) - Web application framework
- ChromaDB (0.5.18) - Vector database
- Sentence Transformers (3.2.1) - Embedding generation
- NLTK (3.9.1) - Natural Language Processing
- BeautifulSoup4 (4.13.3) - HTML parsing
- Rank-BM25 (0.2.2) - Sparse retrieval
- OpenAI (1.68.2) - API integration
- Matplotlib (3.10.1) - Data visualization
- Seaborn (0.13.2) - Statistical visualization
- SciPy (1.15.2) - Scientific computing
- scikit-posthocs (0.11.3) - Statistical analysis
- ipykernel (6.29.5) - Jupyter integration
- pathlib (1.0.1) - File system operations

### Project Structure

```
icesrag/
├── app.py                 # Main Streamlit application
├── frontend/              # Frontend components and assets
├── icesrag/              # Core RAG implementation
├── data/                 # Database files
├── nltk_data_local/      # Local NLTK data
└── .streamlit/           # Streamlit configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Sam Brooks
- Ian Ortega
- Hunter Sandidge

## 📚 Resources

- [NASA ICES Paper Repository](https://hdl.handle.net/2346/58495)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Poetry Documentation](https://python-poetry.org/docs/)

## ⚠️ Troubleshooting

If you encounter any issues:

1. **Dependency Installation**:
   - Make sure you're using Python 3.12.0
   - Try running `poetry install` with the `--no-cache` flag

2. **NLTK Data**:
   - If you get NLTK-related errors, run `poetry run python nltkdownload.py` again

3. **Database Access**:
   - Ensure the `data/` directory contains the necessary database files
   - Check file permissions if you encounter database access errors

For additional help, please open an issue in the repository.
