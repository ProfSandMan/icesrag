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
   poetry run streamlit run icesrag/app.py
   ```

2. The application will open in your default web browser. If it doesn't, you can access it at:
   ```
   http://localhost:8501
   ```

## 📖 Using the Streamlit Frontend

### Getting Started

1. **Launch the app** using the command above
2. Click the **"Quick Start Guide"** button on the main page for an overview
3. Enter your search query in the text area

### Search Interface

#### Main Search Bar
- **Location**: Large text area at the top of the page
- **Usage**: Type your search query (supports multi-line queries)
- **Keyboard Shortcut**: Press `Ctrl+Enter` (or `Cmd+Enter` on Mac) to submit your search
- **Search Button**: Click the 🔍 button to the right of the search bar

#### Sidebar Controls

The left sidebar provides several configuration options:

1. **Number of Papers to Retrieve**
   - Use the slider to select how many results you want (1-30 papers)
   - Default: 10 papers

2. **Retriever Type Selection**
   Choose from three retrieval methods:
   
   - **Dense Retrieval (Vanilla RAG)**
     - Uses AI-generated embeddings to match the semantic meaning of your query
     - Best for: Conceptual questions, natural language queries, finding papers by topic or research area
     - Example: "How do life support systems handle CO2 removal in space habitats?"
   
   - **Sparse Retrieval (BM25)**
     - Matches keywords directly using traditional information retrieval
     - Best for: Specific technical terms, known jargon, exact phrase matching
     - Example: "ECLSS thermal control subsystem"
   
   - **Composite Retrieval (RRF)**
     - Combines both Dense and Sparse methods using Reciprocal Rank Fusion
     - Best for: General use when you want balanced results
     - Balances the precision of keyword matching with the flexibility of semantic search

3. **HyDE Search (Optional)**
   - **Toggle**: Enable "HyDE Search" to use Hypothetical Document Embedding
   - **What it does**: Uses an LLM to transform your query into a hypothetical research paper abstract before searching
   - **Requires**: OpenAI API key (enter in the text field that appears when toggled on)
   - **Best for**: Complex research questions that benefit from query expansion
   - **Note**: This feature requires an OpenAI API key and will make API calls (costs may apply)

### Viewing Results

Once you submit a search:

1. **Results Display**: Papers are shown one at a time with:
   - **Title**: Paper title
   - **Abstract**: Full abstract text
   - **Metadata Panel** (right side):
     - Paper URL (clickable link)
     - Authors
     - Keywords
     - Publication date

2. **Navigation**:
   - Use **⬅️ Previous** and **Next ➡️** buttons to browse through results
   - Page indicator shows your current position (e.g., "Page 3 of 15")
   - Results are paginated, showing one paper per page

3. **Search Tips**:
   - **Use longer, more specific queries** for better results
   - Try using your actual research question or working abstract
   - Example of a good query:
     > "I am exploring whether or not human beings can survive on Mars. The focus of my research is on whether or not Mars has the chemical compounds necessary to support human life. I care both about the elemental composition and compound structures we have discovered so far on the planet."
   - Adjust the number of papers if you want more or fewer results
   - Try different retrieval methods if initial results aren't relevant

### Important Notes

- **Data Source**: All results come from the [NASA ICES Paper Repository](https://hdl.handle.net/2346/58495)
- **HyDE Search**: Requires a valid OpenAI API key and will incur API costs
- **Database**: Ensure the `data/` directory contains the required database files (`ices.db`, `chroma.db`, `bm25.db`)

## 🔍 Features

- **Multiple Retrieval Methods**:
  - **Dense Retrieval**: Semantic search using AI embeddings for conceptual matching
  - **Sparse Retrieval**: Keyword-based BM25 search for precise term matching
  - **Composite Retrieval**: Hybrid approach combining both methods via Reciprocal Rank Fusion

- **Advanced Search Options**:
  - **HyDE (Hypothetical Document Embedding)**: Query expansion using LLMs to improve search quality
  - Configurable result count (1-30 papers)
  - Real-time retrieval method switching

- **User-Friendly Interface**:
  - Clean, modern dark theme design
  - Intuitive sidebar controls
  - Detailed paper metadata display
  - One-at-a-time result browsing with navigation
  - Keyboard shortcuts (Ctrl+Enter to search)
  - Quick Start Guide dialog for new users

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
├── icesrag/                    # Main package directory
│   ├── app.py                  # Streamlit application entry point
│   ├── app_retrievers.py       # Retriever initialization and setup
│   ├── corpus_builder/         # Corpus building and data scraping
│   │   ├── main_builder.py     # Main corpus building pipeline
│   │   ├── scrape_ices_repo.py  # Repository scraping
│   │   ├── embed_abstracts.py  # Abstract embedding generation
│   │   └── ...
│   ├── load/                   # Data loading and storage
│   │   ├── package/            # Data packaging strategies
│   │   ├── store/              # Database storage implementations
│   │   └── pipeline/           # Loading pipeline
│   ├── retrieve/               # Retrieval system
│   │   ├── retrievers/         # Retriever implementations (Chroma, SQLite)
│   │   ├── rerank/             # Re-ranking strategies (RRF)
│   │   └── pipeline/           # Retrieval pipeline
│   ├── utils/                  # Utility modules
│   │   ├── embed/              # Embedding engines
│   │   ├── text_preprocess/    # Text preprocessing
│   │   ├── hyde.py             # HyDE implementation
│   │   └── llms.py             # LLM interfaces
│   └── .streamlit/             # Streamlit configuration
│       └── config.toml          # Theme and server settings
├── assets/                      # Static assets (logos, images)
├── data/                        # Database files
│   ├── ices.db                  # Main SQLite database
│   ├── chroma.db/               # ChromaDB vector database
│   └── bm25.db                  # BM25 sparse index
├── evaluation/                  # Evaluation scripts and notebooks
│   ├── 2025 (sparse vs dense)/  # 2025 evaluation experiments
│   └── 2026 (hyde)/             # 2026 HyDE evaluation experiments
├── nltk_data_local/             # Local NLTK data files
├── app.py                       # (Legacy - use icesrag/app.py instead)
├── pyproject.toml               # Poetry dependencies
└── README.md                    # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Sam Brooks
- Lilly Hanslik
- Ian Ortega
- Hunter Sandidge
- Joseph Wall

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
   - Ensure the `data/` directory contains the necessary database files:
     - `ices.db` - Main database with paper abstracts
     - `chroma.db/` - ChromaDB vector database for dense retrieval
     - `bm25.db` - BM25 index for sparse retrieval
   - Check file permissions if you encounter database access errors

4. **Streamlit App Path**:
   - Make sure you're running `streamlit run icesrag/app.py` (not `app.py` from root)
   - The app is located in the `icesrag/` package directory

5. **HyDE Search Issues**:
   - Ensure you have a valid OpenAI API key
   - Check your API key has sufficient credits/quota
   - Verify the API key is entered correctly in the sidebar

For additional help, please open an issue in the repository.
