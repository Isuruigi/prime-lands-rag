# 🚀 Quick Start Guide

Welcome to the Prime Lands project! Follow these steps to get started.

## Prerequisites Checklist

- [ ] Python 3.10 or higher installed
- [ ] Docker Desktop installed and running
- [ ] OpenAI API key obtained
- [ ] Anthropic API key obtained

## Setup Steps

### 1. Navigate to Project Directory
```powershell
cd "d:\Zuu Crew Agentic AI\Projects\Mini Project 2"
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
# Install package in editable mode
pip install -e ".[dev]"

# Install spaCy English model
python -m spacy download en_core_web_sm

# Install Playwright browsers
playwright install chromium
```

### 4. Configure Environment
```powershell
# Copy environment template
copy .env.example .env

# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. Start Qdrant Vector Database
```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 ^
  -v "%cd%\qdrant_storage:/qdrant/storage" ^
  qdrant/qdrant
```

### 6. Verify Qdrant is Running  
```powershell
curl http://localhost:6333/health
```

You should see: `{"title":"qdrant - vector search engine","version":"..."}`

## Running the Notebooks

```powershell
# Start Jupyter
jupyter notebook
```

Then open and run the notebooks in order:
1. `01_crawl_primelands.ipynb` - Web crawling
2. `02_chunk_lab.ipynb` - Chunking comparison
3. `03_intelligence_layers.ipynb` - RAG/CAG/CRAG
4. `04_performance_arena.ipynb` - RAGAS evaluation

## Common Issues

### Issue: Port 6333 already in use
```powershell
# Stop existing Qdrant container
docker stop qdrant
docker rm qdrant
# Then restart with the command in Step 5
```

### Issue: Playwright browser not found
```powershell
playwright install chromium --force
```

### Issue: spaCy model not found
```powershell
python -m spacy download en_core_web_sm --force
```

## Project Structure

```
prime_lands/
├── src/prime_lands/         # Source code (importable package)
├── notebooks/               # Jupyter notebooks (run these)
├── data/                    # Crawled data & chunks
├── outputs/                 # Evaluation results
├── config.yaml             # Configuration (edit if needed)
└── .env                    # API keys (keep secret!)
```

## Next Steps

1. ✅ Complete setup steps above
2. 📓 Run notebooks in sequence
3. 📊 Review outputs in `outputs/` directory
4. 📝 Write engineering report
5. 🎯 Submit project

## Need Help?

- Check `README.md` for detailed documentation
- Review `miniproject2_guide.md` for project requirements
- Ensure all API keys are correct in `.env`

Happy building! 🚀
