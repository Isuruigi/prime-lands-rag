# Crawler Windows Fix

## Problem
The crawler fails in Jupyter notebooks on Windows due to asyncio subprocess limitations.

## Solution: Run as Standalone Script

Instead of running the crawler in Jupyter, use the terminal:

```powershell
# Make sure you're in the project directory with venv activated
cd "d:\Zuu Crew Agentic AI\Projects\Mini Project 2"
.venv\Scripts\activate

# Run the crawler script
python run_crawler.py
```

This will:
- ✅ Crawl 15 pages from primelands.lk
- ✅ Save to `data/primelands_corpus.jsonl`
- ✅ Avoid the Windows asyncio issue
- ✅ Take ~5-10 minutes

## After Crawling

Once the crawler finishes, you can:
1. **Skip notebook 01** (crawler already done)
2. **Continue with notebook 02** (chunking lab) - this works fine in Jupyter
3. **Continue with notebook 03** (intelligence layers) - also works in Jupyter
4. **Continue with notebook 04** (performance arena) - also works in Jupyter

## Alternative: Use Google Colab

If you prefer notebooks, you can also run the entire project on Google Colab where this asyncio issue doesn't exist!
