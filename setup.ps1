# Quick Setup Script for Prime Lands Project
# Run this after dependencies are installed

Write-Host "🚀 Prime Lands Quick Setup" -ForegroundColor Cyan
Write-Host "=" * 50

# Step 1: Check if virtual environment is activated
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "⚠ Activating virtual environment..." -ForegroundColor Yellow
    & ".\.venv\Scripts\Activate.ps1"
}

# Step 2: Install spaCy model
Write-Host "`n📦 Installing spaCy English model..." -ForegroundColor Cyan
python -m spacy download en_core_web_sm

# Step 3: Install Playwright browser
Write-Host "`n🌐 Installing Playwright Chromium browser..." -ForegroundColor Cyan
playwright install chromium

# Step 4: Start Qdrant (if Docker is available)
Write-Host "`n🗄️ Checking for Docker..." -ForegroundColor Cyan
$dockerRunning = docker info 2>$null
if ($dockerRunning) {
    Write-Host "✓ Docker is running" -ForegroundColor Green
    Write-Host "Starting Qdrant..." -ForegroundColor Cyan
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 `
        -v "${PWD}\qdrant_storage:/qdrant/storage" `
        qdrant/qdrant
    Write-Host "✓ Qdrant started at http://localhost:6333" -ForegroundColor Green
} else {
    Write-Host "⚠ Docker not found or not running" -ForegroundColor Yellow
    Write-Host "  You can either:" -ForegroundColor Yellow
    Write-Host "  1. Install Docker Desktop and run this script again" -ForegroundColor Yellow
    Write-Host "  2. Use Qdrant Cloud (sign up at qdrant.tech)" -ForegroundColor Yellow
}

# Step 5: Create necessary directories
Write-Host "`n📁 Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "data", "outputs", "logs" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Final message
Write-Host "`n✅ Setup Complete!" -ForegroundColor Green
Write-Host "`n🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Start Jupyter: jupyter notebook" -ForegroundColor White
Write-Host "  2. Open notebooks/01_crawl_primelands.ipynb" -ForegroundColor White
Write-Host "  3. Run cells sequentially" -ForegroundColor White
Write-Host "`n💡 Tip: The crawler is set to 15 pages for faster testing" -ForegroundColor Yellow
