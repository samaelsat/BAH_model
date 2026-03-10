#!/usr/bin/env pwsh
# ─────────────────────────────────────────────────────────────────────────────
# push_to_github.ps1
# Run this from inside your project folder:
#   cd C:\Users\samae\OneDrive\Desktop\BAH_model\BAH_model2\files
#   .\push_to_github.ps1
# ─────────────────────────────────────────────────────────────────────────────

$REPO_URL = "https://github.com/samaelsat/BAH_model-2.git"
$BRANCH   = "main"

Write-Host "=== BAH Model — Push to GitHub ===" -ForegroundColor Cyan

# ── Check git is installed ────────────────────────────────────────────────────
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: git is not installed. Download from https://git-scm.com" -ForegroundColor Red
    exit 1
}

# ── Init repo if not already a git repo ──────────────────────────────────────
if (-not (Test-Path ".git")) {
    Write-Host "Initialising git repository..." -ForegroundColor Yellow
    git init
    git branch -M $BRANCH
}

# ── Set remote (update if already exists) ────────────────────────────────────
$remotes = git remote
if ($remotes -contains "origin") {
    Write-Host "Updating remote origin to $REPO_URL" -ForegroundColor Yellow
    git remote set-url origin $REPO_URL
} else {
    Write-Host "Adding remote origin $REPO_URL" -ForegroundColor Yellow
    git remote add origin $REPO_URL
}

# ── Stage all files ───────────────────────────────────────────────────────────
Write-Host "Staging all files..." -ForegroundColor Yellow
git add .

# ── Show what will be committed ───────────────────────────────────────────────
Write-Host "`nFiles to be committed:" -ForegroundColor Cyan
git status --short

# ── Commit ────────────────────────────────────────────────────────────────────
$MSG = "Add full BAH A/H recognition pipeline — VideoMAE + HuBERT + CrossModal Transformer + BiLSTM"
Write-Host "`nCommitting with message: '$MSG'" -ForegroundColor Yellow
git commit -m $MSG

# ── Push ──────────────────────────────────────────────────────────────────────
Write-Host "`nPushing to $REPO_URL ($BRANCH)..." -ForegroundColor Yellow
git push -u origin $BRANCH

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Successfully pushed to https://github.com/samaelsat/BAH_model-2" -ForegroundColor Green
} else {
    Write-Host "`n❌ Push failed. See error above." -ForegroundColor Red
    Write-Host "Common fixes listed below." -ForegroundColor Yellow
}
