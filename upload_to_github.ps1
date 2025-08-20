# NeuroBM GitHub Upload Script (PowerShell)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NeuroBM GitHub Upload Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "neurobm")) {
    Write-Host "❌ ERROR: neurobm directory not found!" -ForegroundColor Red
    Write-Host "Please run this script from the NeuroBM project root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Found NeuroBM project directory" -ForegroundColor Green
Write-Host ""

# Step 1: Initialize git repository if needed
Write-Host "🔧 Step 1: Initializing Git repository..." -ForegroundColor Yellow
try {
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
} catch {
    Write-Host "❌ Git init failed. Make sure Git is installed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Step 2: Check if remote origin exists
Write-Host "🔧 Step 2: Checking remote repository..." -ForegroundColor Yellow
try {
    $remoteUrl = git remote get-url origin 2>$null
    if ($remoteUrl) {
        Write-Host "✅ Remote origin already exists: $remoteUrl" -ForegroundColor Green
    } else {
        Write-Host "🔧 Adding remote origin..." -ForegroundColor Yellow
        git remote add origin https://github.com/ai-in-pm/NeuroBM.git
        Write-Host "✅ Remote origin added" -ForegroundColor Green
    }
} catch {
    Write-Host "🔧 Adding remote origin..." -ForegroundColor Yellow
    git remote add origin https://github.com/ai-in-pm/NeuroBM.git
    Write-Host "✅ Remote origin added" -ForegroundColor Green
}
Write-Host ""

# Step 3: Configure git user (if not already configured)
Write-Host "🔧 Step 3: Configuring Git user..." -ForegroundColor Yellow
try {
    $userName = git config user.name 2>$null
    if (-not $userName) {
        Write-Host "🔧 Setting Git username..." -ForegroundColor Yellow
        git config --global user.name "ai-in-pm"
        Write-Host "✅ Git username set to 'ai-in-pm'" -ForegroundColor Green
    } else {
        Write-Host "✅ Git username already configured: $userName" -ForegroundColor Green
    }
    
    $userEmail = git config user.email 2>$null
    if (-not $userEmail) {
        Write-Host "🔧 Setting Git email..." -ForegroundColor Yellow
        $email = Read-Host "Enter your email address"
        git config --global user.email $email
        Write-Host "✅ Git email configured" -ForegroundColor Green
    } else {
        Write-Host "✅ Git email already configured: $userEmail" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Git configuration may need manual setup" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Add all files
Write-Host "🔧 Step 4: Adding all files to Git..." -ForegroundColor Yellow
try {
    git add .
    Write-Host "✅ All files added to staging area" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to add files" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Step 5: Show status
Write-Host "📋 Git Status:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Step 6: Create commit
Write-Host "🔧 Step 5: Creating initial commit..." -ForegroundColor Yellow
$commitMessage = @"
Initial commit: Complete NeuroBM implementation

- Core platform with RBM, DBM, and CRBM models
- Synthetic data generation for multiple cognitive regimes  
- Training infrastructure with callbacks and evaluation
- Interpretability tools and saliency analysis
- Interactive dashboards for monitoring and visualization
- Comprehensive automation system for research integration
- Weekly release pipeline with quality gates
- Ethical framework and responsible AI guidelines
- Complete documentation and test suite
- Ready for automated releases starting August 20, 2025
"@

try {
    git commit -m $commitMessage
    Write-Host "✅ Initial commit created" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create commit" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Step 7: Ensure main branch
Write-Host "🔧 Step 6: Ensuring main branch..." -ForegroundColor Yellow
try {
    git branch -M main
    Write-Host "✅ Main branch ready" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Branch setup may need manual attention" -ForegroundColor Yellow
}
Write-Host ""

# Step 8: Push to GitHub
Write-Host "🚀 Step 7: Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "This may take a few minutes depending on your internet connection..." -ForegroundColor Cyan
try {
    git push -u origin main
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "🎉 SUCCESS! NeuroBM uploaded to GitHub" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} catch {
    Write-Host "❌ Push failed. This might be due to:" -ForegroundColor Red
    Write-Host "  - Authentication issues (you may need to enter username/password)" -ForegroundColor Yellow
    Write-Host "  - Network connectivity" -ForegroundColor Yellow
    Write-Host "  - Repository permissions" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔧 Trying alternative push method..." -ForegroundColor Yellow
    try {
        git push --set-upstream origin main
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "🎉 SUCCESS! NeuroBM uploaded to GitHub" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } catch {
        Write-Host "❌ Alternative push also failed" -ForegroundColor Red
        Write-Host ""
        Write-Host "💡 Manual steps needed:" -ForegroundColor Cyan
        Write-Host "1. Check your GitHub authentication" -ForegroundColor White
        Write-Host "2. Ensure you have write access to the repository" -ForegroundColor White
        Write-Host "3. Try running: git push -u origin main" -ForegroundColor White
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "📍 Repository URL: https://github.com/ai-in-pm/NeuroBM" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 What was uploaded:" -ForegroundColor Cyan
Write-Host "✅ Complete NeuroBM platform" -ForegroundColor Green
Write-Host "✅ Automation system for weekly releases" -ForegroundColor Green
Write-Host "✅ Comprehensive documentation" -ForegroundColor Green
Write-Host "✅ Test suite and validation tools" -ForegroundColor Green
Write-Host "✅ Ethical framework and guidelines" -ForegroundColor Green
Write-Host "✅ Interactive dashboards and notebooks" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Next steps:" -ForegroundColor Cyan
Write-Host "1. Visit your repository: https://github.com/ai-in-pm/NeuroBM" -ForegroundColor White
Write-Host "2. Review the uploaded files" -ForegroundColor White
Write-Host "3. Set up branch protection rules (optional)" -ForegroundColor White
Write-Host "4. Configure automation system for weekly releases" -ForegroundColor White
Write-Host "5. Invite collaborators if needed" -ForegroundColor White
Write-Host ""
Write-Host "📅 Automation system ready for weekly releases starting August 20, 2025" -ForegroundColor Magenta
Write-Host ""
Read-Host "Press Enter to exit"
