@echo off
echo ========================================
echo NeuroBM GitHub Upload Script
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "neurobm" (
    echo ERROR: neurobm directory not found!
    echo Please run this script from the NeuroBM project root directory.
    pause
    exit /b 1
)

echo ✅ Found NeuroBM project directory
echo.

REM Step 1: Initialize git repository if needed
echo 🔧 Step 1: Initializing Git repository...
git init
if %errorlevel% neq 0 (
    echo ❌ Git init failed. Make sure Git is installed.
    pause
    exit /b 1
)
echo ✅ Git repository initialized
echo.

REM Step 2: Check if remote origin exists
echo 🔧 Step 2: Checking remote repository...
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Remote origin already exists
) else (
    echo 🔧 Adding remote origin...
    git remote add origin https://github.com/ai-in-pm/NeuroBM.git
    if %errorlevel% neq 0 (
        echo ❌ Failed to add remote origin
        pause
        exit /b 1
    )
    echo ✅ Remote origin added
)
echo.

REM Step 3: Configure git user (if not already configured)
echo 🔧 Step 3: Configuring Git user...
git config user.name >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔧 Setting Git username...
    git config --global user.name "ai-in-pm"
    echo ✅ Git username set to 'ai-in-pm'
) else (
    echo ✅ Git username already configured
)

git config user.email >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔧 Setting Git email...
    set /p email="Enter your email address: "
    git config --global user.email "%email%"
    echo ✅ Git email configured
) else (
    echo ✅ Git email already configured
)
echo.

REM Step 4: Add all files
echo 🔧 Step 4: Adding all files to Git...
git add .
if %errorlevel% neq 0 (
    echo ❌ Failed to add files
    pause
    exit /b 1
)
echo ✅ All files added to staging area
echo.

REM Step 5: Show status
echo 📋 Git Status:
git status --short
echo.

REM Step 6: Create commit
echo 🔧 Step 5: Creating initial commit...
git commit -m "Initial commit: Complete NeuroBM implementation

- Core platform with RBM, DBM, and CRBM models
- Synthetic data generation for multiple cognitive regimes  
- Training infrastructure with callbacks and evaluation
- Interpretability tools and saliency analysis
- Interactive dashboards for monitoring and visualization
- Comprehensive automation system for research integration
- Weekly release pipeline with quality gates
- Ethical framework and responsible AI guidelines
- Complete documentation and test suite
- Ready for automated releases starting August 20, 2025"

if %errorlevel% neq 0 (
    echo ❌ Failed to create commit
    pause
    exit /b 1
)
echo ✅ Initial commit created
echo.

REM Step 7: Check if main branch exists, create if needed
echo 🔧 Step 6: Ensuring main branch...
git branch -M main
echo ✅ Main branch ready
echo.

REM Step 8: Push to GitHub
echo 🚀 Step 7: Pushing to GitHub...
echo This may take a few minutes depending on your internet connection...
git push -u origin main
if %errorlevel% neq 0 (
    echo ❌ Push failed. This might be due to:
    echo   - Authentication issues (you may need to enter username/password)
    echo   - Network connectivity
    echo   - Repository permissions
    echo.
    echo 🔧 Trying alternative push method...
    git push --set-upstream origin main
    if %errorlevel% neq 0 (
        echo ❌ Alternative push also failed
        echo.
        echo 💡 Manual steps needed:
        echo 1. Check your GitHub authentication
        echo 2. Ensure you have write access to the repository
        echo 3. Try running: git push -u origin main
        pause
        exit /b 1
    )
)
echo.

echo ========================================
echo 🎉 SUCCESS! NeuroBM uploaded to GitHub
echo ========================================
echo.
echo 📍 Repository URL: https://github.com/ai-in-pm/NeuroBM
echo.
echo 📋 What was uploaded:
echo ✅ Complete NeuroBM platform
echo ✅ Automation system for weekly releases
echo ✅ Comprehensive documentation
echo ✅ Test suite and validation tools
echo ✅ Ethical framework and guidelines
echo ✅ Interactive dashboards and notebooks
echo.
echo 🚀 Next steps:
echo 1. Visit your repository: https://github.com/ai-in-pm/NeuroBM
echo 2. Review the uploaded files
echo 3. Set up branch protection rules (optional)
echo 4. Configure automation system for weekly releases
echo 5. Invite collaborators if needed
echo.
echo 📅 Automation system ready for weekly releases starting August 20, 2025
echo.
pause
