@echo off
if exist "topicpulse\.git" (
    rmdir /s /q "topicpulse\.git"
)

git init
git branch -m main
git checkout -b deploy

:: Add a root gitignore to exclude environment files and pycache globally
echo .env > .gitignore
echo __pycache__/ >> .gitignore
echo venv/ >> .gitignore
echo *.pyc >> .gitignore
echo output/ >> .gitignore

git add .
git commit -m "feat: Include TopicPulse production build and refactored scrapers"

:: Remove the remote if it already exists, then add the new one
git remote remove origin 2>nul
git remote add origin https://github.com/cruspy2004/Medium-scrapper-and-sentiment-analysis-.git

:: Push to the deploy branch forcefully
git push -u origin deploy --force
