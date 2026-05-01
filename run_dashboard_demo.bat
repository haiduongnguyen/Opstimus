@echo off
cd /d "%~dp0"
venv_opstimus\Scripts\python.exe visualization\dashboard.py --port 8765
