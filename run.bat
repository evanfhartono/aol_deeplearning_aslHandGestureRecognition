@echo off
cd /d %~dp0

echo ========================================
echo  ASL Hand Gesture Recognition System
echo ========================================
echo.

echo Activating virtual environment...
python -m venv venv\aol_fastapi
call venv\aol_fastapi\Scripts\activate
pip install -r requirements.txt

echo.
echo Starting server...
echo Open your browser and go to: http://localhost:8000
echo Press Ctrl+C in this window to stop the server.
echo.

uvicorn main:app --reload

echo.
echo Server has stopped.
pause
