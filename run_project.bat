@echo off
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Running code...
python main.py
echo.
echo Starting Web Interface...
streamlit run app.py
pause
