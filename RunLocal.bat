@echo off
REM Activate the virtual environment
call myenv\Scripts\activate

REM Run the Uvicorn server
uvicorn App:app --reload