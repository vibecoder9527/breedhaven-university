@echo off
where py >nul 2>nul && (
  py -3 "%~dp0bhu_llm_translate.py"
  if errorlevel 1 pause
  goto :eof
)
where python >nul 2>nul && (
  python "%~dp0bhu_llm_translate.py"
  if errorlevel 1 pause
  goto :eof
)
echo Python not found. Install Python 3 and enable Add to PATH.
echo https://www.python.org/downloads/windows/
pause
