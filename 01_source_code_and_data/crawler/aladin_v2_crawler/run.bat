@echo off
chcp 65001 > nul
set PYTHON=python
for %%P in (
    "C:\Users\%USERNAME%\anaconda3\envs\aiservice26\python.exe"
    "C:\Users\%USERNAME%\anaconda3\python.exe"
    "C:\ProgramData\anaconda3\envs\aiservice26\python.exe"
) do ( if exist %%P set PYTHON=%%P )

echo [1/2] 패키지 설치...
%PYTHON% -m pip install -r requirements.txt --quiet

echo [2/2] 크롤러 시작...
%PYTHON% main.py
pause
