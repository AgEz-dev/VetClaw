@echo off
chcp 65001 >nul
title VetClaw AI Service (Port 8000)
cd /d "%~dp0"

echo ========================================================
echo  🐾 VetClaw 临床兽医分诊 Agent 启动中...
echo  端口: 8000  ^| 契约: A2A v2.2 (/a2a/task/stream)
echo ========================================================

python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
