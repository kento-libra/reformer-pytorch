echo on
setlocal
cd /d %~dp0 && ^
git add . && ^
git commit -m "autopusher" && ^
git push
pause
exit /b 0