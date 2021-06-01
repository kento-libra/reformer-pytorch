echo on
setlocal
cd /d %~dp0 && ^
git add . && ^
git commit -m "autopusher" && ^
git push
exit /b 0