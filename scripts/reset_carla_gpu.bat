@echo off
REM Recover from a CARLA / Unreal GPU fatal without rebooting the laptop.
REM Double-click this on Windows, or run from an Admin cmd if services fail.
setlocal EnableExtensions

echo === Kill CARLA / Unreal leftovers ===
taskkill /F /IM CarlaUE4.exe /T 2>nul
taskkill /F /IM CarlaUE4-Win64-Shipping.exe /T 2>nul
taskkill /F /IM UE4Editor.exe /T 2>nul
taskkill /F /IM CrashReportClient.exe /T 2>nul

echo.
echo === Soft-reset NVIDIA display driver ===
echo Press Win+Ctrl+Shift+B now if the screen flickers / GPU feels wedged.
echo (That key combo restarts the display driver in ~1 second.)
echo.

echo === Restart NVIDIA container services (safe if CARLA is closed) ===
net stop "NVIDIA Display Container LS" >nul 2>&1
net start "NVIDIA Display Container LS" >nul 2>&1
net stop "NVIDIA Container" >nul 2>&1
net start "NVIDIA Container" >nul 2>&1

echo.
echo Waiting a few seconds for the driver to settle ...
timeout /t 3 /nobreak >nul

echo.
echo === Processes still holding the GPU (ignore Desktop Window Manager) ===
powershell -NoProfile -Command ^
  "Get-Process | Where-Object { $_.ProcessName -match 'Carla|UE4|python|chrome|Discord' } | Select-Object ProcessName,Id,WorkingSet64 | Format-Table -AutoSize"

echo.
echo Done. Relaunch CARLA with low settings, e.g.:
echo   CarlaUE4.exe -dx11 -quality-level=Low -windowed -ResX=800 -ResY=600
echo.
echo If the next CARLA start still fatals immediately, THEN reboot.
pause
