#!/usr/bin/env bash
# From WSL: kill Windows-side CarlaUE4 after a fatal GPU crash.
# Full driver reset still needs Win+Ctrl+Shift+B on the Windows desktop.

set -euo pipefail

echo "Killing CarlaUE4 on Windows ..."
taskkill.exe /F /IM CarlaUE4.exe /T 2>/dev/null || true
taskkill.exe /F /IM CarlaUE4-Win64-Shipping.exe /T 2>/dev/null || true
taskkill.exe /F /IM CrashReportClient.exe /T 2>/dev/null || true

echo
echo "Next (on Windows desktop, not in the terminal):"
echo "  1. Press Win+Ctrl+Shift+B  (screen blinks = driver reset)"
echo "  2. Relaunch CARLA low quality / dx11"
echo "  3. Only reboot if that still fatals immediately"
echo
echo "Or double-click: scripts/reset_carla_gpu.bat"
