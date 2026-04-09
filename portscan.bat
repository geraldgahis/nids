@echo off
set TARGET=%1
if "%TARGET%"=="" set TARGET=192.168.1.1

echo.
echo ======================================================
echo       NIDS LIVE DEMO - nmap Port Scan (Windows)
echo       Target: %TARGET%
echo ======================================================
echo.
echo Check your NIDS Dashboard at http://localhost:8000
pause

echo.
echo [ %TIME% ] Launching nmap SYN scan on %TARGET%...

:: Check if nmap exists
where nmap >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: nmap not found. Simulating scan for demo...
    echo.
    for %%P in (22, 80, 443, 3306, 5432, 8080) do (
        echo [ %TIME% ] Scanning port %%P...
        timeout /t 1 >nul
    )
    echo.
    echo PORT      STATE   SERVICE
    echo 22/tcp    open    ssh
    echo 80/tcp    open    http
    echo 443/tcp   open    https
    echo.
    echo Nmap done: 1 IP address scanned in 3.21 seconds.
) else (
    nmap -sS -T4 -p 1-10000 --open %TARGET%
)

echo.
echo ======================================================
echo SCAN COMPLETE. CHECK DASHBOARD FOR HIGH ALERTS.
echo ======================================================
pause