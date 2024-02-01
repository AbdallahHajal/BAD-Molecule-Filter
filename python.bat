@echo off
echo Installing Python....

:CheckOS
IF EXIST "%PROGRAMFILES(X86)%" (GOTO 64BIT) ELSE (GOTO 32BIT)

:64BIT
echo 64-bit...
set "version=3.9.9"
set "url=https://www.python.org/ftp/python/%version%/python-%version%-amd64.exe"
set "installer=python-%version%-amd64.exe"

REM Download Python installer
certutil -urlcache -split -f %url% %installer%

REM Wait for the download to complete
:WAIT
timeout /t 1 >nul
if not exist %installer% goto WAIT

REM Install Python
start /wait %installer% /quiet TargetDir=%ProgramFiles%\Python%version%

REM Add Python to the system PATH
set "envVariable=%PATH%"
set "installPath=%ProgramFiles%\Python%version%"
echo %envVariable% | findstr /i /c:"%installPath%" > nul || (
    setx PATH "%envVariable%;%installPath%" /M
    echo Added Python to PATH.
)

REM Clean up
del %installer%

GOTO END

:32BIT
echo 32-bit...
set "version=3.9.9"
set "url=https://www.python.org/ftp/python/%version%/python-%version%.exe"
set "installer=python-%version%.exe"

REM Download Python installer
certutil -urlcache -split -f %url% %installer%

REM Wait for the download to complete
:WAIT
timeout /t 1 >nul
if not exist %installer% goto WAIT

REM Install Python
start /wait %installer% /quiet TargetDir=%ProgramFiles%\Python%version%

REM Add Python to the system PATH
set "envVariable=%PATH%"
set "installPath=%ProgramFiles%\Python%version%"
echo %envVariable% | findstr /i /c:"%installPath%" > nul || (
    setx PATH "%envVariable%;%installPath%" /M
    echo Added Python to PATH.
)

REM Clean up
del %installer%

start dep.bat

GOTO END

:END

