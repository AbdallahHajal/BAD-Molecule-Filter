; -- BADMOLFILTER.iss --
; Packaging For Windows

[Setup]
AppName=BAD Molecule Filter
AppVersion=1.1
WizardStyle=modern
DefaultDirName={autopf}\BAD-Molecule-Filter
DefaultGroupName=BAD Molecule Filter
UninstallDisplayIcon={app}\Uninstall.exe
Compression=lzma2
SolidCompression=yes
OutputDir=userdocs:BAD Molecule Filter
PrivilegesRequired=admin
AppCopyright=GNU General Public License v3.0
LicenseFile=license.txt

[Dirs]

Name: "{app}\templates"
Name: "{app}\guides"
Name: "{app}\test"
Name: "{app}\uploads"
Name: "{app}\__pycache__"

[Files]
Source: "license.txt"; DestDir: "{app}"; Components:main/server
Source: "app.py"; DestDir: "{app}"; Components:main/server
Source: "python.bat"; DestDir: "{app}"; Components:other/python
Source: "dep.bat"; DestDir: "{app}"; Components: other/python
Source: "server.bat"; DestDir: "{app}"; Components:main/server
Source: "requirements.txt"; DestDir: "{app}";Components: other/python
Source: "guides\Installation_Guide.ipynb"; DestDir: "{app}\guides"; Components: main/guide
Source: "templates\BAD Molecule Filter.html"; DestDir: "{app}\templates"; Components: main/server 
Source: "test\test.xlsx"; DestDir: "{app}\test"; Components: other/data
Source: "MM_model.pkl"; DestDir: "{app}";  Components: other/data
Source: "scaler_MM.pkl"; DestDir: "{app}"; Components: other/data          
Source: "x_train1_MM_Binary_training_set.pkl"; DestDir: "{app}"; Components: other/data
Source: "x_train1_MM_Continuous_training_set.pkl"; DestDir: "{app}";  Components: other/data

[Components]
Name: "main"; Description: "Application"; Types: full;  Flags:fixed;
Name: "main/server"; Description: "Server Script"; Types: full compact; Flags:fixed;
Name: "main/guide"; Description: "Installation Guide"; Types: full compact ;

Name: "other"; Description: "Requirements"; Types: full;   Flags:fixed;
Name: "other/python"; Description: "Python and Dependencies"; Types: full compact;
Name: "other/data"; Description:"Application Data";Types: full compact ; Flags: fixed

[Run]
Filename: "{app}\server.bat"; Description:"Run Development Server"; Flags: runascurrentuser postinstall shellexec skipifsilent
Filename: "https://nbviewer.org/github/AbdallahHajal/BAD-Molecule-Filter/blob/main/Installation_Guide.ipynb";Description:"Open Installation Guide(Chrome/Opera/Edge)"; Flags: shellexec runasoriginaluser

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
    ErrCode: integer;
begin
    if (CurStep=ssDone) then
    begin
        ShellExec('open', 'https://nbviewer.org/github/AbdallahHajal/BAD-Molecule-Filter/blob/main/Installation_Guide.ipynb', '', '', SW_SHOW, ewNoWait, ErrCode);
    end;
end;

[Icons]
Name: "{group}\Bad Molecule Filter"; Filename: "{app}\Bad-Molecule-Filter.exe"
