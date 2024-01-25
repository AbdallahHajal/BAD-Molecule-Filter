; -- BADMOLFILTER.iss --
; Packaging For Windows

[Setup]
AppName=BAD Molecule Filter
AppVersion=1.0
WizardStyle=modern
DefaultDirName={autopf}\BAD-Molecule-Filter
DefaultGroupName=BAD Molecule Filter
UninstallDisplayIcon={app}\Uninstall.exe
Compression=lzma2
SolidCompression=yes
OutputDir=userdocs:guide

[Dirs]

Name: "{app}\templates"
Name: "{app}\guides"


[Files]
Source: "app.py"; DestDir: "{app}"
Source: "setup.bat"; DestDir: "{app}"
Source: "requirements.txt"; DestDir: "{app}"
Source: "guides\Installation_Guide.ipynb"; DestDir: "{app}\guides";
Source: "templates\BAD Molecule Filter.html"; DestDir: "{app}\templates"; 
Source: "templates\index.html"; DestDir: "{app}\templates"; 
Source: "MM_model.pkl"; DestDir: "{app}"; 
Source: "model_For_Lactamase_Without_2_descriptors.pkl"; DestDir: "{app}";  
Source: "model_trial_CRUUUZAAAIN.pkl"; DestDir: "{app}"; 
Source: "model_trial_Laccccccc.pkl"; DestDir: "{app}"; 
Source: "Newest_Model_Shoichet&ZINC.pkl"; DestDir: "{app}"; 
Source: "scaler_Cruzain.pkl"; DestDir: "{app}"; 
Source: "scaler_Lactamase.pkl"; DestDir: "{app}"; 
Source: "scaler_MM.pkl"; DestDir: "{app}";           
Source: "scaler_Shoichet.pkl"; DestDir: "{app}"; 
Source: "x_train1_Cruzain_Binary_training_set.pkl"; DestDir: "{app}"; 
Source: "x_train1_Cruzain_Continuous_training_set.pkl"; DestDir: "{app}"; 
Source: "x_train1_Lactamase_Binary_training_set.pkl"; DestDir: "{app}"; 
Source: "x_train1_Lactamase_Continuous_training_set.pkl"; DestDir: "{app}"; 
Source: "x_train1_MM_Binary_training_set.pkl"; DestDir: "{app}"; 
Source: "x_train1_MM_Continuous_training_set.pkl"; DestDir: "{app}"; 
Source: "x_train1_Shoichet_Binary_training_set.pkl"; DestDir: "{app}"; 
Source: "x_train1_Shoichet_Continuous_training_set.pkl"; DestDir: "{app}"; 

[Run]
Filename: "{app}\setup.bat"; Description:"Run Development Server"; Flags: postinstall shellexec skipifsilent
Filename: "{app}\guides\Installation_Guide.ipynb";Description:"Read Installation Guide(Chrome/Opera/Edge)"; Flags: postinstall shellexec skipifsilent

[Icons]
Name: "{group}\Bad Molecule Filter"; Filename: "{app}\Bad-Molecule-Filter.exe"
