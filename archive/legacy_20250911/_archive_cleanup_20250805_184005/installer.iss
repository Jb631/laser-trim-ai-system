; Inno Setup Script for Laser Trim Analyzer
; Requires Inno Setup 6.0 or later

#define MyAppName "Laser Trim Analyzer"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "Your Company Name"
#define MyAppURL "http://www.yourcompany.com"
#define MyAppExeName "LaserTrimAnalyzer.exe"
#define MyAppAssocName "Laser Trim Data File"
#define MyAppAssocExt ".xls"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{E9F5D8A4-6B2C-4D89-9E7F-3A2B5C6D8E9F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\LaserTrimAnalyzer
DisableProgramGroupPage=yes
LicenseFile=LICENSE
; Remove the following line to run in administrative install mode
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=dist
OutputBaseFilename=LaserTrimAnalyzer_Setup_{#MyAppVersion}
SetupIconFile=assets\app_icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Setup
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode
Name: "fileassoc"; Description: "Associate Excel files with {#MyAppName}"; GroupDescription: "File associations:"; Flags: unchecked

[Files]
; Main application files from PyInstaller output
Source: "dist\LaserTrimAnalyzer\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\LaserTrimAnalyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Configuration files
Source: "config\default.yaml"; DestDir: "{app}\config"; Flags: ignoreversion
Source: "config\production.yaml"; DestDir: "{app}\config"; Flags: ignoreversion
Source: "config\deployment.yaml"; DestDir: "{app}\config"; Flags: ignoreversion onlyifdoesntexist

; Documentation
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

; Create necessary directories
[Dirs]
Name: "{app}\logs"
Name: "{app}\data"
Name: "{app}\models"
Name: "{app}\temp"

; For shared deployment - create common data directories
Name: "{commonappdata}\LaserTrimAnalyzer"
Name: "{commonappdata}\LaserTrimAnalyzer\logs"
Name: "{commonappdata}\LaserTrimAnalyzer\data"
Name: "{commonappdata}\LaserTrimAnalyzer\models"
Name: "{commonappdata}\LaserTrimAnalyzer\database"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Registry]
; File association
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".xls"; ValueData: ""; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".xlsx"; ValueData: ""; Tasks: fileassoc

; Application settings
Root: HKA; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"

; For IT management - machine-wide settings
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey; Check: IsAdminInstallMode
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"; Check: IsAdminInstallMode
Root: HKLM; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "SharedDataPath"; ValueData: "{commonappdata}\LaserTrimAnalyzer"; Check: IsAdminInstallMode

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any generated files
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\temp"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{userappdata}\LaserTrimAnalyzer"

[Code]
var
  DeploymentModePage: TInputOptionWizardPage;
  ServerPathPage: TInputQueryWizardPage;
  
// Function to check if .NET Framework is installed (if needed)
function IsDotNetInstalled: Boolean;
begin
  Result := True; // Python app doesn't need .NET
end;

// Function to handle command-line silent install with config
function GetCommandlineParam(param: String): String;
var
  i: Integer;
  s: String;
begin
  Result := '';
  for i := 1 to ParamCount do
  begin
    s := ParamStr(i);
    if Pos('/' + param + '=', LowerCase(s)) = 1 then
    begin
      Result := Copy(s, Length(param) + 3, Length(s));
      Break;
    end;
  end;
end;

// Initialize setup
function InitializeSetup(): Boolean;
var
  ConfigFile: String;
begin
  Result := True;
  
  // Check for silent install config
  ConfigFile := GetCommandlineParam('CONFIG');
  if ConfigFile <> '' then
  begin
    // Handle custom config file for deployment
    // This would copy the specified config to the installation
  end;
end;

// Create custom wizard pages
procedure InitializeWizard();
begin
  // Create deployment mode selection page
  DeploymentModePage := CreateInputOptionPage(wpSelectDir,
    'Deployment Mode', 'Choose how the application will be used',
    'Please select the deployment mode for Laser Trim Analyzer:',
    True, False);
    
  DeploymentModePage.Add('Single User Mode (Local Database)');
  DeploymentModePage.Add('Multi-User Mode (Network Database)');
  
  // Default to single user mode
  DeploymentModePage.SelectedValueIndex := 0;
  
  // Create server path input page
  ServerPathPage := CreateInputQueryPage(DeploymentModePage.ID,
    'Network Database Configuration', 'Specify the network database location',
    'Please enter the UNC path to the shared database location (e.g., \\server\share\LaserTrimAnalyzer):');
    
  ServerPathPage.Add('Database Server Path:', False);
  ServerPathPage.Values[0] := '\\fileserver\QA_Data\LaserTrimAnalyzer';
end;

// Skip server path page if single user mode is selected
function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  
  // Skip server path page if single user mode is selected
  if PageID = ServerPathPage.ID then
  begin
    Result := DeploymentModePage.SelectedValueIndex = 0;
  end;
end;

// Update configuration based on selected mode
procedure UpdateDeploymentConfig();
var
  ConfigFile: String;
  ConfigContent: TStringList;
  DeploymentMode: String;
  DatabasePath: String;
begin
  ConfigFile := ExpandConstant('{app}\config\deployment.yaml');
  ConfigContent := TStringList.Create;
  
  try
    // Determine selected mode
    if DeploymentModePage.SelectedValueIndex = 0 then
      DeploymentMode := 'single_user'
    else
      DeploymentMode := 'multi_user';
      
    // Load existing config
    if FileExists(ConfigFile) then
      ConfigContent.LoadFromFile(ConfigFile);
      
    // Update deployment mode
    ConfigContent.Text := StringReplace(ConfigContent.Text,
      'deployment_mode: "single_user"',
      'deployment_mode: "' + DeploymentMode + '"',
      [rfIgnoreCase]);
      
    // Update network path if multi-user mode
    if DeploymentMode = 'multi_user' then
    begin
      DatabasePath := ServerPathPage.Values[0] + '\shared_database.db';
      ConfigContent.Text := StringReplace(ConfigContent.Text,
        '//fileserver/QA_Data/LaserTrimAnalyzer/shared_database.db',
        StringReplace(DatabasePath, '\', '/', [rfReplaceAll]),
        [rfIgnoreCase]);
    end;
    
    // Save updated config
    ConfigContent.SaveToFile(ConfigFile);
    
  finally
    ConfigContent.Free;
  end;
end;

// Post-install configuration
procedure CurStepChanged(CurStep: TSetupStep);
var
  DatabasePath: String;
  LocalDataPath: String;
begin
  if CurStep = ssPostInstall then
  begin
    // Update deployment configuration
    UpdateDeploymentConfig();
    
    // Create necessary directories based on mode
    if DeploymentModePage.SelectedValueIndex = 0 then
    begin
      // Single user mode - create local directories
      LocalDataPath := ExpandConstant('{localappdata}\LaserTrimAnalyzer');
      CreateDir(LocalDataPath);
      CreateDir(LocalDataPath + '\database');
      CreateDir(LocalDataPath + '\logs');
      CreateDir(LocalDataPath + '\models');
      CreateDir(LocalDataPath + '\data');
    end
    else
    begin
      // Multi-user mode - create common directories if admin
      if IsAdminInstallMode then
      begin
        DatabasePath := ExpandConstant('{commonappdata}\LaserTrimAnalyzer');
        CreateDir(DatabasePath);
        CreateDir(DatabasePath + '\logs');
        // Note: Network directories should be created on the server
      end;
    end;
  end;
end;