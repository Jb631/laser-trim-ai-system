# Laser Trim AI System - One-Click Launcher Guide

## Overview

The One-Click Launcher provides a seamless way to start the Laser Trim AI System without technical knowledge. It automatically handles all setup requirements and launches the application with a single click.

## 🚀 Quick Start

### Windows Users
1. **Double-click** `launch_laser_trim.bat`
2. Wait for the launcher window to appear
3. Click **"Launch Application"**
4. The system will start automatically

### Mac/Linux Users
1. Open Terminal in the project directory
2. Run: `./launch_laser_trim.sh`
3. Follow the on-screen instructions

## 📋 Features

### Automatic Setup
- ✅ Python version checking (3.8+ required)
- ✅ Virtual environment creation
- ✅ Dependency installation
- ✅ Component verification
- ✅ Configuration setup

### User-Friendly Interface
- 📊 Progress tracking with visual feedback
- 📝 Real-time log display
- ❌ Clear error messages
- 💡 Built-in help system

### Smart Features
- 🔄 Checks if dependencies are already installed
- 📁 Creates missing directories automatically
- ⚙️ Generates default configuration if needed
- 🖥️ Optional desktop shortcut creation

## 🛠️ Installation Methods

### Method 1: Batch File (Windows) - Recommended
```
launch_laser_trim.bat
```
- Simplest method for Windows users
- Double-click to run
- Handles all Python checks automatically

### Method 2: Python Launcher GUI
```
python launch_app.py
```
- Cross-platform compatibility
- Visual progress tracking
- Detailed logging

### Method 3: PowerShell (Windows) - Advanced
```
powershell -ExecutionPolicy Bypass -File launch_laser_trim.ps1
```
- Advanced features
- Can install Python automatically
- Interactive menu system

### Method 4: Command Line - No GUI
```
python launch_app.py --no-gui
```
- For automated deployments
- Runs without graphical interface
- Suitable for remote systems

## 📁 File Structure

```
laser-trim-ai-system/
├── launch_app.py           # Main Python launcher
├── launch_laser_trim.bat   # Windows batch launcher
├── launch_laser_trim.ps1   # PowerShell launcher (advanced)
├── launch_laser_trim.sh    # Unix/Linux/Mac launcher
├── launcher.log           # Launcher log file (created on first run)
└── venv/                  # Virtual environment (created automatically)
```

## 🔧 First-Time Setup

On first run, the launcher will:

1. **Check Python Version**
   - Requires Python 3.8 or higher
   - Displays current version
   - Provides download link if not installed

2. **Create Virtual Environment**
   - Isolates dependencies from system Python
   - Named `venv` in project root
   - Takes 30-60 seconds

3. **Install Dependencies**
   - Reads from `requirements.txt`
   - Installs all required packages
   - Shows progress in real-time
   - Takes 2-5 minutes (first time only)

4. **Verify Components**
   - Checks all required modules exist
   - Creates missing directories
   - Generates default config if needed

5. **Launch Application**
   - Starts the main GUI
   - Closes launcher after successful start

## ⚠️ Troubleshooting

### Common Issues

#### "Python is not installed or not in PATH"
**Solution:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, check ✅ "Add Python to PATH"
3. Restart your computer
4. Try launching again

#### "Failed to create virtual environment"
**Solution:**
1. Run as Administrator (Windows)
2. Check disk space (need ~500MB free)
3. Disable antivirus temporarily
4. Try manual creation: `python -m venv venv`

#### "Dependency installation failed"
**Solution:**
1. Check internet connection
2. Update pip: `python -m pip install --upgrade pip`
3. Try manual install: `pip install -r requirements.txt`
4. Check `launcher.log` for specific errors

#### "Missing components"
**Solution:**
1. Ensure all source files are present
2. Check file permissions
3. Re-download missing files
4. Verify directory structure

### Log Files

The launcher creates detailed logs in `launcher.log`:
```
2024-01-15 10:23:45 - INFO - Checking Python version...
2024-01-15 10:23:45 - INFO - Python version: 3.11.5
2024-01-15 10:23:45 - INFO - ✓ Python version OK
```

### Manual Recovery

If the launcher fails, you can set up manually:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
python src/gui/gui_application.py
```

## 🎯 Advanced Usage

### Command Line Options
```bash
# Run without GUI
python run_app.py --no-gui

# Verbose logging
python run_app.py --verbose

# Skip dependency check
python run_app.py --skip-deps
```

### Environment Variables
```bash
# Set custom venv name
set VENV_NAME=myenv

# Set custom Python path
set PYTHON_PATH=C:\Python311\python.exe
```

### Configuration

Create `launcher_config.json` for custom settings:
```json
{
    "python_min_version": [3, 8],
    "venv_name": "venv",
    "auto_launch": true,
    "create_shortcut": true,
    "check_updates": false
}
```

## 🔐 Security Considerations

- The launcher only installs packages from `requirements.txt`
- Virtual environment isolates dependencies
- No admin rights required (except for desktop shortcut)
- All operations logged for audit trail

## 📊 Performance

Typical launch times:
- **First run**: 2-5 minutes (dependency installation)
- **Subsequent runs**: 5-10 seconds
- **With dependencies cached**: 2-3 seconds

## 🆘 Getting Help

1. **Built-in Help**: Click "Help" button in launcher
2. **Logs**: Check `launcher.log` for detailed information
3. **Documentation**: See main README.md
4. **Support**: Contact QA team

## 🔄 Updating

To update the launcher:
1. Replace `launch_app.py` with new version
2. Delete `venv` folder to force clean install
3. Run launcher normally

## 💡 Tips

- **Create Desktop Shortcut**: Run PowerShell launcher and select option 3
- **Faster Startup**: Keep `venv` folder for instant launches
- **Offline Mode**: Download packages first, then work offline
- **Multiple Environments**: Rename `venv` to keep multiple setups

## 🐛 Debug Mode

For developers, enable debug mode:
```python
# In run_app.py, set:
DEBUG = True
```

This provides:
- Verbose console output
- Detailed error traces
- Component loading times
- Memory usage stats

## 📝 License

The launcher system is part of the Laser Trim AI System and follows the same MIT license terms.

---

**Remember**: The launcher is designed to "just work" - if you're reading this for troubleshooting, check the simple solutions first!