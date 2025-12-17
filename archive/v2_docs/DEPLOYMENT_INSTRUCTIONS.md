# Laser Trim Analyzer - Deployment Instructions

## Version 2.0.0

### Package Contents

The deployment package contains everything needed to run the Laser Trim Analyzer application without installation.

```
dist/LaserTrimAnalyzer/
├── LaserTrimAnalyzer.exe     # Main executable
├── config/                    # Configuration files
│   ├── deployment.yaml        # Default deployment config
│   ├── development.yaml       # Development config
│   └── production.yaml        # Production config
├── models/                    # ML models folder (created on first run)
├── README.md                  # User documentation
├── CHANGELOG.md              # Version history
├── CLAUDE.md                 # Technical documentation
└── [various DLL files]       # Required libraries
```

### Deployment Steps

#### 1. Copy to Work Computer

1. Copy the entire `LaserTrimAnalyzer` folder from `dist/` to your work computer
2. Place it in any location (e.g., `D:\Applications\LaserTrimAnalyzer`)
3. No installation or admin rights required

#### 2. Configure Database Location

The application now supports configurable database locations:

**Option A: Use Settings Page (Recommended)**
1. Run `LaserTrimAnalyzer.exe`
2. Go to Settings page
3. In Database section, click "Browse" to select database location
4. Choose from:
   - **Portable** - Database travels with the app (`./data/laser_trim.db`)
   - **Documents** - User's Documents folder
   - **Custom** - Any location you specify (e.g., network drive)
5. Click "Save Settings"
6. Restart the application

**Option B: Edit Configuration File**
1. Edit `config/deployment.yaml`
2. Change the database path:
   ```yaml
   database:
     path: "//network/share/laser_trim.db"  # Network location
     # OR
     path: "./data/laser_trim.db"           # Portable
     # OR
     path: "D:/LaserTrimData/production.db" # Fixed location
   ```

#### 3. Create Desktop Shortcut (Optional)

1. Right-click `LaserTrimAnalyzer.exe`
2. Select "Create shortcut"
3. Move shortcut to Desktop
4. Optionally rename to "Laser Trim Analyzer"

### Database Migration

If you have existing data in a different database:

1. Copy your existing database file to the new location
2. Update the path in Settings or config file
3. The application will automatically use the existing data

### Network Database Setup (Multi-User)

For shared database access across multiple users:

1. Place database on network share (e.g., `//server/laser_data/laser_trim.db`)
2. Ensure all users have read/write access to the location
3. Configure each installation to point to the same database
4. Database will handle concurrent access automatically

### ML Models

Machine Learning models are now persistent:

- Models are saved in the `models/` folder
- They persist between sessions automatically
- Models can be retrained from the ML Tools page
- Trained models are portable with the application

### Updating the Application

When a new version is available:

1. Download the new deployment package
2. **Backup current installation** (especially `models/` folder and database)
3. Replace the old files with new ones:
   - Copy new `LaserTrimAnalyzer.exe`
   - Copy any updated DLLs
   - Keep your existing:
     - Database file
     - Settings (`settings.json` in AppData)
     - Trained models (`models/` folder)
4. Run the updated application

### Environment-Specific Configuration

The application supports three environments:

1. **Production** - For daily use with real data
2. **Development** - For testing new features
3. **Deployment** - Default portable configuration

To switch environments:
- Create a batch file with: `set LTA_ENV=production && LaserTrimAnalyzer.exe`
- Or use the Settings page to configure paths

### File Locations

The application uses these locations:

- **Settings**: `%LOCALAPPDATA%/LaserTrimAnalyzer/settings.json`
- **Logs**: `%LOCALAPPDATA%/LaserTrimAnalyzer/logs/`
- **Cache**: `%LOCALAPPDATA%/LaserTrimAnalyzer/cache/`
- **Database**: Configurable (see above)
- **ML Models**: `./models/` (with the executable)

### Troubleshooting

**Application won't start:**
- Check Windows Defender hasn't quarantined the exe
- Try running as administrator once
- Check the logs in `%LOCALAPPDATA%/LaserTrimAnalyzer/logs/`

**Database errors:**
- Verify database path exists and is accessible
- Check file permissions
- For network databases, ensure network connection

**ML features not working:**
- Ensure ML is enabled in Settings
- Check that models folder exists
- Retrain models from ML Tools page if needed

**Missing DLL errors:**
- Ensure all files from dist folder are present
- Don't delete any DLL files
- May need Visual C++ Redistributables (usually pre-installed)

### Features Overview

All features are now fully functional:

- **File Processing**: Batch and individual Excel file analysis
- **Database Storage**: All results saved with configurable location
- **ML Predictions**: Failure prediction and anomaly detection
- **Visualizations**: Interactive charts and plots
- **Export**: Excel reports with all analyses
- **Drag & Drop**: Direct file dropping for processing
- **Multi-Track Analysis**: Support for multiple track data
- **Historical Analysis**: Trend analysis over time
- **Settings Persistence**: All settings saved between sessions

### Support

For issues or questions:
- Check the CHANGELOG.md for recent changes
- Review logs in `%LOCALAPPDATA%/LaserTrimAnalyzer/logs/`
- Contact your IT administrator for network setup

### Version History

See CHANGELOG.md for detailed version history and changes.

---

*Last Updated: 2025-08-06*
*Version: 2.0.0*