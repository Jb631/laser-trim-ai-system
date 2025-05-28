# GUI Application

Modern, intuitive graphical interface for the Laser Trim AI System.

## Features

### 🎯 User-Friendly Interface
- **Drag-and-Drop**: Simply drag Excel files into the application
- **One-Click Analysis**: Run complete analysis with a single button
- **Real-Time Progress**: Track analysis progress with visual feedback
- **Tabbed Interface**: Organized views for analysis, results, and history

### 📊 Analysis Features
- **Batch Processing**: Analyze multiple files at once
- **Parallel Processing**: Optional multi-threaded analysis for speed
- **ML Integration**: Automatic machine learning predictions
- **Auto-Report**: Generate reports automatically after analysis

### 🎨 Modern Design
- **Clean UI**: Professional appearance with modern styling
- **Responsive Layout**: Adapts to different window sizes
- **Hover Effects**: Interactive elements with visual feedback
- **Status Updates**: Clear communication of application state

## Installation

1. Install required dependencies:
```bash
pip install tkinterdnd2 pandas numpy scikit-learn openpyxl matplotlib seaborn
```

2. Run the launcher:
```bash
python launch_gui.py
```

## Usage

### Quick Start
1. Launch the application
2. Drag Excel files into the drop zone
3. Click "Run Analysis"
4. View results and generate reports

### Main Window Components

#### File Management Panel
- **Drop Zone**: Drag files here or click to browse
- **File List**: Shows all loaded files
- **Remove/Clear**: Manage loaded files

#### Analysis Controls
- **ML Analysis**: Toggle machine learning features
- **Auto-Report**: Generate reports automatically
- **Parallel Processing**: Speed up batch analysis

#### Quick Statistics
- Real-time summary of analysis results
- Pass/fail counts and rates
- Key metrics at a glance

### Menu Options

#### File Menu
- **Load Files** (Ctrl+O): Browse for Excel files
- **Clear All** (Ctrl+N): Remove all loaded files
- **Exit** (Ctrl+Q): Close application

#### Analysis Menu
- **Run Analysis** (F5): Start processing loaded files
- **Generate Report** (Ctrl+R): Create Excel report

#### Tools Menu
- **Settings**: Configure application preferences
- **Model Training**: Access ML model management

#### Help Menu
- **Documentation**: Open online help
- **About**: Application information

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Load Files | Ctrl+O |
| Clear All | Ctrl+N |
| Run Analysis | F5 |
| Generate Report | Ctrl+R |
| Exit | Ctrl+Q |

## Settings

### General Settings
- Auto-save analysis results
- Default output directory
- Interface preferences

### Analysis Settings
- Default sigma threshold
- Filter cutoff frequency
- Processing parameters

### ML Settings
- Enable/disable ML analysis
- Auto-retrain models
- Model directory path

## Troubleshooting

### Application Won't Start
1. Check Python version (3.7+ required)
2. Verify all dependencies are installed
3. Run `launch_gui.py` for automatic dependency check

### Drag and Drop Not Working
- Ensure tkinterdnd2 is properly installed
- Try clicking the drop zone to browse instead

### Analysis Errors
- Check Excel file format (must be .xlsx or .xls)
- Verify file contains expected data structure
- Check console for detailed error messages

## Advanced Features

### Batch Processing
1. Load multiple files using Ctrl+Click or Shift+Click
2. Enable parallel processing for faster analysis
3. All files processed in single operation

### History Tab
- View previous analysis sessions
- Access generated reports
- Track analysis trends over time

### Custom Themes
The application uses modern styling with:
- Clean white backgrounds
- Blue accent colors
- Subtle hover effects
- Professional typography

## Integration with Core Components

The GUI seamlessly integrates with:
- **Data Processor**: For file analysis
- **ML Analyzer**: For predictions
- **Report Generator**: For Excel output
- **Config Manager**: For settings persistence

## Development

### Adding New Features
1. Extend `LaserTrimAIApp` class
2. Add menu items and handlers
3. Update UI elements as needed

### Custom Dialogs
- Use `ModernButton` for consistent styling
- Follow `ProgressDialog` pattern for operations
- Maintain white/blue color scheme

### Event Handling
- Use threading for long operations
- Update UI from main thread only
- Show progress for user feedback