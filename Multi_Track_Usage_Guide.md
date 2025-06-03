# Multi-Track Analysis Page Usage Guide

## Overview
The Multi-Track Analysis page is designed for analyzing units that have multiple resistor tracks on the same substrate. This allows you to compare performance between tracks on the same unit and identify consistency issues.

## How to Use the Multi-Track Page

### Option 1: Select Track File
1. **Click "📁 Select Track File"** 
   - Choose an Excel file containing multi-track data
   - The file should have data for multiple tracks from the same unit
   - Example filename: `8340-1_12345_MultiTrack.xlsx`

### Option 2: Analyze Folder  
1. **Click "📂 Analyze Folder"**
   - Select a folder containing multiple track files
   - The system will group files by unit (model + serial)
   - You'll get a dialog to select which unit to analyze

### Option 3: From Database
1. **Click "🗄️ From Database"**
   - Browse units that are already in the database
   - Select a unit with multiple track records
   - The system will load all tracks for that unit

## What You'll See

Once you select data, the page displays:

### Unit Overview Cards
- **Unit ID**: Model and serial number
- **Track Count**: Number of tracks analyzed
- **Overall Status**: Pass/Fail for the unit
- **Track Consistency**: How consistent tracks are
- **Sigma Variation**: Coefficient of variation for sigma values
- **Linearity Variation**: Coefficient of variation for linearity
- **Validation Grade**: Overall quality score
- **Issues Found**: Number of problems detected

### Track Comparison Charts
- **Sigma Comparison**: Bar chart comparing sigma gradients
- **Linearity Comparison**: Bar chart comparing linearity errors  
- **Error Profiles**: Line chart showing error patterns

### Consistency Analysis
- Statistical analysis of track-to-track variation
- Identification of outlier tracks
- Consistency scoring and recommendations

## File Format Requirements

For multi-track analysis to work, your Excel files should contain:
- Multiple sheets or sections for different tracks
- Track identifiers (Track1, Track2, etc.)
- Sigma gradient data for each track
- Linearity measurements for each track
- Consistent unit identification (model + serial)

## Tips for Best Results

1. **Consistent Naming**: Use consistent file naming like `Model_Serial_TrackInfo.xlsx`
2. **Complete Data**: Ensure all tracks have the required measurements
3. **Database Storage**: Save single-track analyses first, then use "From Database" option
4. **Folder Organization**: Group related track files in the same folder

## Troubleshooting

**"No multi-track data found"**
- Check that your file contains data for multiple tracks
- Verify the file format matches expected structure

**"Cannot group files by unit"**
- Ensure filenames contain model and serial information
- Check that multiple files exist for the same unit

**"No units in database"**
- Run single-track analyses first to populate the database
- Ensure database connection is working

## Example Workflow

1. Run individual track analyses using the Analysis page
2. Go to Multi-Track page and click "🗄️ From Database"
3. Select a unit that shows multiple tracks available
4. Review the comparison charts and consistency analysis
5. Export reports if needed

This page is most useful when you have multiple measurements from the same physical unit and want to ensure all tracks are performing consistently. 