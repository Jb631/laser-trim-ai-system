#!/usr/bin/env python3
"""
Debug script to examine what's in the Excel limit columns.
"""

import pandas as pd
from pathlib import Path
import sys
import os
import glob

# Add src to path
sys.path.append('src')

from laser_trim_analyzer.utils.excel_utils import read_excel_sheet, find_data_columns
from laser_trim_analyzer.core.constants import SYSTEM_A_COLUMNS, SYSTEM_B_COLUMNS

def debug_excel_limits(file_path: str, sheet_name: str = None, system: str = 'A'):
    """Debug function to examine Excel file limit columns."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    print(f"Analyzing: {file_path.name}")
    
    # Try to read Excel file
    try:
        excel_file = pd.ExcelFile(file_path)
        print(f"Available sheets: {excel_file.sheet_names}")
        
        # If no sheet specified, look for data sheets
        if sheet_name is None:
            # Look for sheets that likely contain data
            data_sheets = []
            for sheet in excel_file.sheet_names:
                if any(pattern in sheet for pattern in ['TRK', 'SEC', 'test', 'Lin Error']):
                    data_sheets.append(sheet)
            
            if data_sheets:
                sheet_name = data_sheets[0]
                print(f"Found data sheet: {sheet_name}")
            else:
                sheet_name = excel_file.sheet_names[0]
                print(f"No data sheet found, using first sheet: {sheet_name}")
        
        print(f"Reading sheet: {sheet_name}")
        
        df = read_excel_sheet(file_path, sheet_name)
        print(f"Sheet shape: {df.shape}")
        
        # Find columns
        columns = find_data_columns(df, system)
        print(f"Found columns for System {system}: {columns}")
        
        if not columns:
            print("No columns found!")
            # Try other sheets if this one doesn't work
            if sheet_name != excel_file.sheet_names[0]:
                print("Trying other sheets...")
                for other_sheet in excel_file.sheet_names:
                    if other_sheet != sheet_name:
                        print(f"\nTrying sheet: {other_sheet}")
                        try:
                            df = read_excel_sheet(file_path, other_sheet)
                            columns = find_data_columns(df, system)
                            if columns:
                                sheet_name = other_sheet
                                print(f"Found columns in {other_sheet}: {columns}")
                                break
                        except:
                            continue
            
            if not columns:
                return
            
        # Check if limit columns exist
        if 'upper_limit' in columns and 'lower_limit' in columns:
            upper_col = columns['upper_limit']
            lower_col = columns['lower_limit']
            
            print(f"\nLimit columns: Upper={upper_col} (Column {chr(65+upper_col)}), Lower={lower_col} (Column {chr(65+lower_col)})")
            
            # Sample the data
            print(f"\nFirst 10 rows of upper limit column ({chr(65+upper_col)}):")
            upper_data = df.iloc[:10, upper_col]
            for i, val in enumerate(upper_data):
                print(f"  Row {i}: {val} (type: {type(val).__name__})")
                
            print(f"\nFirst 10 rows of lower limit column ({chr(65+lower_col)}):")
            lower_data = df.iloc[:10, lower_col]
            for i, val in enumerate(lower_data):
                print(f"  Row {i}: {val} (type: {type(val).__name__})")
                
            # Try to convert to numeric
            upper_numeric = pd.to_numeric(df.iloc[:, upper_col], errors='coerce')
            lower_numeric = pd.to_numeric(df.iloc[:, lower_col], errors='coerce')
            
            print(f"\nAfter numeric conversion:")
            print(f"Upper limits: {upper_numeric.dropna().tolist()[:10]}")
            print(f"Lower limits: {lower_numeric.dropna().tolist()[:10]}")
            
        else:
            print("Limit columns not found in column mapping!")
            print("Available columns in mapping:", list(columns.keys()))
            
        # Also check position and error columns
        if 'position' in columns and 'error' in columns:
            pos_col = columns['position']
            err_col = columns['error']
            
            print(f"\nData columns: Position={pos_col} (Column {chr(65+pos_col)}), Error={err_col} (Column {chr(65+err_col)})")
            
            # Sample the data
            print(f"\nFirst 5 position values:")
            pos_data = pd.to_numeric(df.iloc[:, pos_col], errors='coerce').dropna()
            print(f"  {pos_data.head().tolist()}")
            
            print(f"\nFirst 5 error values:")
            err_data = pd.to_numeric(df.iloc[:, err_col], errors='coerce').dropna()  
            print(f"  {err_data.head().tolist()}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if file path provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            print(f"=== Testing {file_path} ===")
            debug_excel_limits(file_path, system='A')
        else:
            print(f"File not found: {file_path}")
    else:
        # Look for Excel files in current directory and subdirectories
        test_files = []
        
        # Common locations to check
        locations = [
            ".",
            "D:/LaserTrimData/**",
            "D:/UserFolders/Desktop/**"
        ]
        
        for location in locations:
            test_files.extend(glob.glob(f"{location}/*.xlsx", recursive=True))
            test_files.extend(glob.glob(f"{location}/*.xls", recursive=True))
            
        if test_files:
            file_path = test_files[0]
            print(f"=== Testing first found file: {file_path} ===")
            debug_excel_limits(file_path, system='A')
        else:
            print("No Excel files found. Please specify a file path:")
            print("Usage: python debug_limits.py <file_path>") 