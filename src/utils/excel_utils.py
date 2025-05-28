"""
Excel file utilities.

Helper functions for working with Excel files, including cell reference
conversion and data extraction from specific cells.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Any, Tuple, Optional, List, Union
import re

logger = logging.getLogger(__name__)


def cell_to_indices(cell_ref: str) -> Tuple[int, int]:
    """
    Convert Excel cell reference to row and column indices.

    Args:
        cell_ref: Excel cell reference (e.g., 'B26', 'AA10')

    Returns:
        Tuple of (row_index, col_index) as 0-based indices

    Examples:
        >>> cell_to_indices('A1')
        (0, 0)
        >>> cell_to_indices('B26')
        (25, 1)
        >>> cell_to_indices('AA10')
        (9, 26)
    """
    # Separate letters and numbers
    match = re.match(r'([A-Z]+)(\d+)', cell_ref.upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")

    col_letters, row_num = match.groups()

    # Convert column letters to index (A=0, B=1, ..., AA=26, ...)
    col_idx = 0
    for i, letter in enumerate(reversed(col_letters)):
        col_idx += (ord(letter) - ord('A') + 1) * (26 ** i)
    col_idx -= 1  # Convert to 0-based

    # Convert row to 0-based index
    row_idx = int(row_num) - 1

    return row_idx, col_idx


def indices_to_cell(row: int, col: int) -> str:
    """
    Convert row and column indices to Excel cell reference.

    Args:
        row: 0-based row index
        col: 0-based column index

    Returns:
        Excel cell reference (e.g., 'B26')

    Examples:
        >>> indices_to_cell(0, 0)
        'A1'
        >>> indices_to_cell(25, 1)
        'B26'
    """
    # Convert column index to letters
    col_letters = ''
    col_num = col + 1  # Convert to 1-based

    while col_num > 0:
        col_num -= 1
        col_letters = chr(col_num % 26 + ord('A')) + col_letters
        col_num //= 26

    # Convert row index to number (1-based)
    row_num = row + 1

    return f"{col_letters}{row_num}"


def extract_cell_value(
        file_path: Union[str, Path],
        sheet_name: str,
        cell_ref: str,
        dtype: Optional[type] = None
) -> Optional[Any]:
    """
    Extract a value from a specific cell in an Excel sheet.

    Args:
        file_path: Path to Excel file
        sheet_name: Name of the sheet
        cell_ref: Cell reference (e.g., 'B26')
        dtype: Expected data type (float, int, str)

    Returns:
        Cell value or None if not found/invalid
    """
    try:
        # Convert cell reference to indices
        row_idx, col_idx = cell_to_indices(cell_ref)

        # Read the specific cell using pandas
        # Use header=None to prevent first row being used as header
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            nrows=row_idx + 1,
            usecols=range(col_idx + 1)
        )

        # Check if indices are within bounds
        if row_idx < df.shape[0] and col_idx < df.shape[1]:
            value = df.iloc[row_idx, col_idx]

            # Handle pandas NA
            if pd.isna(value):
                return None

            # Convert to requested type if specified
            if dtype is not None:
                try:
                    if dtype == float:
                        # Handle string values that might contain units
                        if isinstance(value, str):
                            # Extract numeric part
                            numeric_match = re.search(r'[-+]?\d*\.?\d+', value)
                            if numeric_match:
                                value = float(numeric_match.group())
                            else:
                                return None
                        else:
                            value = float(value)
                    elif dtype == int:
                        value = int(float(value))  # Handle decimal inputs
                    elif dtype == str:
                        value = str(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert cell {cell_ref} value '{value}' to {dtype.__name__}"
                    )
                    return None

            return value
        else:
            logger.debug(f"Cell {cell_ref} is out of bounds in sheet {sheet_name}")
            return None

    except Exception as e:
        logger.error(f"Error extracting cell {cell_ref} from {sheet_name}: {e}")
        return None


def get_sheet_names(file_path: Union[str, Path]) -> List[str]:
    """
    Get all sheet names from an Excel file.

    Args:
        file_path: Path to Excel file

    Returns:
        List of sheet names
    """
    try:
        with pd.ExcelFile(file_path) as xls:
            return xls.sheet_names
    except Exception as e:
        logger.error(f"Error reading sheet names from {file_path}: {e}")
        return []


def read_data_range(
        file_path: Union[str, Path],
        sheet_name: str,
        start_cell: str,
        end_cell: str,
        header: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Read a specific range of cells from Excel.

    Args:
        file_path: Path to Excel file
        sheet_name: Name of the sheet
        start_cell: Top-left cell of range (e.g., 'A1')
        end_cell: Bottom-right cell of range (e.g., 'J100')
        header: Row number to use as header (None for no header)

    Returns:
        DataFrame with the data or None if error
    """
    try:
        # Convert cell references to indices
        start_row, start_col = cell_to_indices(start_cell)
        end_row, end_col = cell_to_indices(end_cell)

        # Read the range
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header,
            skiprows=start_row if header is None else start_row - 1,
            nrows=end_row - start_row + 1,
            usecols=range(start_col, end_col + 1)
        )

        return df

    except Exception as e:
        logger.error(f"Error reading range {start_cell}:{end_cell} from {sheet_name}: {e}")
        return None


def find_value_in_sheet(
        file_path: Union[str, Path],
        sheet_name: str,
        search_value: Any,
        search_area: Optional[Tuple[str, str]] = None
) -> Optional[str]:
    """
    Find a value in a sheet and return its cell reference.

    Args:
        file_path: Path to Excel file
        sheet_name: Name of the sheet
        search_value: Value to search for
        search_area: Optional tuple of (start_cell, end_cell) to limit search

    Returns:
        Cell reference where value was found, or None
    """
    try:
        if search_area:
            df = read_data_range(file_path, sheet_name, search_area[0], search_area[1])
            if df is None:
                return None
            start_row, start_col = cell_to_indices(search_area[0])
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            start_row, start_col = 0, 0

        # Search for the value
        for row_idx in range(df.shape[0]):
            for col_idx in range(df.shape[1]):
                if df.iloc[row_idx, col_idx] == search_value:
                    # Convert back to cell reference
                    actual_row = start_row + row_idx
                    actual_col = start_col + col_idx
                    return indices_to_cell(actual_row, actual_col)

        return None

    except Exception as e:
        logger.error(f"Error searching for value in {sheet_name}: {e}")
        return None


def safe_read_excel(
        file_path: Union[str, Path],
        sheet_name: Optional[Union[str, int]] = 0,
        **kwargs
) -> Optional[pd.DataFrame]:
    """
    Safely read an Excel file with error handling.

    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name or index
        **kwargs: Additional arguments passed to pd.read_excel

    Returns:
        DataFrame or None if error
    """
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    except FileNotFoundError:
        logger.error(f"Excel file not found: {file_path}")
    except ValueError as e:
        logger.error(f"Invalid sheet name '{sheet_name}': {e}")
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")

    return None


def get_column_letter(col_idx: int) -> str:
    """
    Convert column index to Excel column letter.

    Args:
        col_idx: 0-based column index

    Returns:
        Column letter (A, B, ..., AA, AB, ...)
    """
    letters = ''
    col_num = col_idx + 1

    while col_num > 0:
        col_num -= 1
        letters = chr(col_num % 26 + ord('A')) + letters
        col_num //= 26

    return letters


def get_column_index(col_letter: str) -> int:
    """
    Convert Excel column letter to index.

    Args:
        col_letter: Column letter (A, B, ..., AA, AB, ...)

    Returns:
        0-based column index
    """
    col_idx = 0
    for i, letter in enumerate(reversed(col_letter.upper())):
        col_idx += (ord(letter) - ord('A') + 1) * (26 ** i)
    return col_idx - 1