"""
Date and time utilities for Laser Trim Analyzer.

Provides comprehensive filename-based date extraction and parsing utilities.
"""

import re
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date and time from filename with comprehensive pattern matching.
    
    This function handles various filename patterns commonly found in laser trim data files,
    such as "7844_258B_TEST DATA_9-13-2024_6-48 AM.xls".
    
    Args:
        filename: Filename to parse (with or without extension)
        
    Returns:
        datetime object if parsing successful, None otherwise
    """
    if not filename:
        return None
    
    # Enhanced patterns to capture both date and time from filenames
    # Based on patterns like "7844_258B_TEST DATA_9-13-2024_6-48 AM.xls"
    patterns = [
        # Pattern for files like "7844_258B_TEST DATA_9-13-2024_6-48 AM.xls"
        r'(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2})-(\d{1,2})\s*(AM|PM)',
        # Pattern for files like "7844_258B_TEST DATA_9-13-2024_6:48 AM.xls"  
        r'(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2}):(\d{1,2})\s*(AM|PM)',
        # Pattern for files like "7844_258B_TEST DATA_9-13-2024_6-48AM.xls"
        r'(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2})-(\d{1,2})(AM|PM)',
        # Pattern for files like "7844_258B_TEST DATA_9-13-2024_6:48AM.xls"
        r'(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2}):(\d{1,2})(AM|PM)',
        # Pattern for 24-hour format like "9-13-2024_18-48"
        r'(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2})-(\d{1,2})(?![AP])',
        # Pattern for 24-hour format like "9-13-2024_18:48"
        r'(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2}):(\d{1,2})(?![AP])',
        # Alternative separators with forward slashes
        r'(\d{1,2})/(\d{1,2})/(\d{4})_(\d{1,2})-(\d{1,2})\s*(AM|PM)',
        r'(\d{1,2})/(\d{1,2})/(\d{4})_(\d{1,2}):(\d{1,2})\s*(AM|PM)',
        r'(\d{1,2})/(\d{1,2})/(\d{4})_(\d{1,2})-(\d{1,2})(AM|PM)',
        r'(\d{1,2})/(\d{1,2})/(\d{4})_(\d{1,2}):(\d{1,2})(AM|PM)',
        # Date only patterns as fallback
        r'(\d{1,2})-(\d{1,2})-(\d{4})',
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
        r'(\d{1,2})_(\d{1,2})_(\d{4})',
        # ISO format
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        # Compact format YYYYMMDD
        r'(\d{8})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, filename)
        if matches:
            match = matches[0]
            try:
                if len(match) == 6:  # Date + time with AM/PM
                    month, day, year, hour, minute, ampm = match
                    month, day, year = int(month), int(day), int(year)
                    hour, minute = int(hour), int(minute)
                    
                    # Convert to 24-hour format
                    if ampm.upper() == 'PM' and hour != 12:
                        hour += 12
                    elif ampm.upper() == 'AM' and hour == 12:
                        hour = 0
                    
                    # Validate date/time components
                    if (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100 and
                        0 <= hour <= 23 and 0 <= minute <= 59):
                        extracted_datetime = datetime(year, month, day, hour, minute)
                        logger.debug(f"Extracted datetime {extracted_datetime} from filename '{filename}' using pattern with AM/PM")
                        return extracted_datetime
                        
                elif len(match) == 5:  # Date + time without AM/PM (24-hour)
                    month, day, year, hour, minute = match
                    month, day, year = int(month), int(day), int(year)
                    hour, minute = int(hour), int(minute)
                    
                    # Validate date/time components
                    if (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100 and
                        0 <= hour <= 23 and 0 <= minute <= 59):
                        extracted_datetime = datetime(year, month, day, hour, minute)
                        logger.debug(f"Extracted datetime {extracted_datetime} from filename '{filename}' using 24-hour pattern")
                        return extracted_datetime
                        
                elif len(match) == 3:  # Date only
                    part1, part2, part3 = match
                    
                    # Determine date format based on values
                    if len(part1) == 4:  # YYYY-MM-DD format
                        year, month, day = int(part1), int(part2), int(part3)
                    elif len(part3) == 4:  # MM-DD-YYYY or DD-MM-YYYY format
                        # Assume MM-DD-YYYY for US format (most common in the data)
                        month, day, year = int(part1), int(part2), int(part3)
                    else:
                        continue  # Skip invalid formats
                    
                    # Validate date components
                    if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                        extracted_date = datetime(year, month, day)
                        logger.debug(f"Extracted date {extracted_date} from filename '{filename}' using date-only pattern")
                        return extracted_date
                        
                elif len(match) == 1:  # YYYYMMDD format
                    date_str = match
                    if len(date_str) == 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        
                        # Validate date components
                        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                            extracted_date = datetime(year, month, day)
                            logger.debug(f"Extracted date {extracted_date} from filename '{filename}' using YYYYMMDD pattern")
                            return extracted_date
                            
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse date/time from match {match} in filename '{filename}': {e}")
                continue
    
    return None


def parse_datetime_string(date_str: str) -> Optional[datetime]:
    """
    Parse datetime string using common formats.
    
    Args:
        date_str: String representation of date/time
        
    Returns:
        datetime object if parsing successful, None otherwise
    """
    if not date_str:
        return None
    
    formats_to_try = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y',
        '%m-%d-%Y %H:%M:%S',
        '%m-%d-%Y',
        '%Y%m%d',
        '%m-%d-%Y %I:%M %p',
        '%m/%d/%Y %I:%M %p'
    ]
    
    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse datetime string: {date_str}")
    return None


def safe_datetime_convert(value) -> Optional[datetime]:
    """
    Safely convert various types to datetime object.
    
    Args:
        value: Value to convert (str, datetime, or None)
        
    Returns:
        datetime object if conversion successful, None otherwise
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, str):
        return parse_datetime_string(value)
    
    # Try to convert other types to string first
    try:
        return parse_datetime_string(str(value))
    except Exception:
        return None 