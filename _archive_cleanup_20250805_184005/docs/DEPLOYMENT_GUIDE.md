# Laser Trim Analyzer - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Laser Trim Analyzer in both single-user and multi-user environments. The application supports flexible deployment modes that can be switched through the settings interface.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Setup](#pre-installation-setup)
3. [Installation Options](#installation-options)
4. [Network Database Configuration](#network-database-configuration)
5. [Multi-User Setup](#multi-user-setup)
6. [Security Considerations](#security-considerations)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **RAM**: 8 GB (16 GB recommended for large datasets)
- **Storage**: 2 GB for application + space for database
- **Display**: 1920x1080 resolution or higher
- **Network**: Gigabit Ethernet for shared database access

### Software Dependencies
- Windows Visual C++ Redistributable 2015-2022
- .NET Framework 4.7.2 or later (for some Windows features)
- Network access to shared database location

## Pre-Installation Setup

### 1. Network Share Preparation

Create a network share for the shared database:

```bash
# Example network path
\\server\SharedData\LaserTrimAnalyzer\
```

Required folder structure:
```
\\server\SharedData\LaserTrimAnalyzer\
├── database\           # Shared database location
├── logs\              # Centralized logs (optional)
├── models\            # Shared ML models
└── config\            # Shared configuration
```

### 2. Permissions Setup

Set the following NTFS permissions on the shared folder:

- **Database folder**: Read/Write for all users
- **Models folder**: Read for users, Write for administrators
- **Config folder**: Read for users, Write for administrators
- **Logs folder**: Write for all users

### 3. Database Initialization

Before first use, initialize the shared database:

```bash
# Run on the server or admin workstation
LaserTrimAnalyzer.exe --init-db --db-path "\\server\SharedData\LaserTrimAnalyzer\database\laser_trim_production.db"
```

## Installation Options

### Option 1: Interactive Installation (Recommended)

1. Run `LaserTrimAnalyzer_Setup_2.0.0.exe`
2. Follow the installation wizard
3. **Choose deployment mode**:
   - **Single User Mode**: Local database for individual use
   - **Multi-User Mode**: Network database for team collaboration
4. If Multi-User Mode selected, enter the network database path

### Option 2: Silent Installation (For IT Departments)

For automated deployment via Group Policy or SCCM:

```bash
# Single-user installation
LaserTrimAnalyzer_Setup_2.0.0.exe /SILENT /MODE=single

# Multi-user installation
LaserTrimAnalyzer_Setup_2.0.0.exe /SILENT /MODE=multi /DBPATH="\\server\share\LaserTrimAnalyzer"
```

Parameters:
- `/SILENT` - No user interaction
- `/VERYSILENT` - No progress window
- `/MODE=single|multi` - Deployment mode
- `/DBPATH=path` - Network database path (multi-user only)
- `/DIR="C:\CustomPath"` - Custom installation directory
- `/GROUP="Custom Start Menu"` - Custom program group
- `/NOICONS` - Don't create shortcuts
- `/LOG="install.log"` - Installation log file

### Option 3: Manual Deployment

For testing or custom deployments:

1. Extract the portable version
2. Copy to desired location
3. Configure `config/deployment.yaml`
4. Create shortcuts manually

## Deployment Mode Configuration

### Switching Between Modes

Users can switch between single-user and multi-user modes through the application settings:

1. Open the application
2. Navigate to **Settings** page
3. In **Database Settings** section, select deployment mode:
   - **Single User (Local)**: Uses local database
   - **Multi-User (Network)**: Uses shared network database
4. If switching to Multi-User mode, enter the network path
5. **Restart the application** for changes to take effect

### Configuration File Structure

The deployment configuration is stored in `config/deployment.yaml`:

```yaml
# Deployment mode selection
deployment_mode: "single_user"  # or "multi_user"

# Database configuration
database:
  # Single-user mode settings
  single_user:
    path: "%LOCALAPPDATA%/LaserTrimAnalyzer/database/laser_trim_local.db"
    connection_timeout: 10
    enable_wal_mode: false
    
  # Multi-user mode settings
  multi_user:
    path: "//server/share/LaserTrimAnalyzer/database.db"
    connection_timeout: 30
    sqlite_timeout: 30
    enable_wal_mode: true

# Path settings based on mode
paths:
  single_user:
    data_directory: "%LOCALAPPDATA%/LaserTrimAnalyzer/data"
    reports_directory: "%USERPROFILE%/Documents/LaserTrimAnalyzer/reports"
    models_directory: "%LOCALAPPDATA%/LaserTrimAnalyzer/models"
    
  multi_user:
    data_directory: "//server/share/LaserTrimAnalyzer/shared_data"
    reports_directory: "//server/share/LaserTrimAnalyzer/reports"
    models_directory: "//server/share/LaserTrimAnalyzer/ml_models"
```

### 2. Test Database Connection

After configuration, test the connection:

```bash
LaserTrimAnalyzer.exe --test-connection
```

Expected output:
```
Testing database connection...
Connection to \\server\SharedData\LaserTrimAnalyzer\database\laser_trim_production.db successful
Database version: 2.0.0
Write test: OK
Multi-user mode: Enabled
```

## Multi-User Setup

### 1. User Permissions

Configure Windows permissions for different user roles:

#### QA Operators
- Read/Write access to database
- Read access to models
- Execute application

#### QA Supervisors
- All operator permissions
- Write access to models
- Access to admin features

#### IT Administrators
- Full control over all directories
- Database maintenance capabilities

### 2. Concurrent Access Configuration

The application uses SQLite with WAL (Write-Ahead Logging) mode for better concurrent access:

```yaml
# In deployment.yaml
database:
  connection:
    journal_mode: "WAL"
    # WAL mode checkpoint settings
    wal_autocheckpoint: 1000  # pages
    wal_checkpoint_mode: "PASSIVE"
```

### 3. Performance Optimization

For optimal multi-user performance:

```yaml
performance:
  # Cache settings
  cache:
    enabled: true
    size: 100  # MB
    location: "%LOCALAPPDATA%\\LaserTrimAnalyzer\\cache"
  
  # Batch processing
  batch:
    chunk_size: 1000
    parallel_workers: 4
```

## Security Considerations

### 1. Data Security

- Database encryption: Not enabled by default
- For sensitive data, consider:
  - Using SQL Server instead of SQLite
  - Implementing folder-level encryption
  - Regular backups with encryption

### 2. Access Control

- Application respects Windows NTFS permissions
- No built-in user authentication (relies on Windows)
- Activity logging for audit trails

### 3. Network Security

- Use secured network shares (SMB 3.0+)
- Disable SMB 1.0 on all systems
- Consider VPN for remote access

## Troubleshooting

### Common Issues

#### 1. Database Locked Error
**Symptom**: "database is locked" error message

**Solutions**:
- Ensure WAL mode is enabled
- Check network connectivity
- Verify file permissions
- Increase timeout values

```yaml
database:
  connection:
    timeout: 60  # Increase timeout
    busy_timeout: 10000  # Increase busy timeout
```

#### 2. Slow Performance
**Symptom**: Application runs slowly with multiple users

**Solutions**:
- Move database to faster storage (SSD)
- Ensure gigabit network connection
- Enable local caching
- Check for network congestion

#### 3. Permission Denied
**Symptom**: Cannot write to database or save files

**Solutions**:
- Verify NTFS permissions
- Check share permissions
- Run as administrator (testing only)
- Review Group Policy settings

### Diagnostic Commands

```bash
# Check database integrity
LaserTrimAnalyzer.exe --check-db

# View connection statistics
LaserTrimAnalyzer.exe --connection-stats

# Export diagnostic information
LaserTrimAnalyzer.exe --diagnostic-export diagnostic_info.zip
```

### Log Files

Check logs in the following locations:

1. **Application logs**: `%LOCALAPPDATA%\LaserTrimAnalyzer\logs\`
2. **Shared logs**: `\\server\SharedData\LaserTrimAnalyzer\logs\`
3. **Windows Event Log**: Application section

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor disk space for database growth
- Check error logs for issues

#### Weekly
- Verify backup completion
- Review performance metrics
- Check for unusual database growth

#### Monthly
- Database optimization (VACUUM)
- Clear old log files
- Update ML models if needed

### Database Maintenance

#### 1. Backup Procedures

Automated backup script (run as scheduled task):

```batch
@echo off
set SOURCE=\\server\SharedData\LaserTrimAnalyzer\database\laser_trim_production.db
set BACKUP_DIR=\\server\Backups\LaserTrimAnalyzer\
set DATE=%date:~-4,4%%date:~-10,2%%date:~-7,2%

echo Backing up database...
copy "%SOURCE%" "%BACKUP_DIR%\laser_trim_production_%DATE%.db"
copy "%SOURCE%-wal" "%BACKUP_DIR%\laser_trim_production_%DATE%.db-wal"
copy "%SOURCE%-shm" "%BACKUP_DIR%\laser_trim_production_%DATE%.db-shm"

echo Backup completed: %BACKUP_DIR%\laser_trim_production_%DATE%.db
```

#### 2. Database Optimization

Run monthly during maintenance window:

```bash
# Optimize database
LaserTrimAnalyzer.exe --optimize-db

# Or use SQLite directly
sqlite3 "\\server\SharedData\LaserTrimAnalyzer\database\laser_trim_production.db" "VACUUM;"
```

#### 3. Archive Old Data

For databases over 1GB, consider archiving:

```bash
# Archive data older than 1 year
LaserTrimAnalyzer.exe --archive-data --older-than 365
```

### Update Procedures

1. **Test updates** in a non-production environment
2. **Backup database** before updating
3. **Notify users** of maintenance window
4. **Deploy update** using chosen method
5. **Verify functionality** before releasing to users

### Performance Monitoring

Monitor these metrics:

- Database size and growth rate
- Query response times
- Concurrent user count
- Network latency to database
- Error frequency in logs

## Appendices

### A. Command-Line Reference

```bash
LaserTrimAnalyzer.exe [options]

Options:
  --help                Show help message
  --version            Show version information
  --config FILE        Use custom configuration file
  --test-connection    Test database connection
  --check-db          Check database integrity
  --optimize-db       Optimize database (VACUUM)
  --init-db           Initialize new database
  --db-path PATH      Specify database path
  --diagnostic-export  Export diagnostic information
  --archive-data      Archive old data
  --older-than DAYS   Days threshold for archiving
```

### B. Configuration File Reference

See `config/deployment.yaml` for full configuration options.

### C. Support Information

For technical support:
- Internal IT Help Desk: ext. 1234
- Application Support: qa-tools-support@company.com
- Documentation: `\\server\Docs\LaserTrimAnalyzer\`

### D. Version History

- v2.0.0 - Initial enterprise deployment with multi-user support
- v1.x.x - Single-user desktop versions (deprecated)

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-19  
**Author**: QA Tools Team