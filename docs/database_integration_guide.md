# Database Integration Guide

## Overview

The database module provides comprehensive data storage and analysis capabilities for the Laser Trim AI System. It enables historical tracking, trend analysis, and continuous improvement through machine learning insights.

## Key Features

1. **Automatic Data Storage**: All analysis results are automatically saved
2. **Historical Analysis**: Track performance trends over time
3. **Anomaly Detection**: Identify and cluster unusual patterns
4. **Cost Impact Analysis**: Calculate financial implications of quality issues
5. **ML Performance Tracking**: Monitor and improve model accuracy
6. **Comprehensive Reporting**: Generate detailed trend reports with visualizations

## Integration Steps

### 1. Update Data Processor

```python
# In data_processor.py, add database integration:

from database import DatabaseManager

class LaserTrimDataProcessor:
    def __init__(self, config_file='config.json'):
        # Existing initialization...
        self.db_manager = DatabaseManager(self.config)
        
    def analyze_file(self, file_path):
        # Existing analysis...
        
        # Save to database
        if hasattr(self, 'db_manager'):
            run_id = getattr(self, 'current_run_id', None)
            if run_id:
                self.db_manager.save_file_result(run_id, result)
        
        return result
    
    def process_folder(self, folder_path):
        # Create analysis run
        if hasattr(self, 'db_manager'):
            self.current_run_id = self.db_manager.create_analysis_run(
                folder_path, 
                self.config.__dict__
            )
        
        # Existing processing...
        
        # Update run statistics
        if hasattr(self, 'db_manager') and hasattr(self, 'current_run_id'):
            self.db_manager.update_analysis_run(
                self.current_run_id,
                processed_files=len(results),
                failed_files=len(errors),
                total_files=total_files,
                processing_time=time.time() - start_time
            )
```

### 2. Update GUI Application

```python
# In gui_application.py, add database features:

from database import DatabaseManager, HistoricalAnalyzer, TrendReporter

class LaserTrimAnalyzerGUI:
    def __init__(self):
        # Existing initialization...
        self.init_database()
        
    def init_database(self):
        """Initialize database components."""
        try:
            self.db_manager = DatabaseManager(self.config)
            self.analyzer = HistoricalAnalyzer(self.db_manager, self.config)
            self.reporter = TrendReporter(self.db_manager, self.analyzer, self.config)
            self.add_database_menu()
        except Exception as e:
            print(f"Database initialization failed: {e}")
            
    def add_database_menu(self):
        """Add database menu to GUI."""
        db_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Database", menu=db_menu)
        
        db_menu.add_command(label="View Historical Data", command=self.view_historical_data)
        db_menu.add_command(label="Generate Trend Report", command=self.generate_trend_report)
        db_menu.add_command(label="Model Analysis", command=self.analyze_models)
        db_menu.add_separator()
        db_menu.add_command(label="Import Data", command=self.import_data)
        db_menu.add_command(label="Export Backup", command=self.export_backup)
```

### 3. Update ML Models

```python
# In ml_models.py, add continuous learning:

class ContinuousLearningMixin:
    """Mixin for continuous learning from historical data."""
    
    def update_from_history(self, days_back=90):
        """Update model with recent historical data."""
        if not hasattr(self, 'db_manager'):
            return
            
        # Get recent data
        df = self.db_manager.get_historical_data(days_back=days_back)
        
        if len(df) > 100:  # Minimum data required
            # Prepare features and labels
            X, y = self.prepare_training_data(df)
            
            # Incremental learning
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X, y)
            else:
                # Retrain completely
                self.train(X, y)
                
            # Save updated model
            self.save_model()
            
            print(f"Model updated with {len(df)} historical records")
```

## Usage Examples

### 1. Query Historical Data

```python
from database import DatabaseManager

# Initialize
db = DatabaseManager(config)

# Get recent data for a model
df = db.get_historical_data(model='8340-1', days_back=30)

# Get model performance history
perf = db.get_model_performance_history('8340-1', days_back=90)
```

### 2. Generate Trend Reports

```python
from database import HistoricalAnalyzer, TrendReporter

# Initialize
analyzer = HistoricalAnalyzer(db, config)
reporter = TrendReporter(db, analyzer, config)

# Generate comprehensive report
report_path = reporter.generate_comprehensive_report(
    output_dir=Path('./reports'),
    days_back=30
)

# Quick model report
summary = reporter.generate_quick_report(
    model='8340-1',
    output_path=Path('./quick_report.txt')
)
```

### 3. Analyze Trends

```python
# Analyze model trends
trends = analyzer.analyze_model_trends('8340-1', days_back=90)

# Detect anomaly clusters
clusters = analyzer.detect_anomaly_clusters(days_back=30)

# Get improvement recommendations
recommendations = analyzer.generate_improvement_recommendations('8340-1')

# Calculate cost impact
cost_analysis = analyzer.calculate_cost_impact('8340-1', days_back=30)
```

### 4. Import Legacy Data

```python
from database import DataMigrator

# Initialize
migrator = DataMigrator(db, config)

# Import from Excel
result = migrator.import_from_excel(Path('./legacy_data.xlsx'))

# Import from directory
stats = migrator.import_legacy_data(Path('./legacy_folder'))

# Export backup
backup_path = migrator.export_for_backup(Path('./backups'))
```

## Database Schema

The database uses SQLite with the following main tables:

1. **analysis_runs**: Tracks analysis sessions
2. **file_results**: Individual file analysis results
3. **model_performance**: Aggregated model statistics
4. **anomalies**: Detected anomalies
5. **ml_predictions**: ML model predictions and accuracy

## Configuration

Add these settings to your config.json:

```json
{
    "DATABASE": {
        "enabled": true,
        "cleanup_days": 365,
        "backup_interval_days": 7
    },
    "ANOMALY_THRESHOLDS": {
        "high_sigma": 0.05,
        "high_failure_prob": 0.7,
        "resistance_change": 20
    }
}
```

## Best Practices

1. **Regular Backups**: Schedule automatic backups using the DataMigrator
2. **Data Cleanup**: Periodically clean old data to prevent database bloat
3. **Performance Monitoring**: Use the ML accuracy tracking to monitor model performance
4. **Trend Analysis**: Generate monthly trend reports for continuous improvement
5. **Anomaly Investigation**: Regularly review anomaly clusters for systematic issues

## Troubleshooting

### Database Locked Error
- Ensure only one process accesses the database at a time
- Use the context manager for all database operations

### Import Failures
- Validate data before import using `DataMigrator.validate_import()`
- Check column mappings match your Excel format

### Performance Issues
- Create indexes on frequently queried fields
- Use `cleanup_old_data()` to remove old records
- Consider upgrading to PostgreSQL for large datasets

## Future Enhancements

1. **Real-time Dashboard**: Web-based dashboard for live monitoring
2. **Predictive Maintenance**: Use trends to predict equipment issues
3. **Advanced Analytics**: Integration with business intelligence tools
4. **Cloud Storage**: Optional cloud database support
5. **API Access**: RESTful API for external integrations