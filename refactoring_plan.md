# Laser Trim Analyzer - Complete Refactoring Plan

## Overview
Transform the current application from a collection of scripts into a professional, integrated QA platform with AI capabilities.

## Phase 1: Foundation and Architecture (Week 1)

### Step 1.1: Create New Project Structure
```
laser_trim_qa_pro/
├── src/
│   ├── __init__.py
│   ├── main.py                    # New entry point
│   ├── core/                      # Core business logic
│   │   ├── __init__.py
│   │   ├── processor.py           # Refactored processor
│   │   ├── analyzer.py            # Analysis engine
│   │   └── models.py              # Data models
│   ├── gui/                       # GUI components
│   │   ├── __init__.py
│   │   ├── main_window.py         # Main QA Professional GUI
│   │   ├── widgets/               # Custom widgets
│   │   └── dialogs/               # Dialog windows
│   ├── database/                  # Database layer
│   │   ├── __init__.py
│   │   ├── manager.py             # Unified DB manager
│   │   └── models.py              # SQLAlchemy models
│   ├── ml/                        # Machine learning
│   │   ├── __init__.py
│   │   ├── optimizer.py           # Threshold optimizer
│   │   └── predictor.py           # Failure predictor
│   ├── ai/                        # AI integration
│   │   ├── __init__.py
│   │   ├── client.py              # AI API client
│   │   └── prompts.py             # Prompt templates
│   └── utils/                     # Utilities
│       ├── __init__.py
│       └── config.py              # Configuration
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── config/                        # Configuration files
└── requirements.txt               # Dependencies
```
laser_trim_analyzer_v2/
├── src/
│   └── laser_trim_analyzer/
│       ├── __init__.py
│       ├── __main__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py          # Pydantic data models
│       │   ├── config.py          # Configuration management
│       │   └── constants.py       # App constants
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── base.py            # Base analyzer class
│       │   ├── sigma_analyzer.py
│       │   ├── linearity_analyzer.py
│       │   └── resistance_analyzer.py
│       ├── database/
│       │   ├── __init__.py
│       │   ├── models.py          # SQLAlchemy models
│       │   ├── manager.py         # Unified DB manager
│       │   └── migrations/
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── models.py          # ML model classes
│       │   └── predictors.py
│       ├── gui/
│       │   ├── __init__.py
│       │   ├── main_window.py
│       │   ├── widgets/
│       │   └── dialogs/
│       ├── api/
│       │   ├── __init__.py
│       │   ├── client.py          # API client for AI services
│       │   └── schemas.py         # API request/response models
│       └── utils/
│           ├── __init__.py
│           ├── file_utils.py
│           └── validators.py
├── tests/
├── docs/
├── config/
│   ├── default.yaml
│   └── production.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
### Step 1.2: Create Data Models

**Prompt for New Chat:**
```
I'm refactoring a laser trim analyzer app for potentiometer QA. I need to create clean data models using dataclasses and/or Pydantic. Here are my current data structures:

1. Analysis Result (per file/track):
- File info: filename, path, model, serial, system, date
- Track info: track_id, status
- Metrics: sigma_gradient, sigma_threshold, sigma_pass, linearity_spec, linearity_pass
- Properties: unit_length, untrimmed_resistance, trimmed_resistance
- Advanced: failure_probability, risk_category, trim_improvement

2. Processing Configuration:
- Input/output paths
- Processing options (plots, parallel, ML, AI)
- API keys and credentials

3. Historical Query:
- Model filter, serial filter, date range
- Aggregation options

Please create clean, type-hinted data models with validation.
```

### Step 1.3: Unified Database Layer

**Prompt for New Chat:**
```
I need to refactor my database layer for a laser trim analyzer. Currently I have multiple database managers with overlapping functionality. I need:

1. Single DatabaseManager class using SQLAlchemy
2. Proper models for:
   - analysis_results (file-level data)
   - track_results (track-level data) 
   - ml_predictions
   - qa_alerts
   - batch_info
3. Efficient methods for:
   - Saving analysis results
   - Querying historical data
   - Aggregating statistics
   - Managing relationships between tables
4. Migration system for schema updates

Current issues: SQLite with raw SQL, no ORM, duplicate save logic
```

## Phase 2: Core Processing Engine (Week 2)

### Step 2.1: Refactor Processing Logic

**Prompt for New Chat:**
```
I'm refactoring the core processing engine for a potentiometer laser trim analyzer. Current issues:
- processor_module.py is 2000+ lines with mixed concerns
- Plotting mixed with processing logic
- Poor error handling
- No clean interfaces

I need to:
1. Extract a clean Processor interface
2. Separate concerns:
   - FileReader (handles Excel reading)
   - DataExtractor (extracts data from sheets)
   - MetricsCalculator (calculates all metrics)
   - ResultsFormatter (formats output)
3. Implement strategy pattern for different systems (A/B)
4. Add proper async/await for file processing
5. Create clean result objects instead of nested dicts

Here's the current process flow: [paste key parts of process_file method]
```

### Step 2.2: Extract Analytics Engine

**Prompt for New Chat:**
```
I need to extract analytics into a clean, testable module. Current analytics are scattered across multiple files:
- linearity_analyzer.py
- failure_analyzer.py  
- resistance_analyzer.py
- zone_analyzer.py
- trim_analyzer.py
- dynamic_range_analyzer.py

Create a unified AnalyticsEngine that:
1. Has a clean interface for all analysis types
2. Returns structured results (not dicts)
3. Can be configured for different analysis profiles
4. Supports plugins for custom analytics
5. Has proper error handling and logging

Example current code: [paste one analyzer]
```

## Phase 3: Machine Learning Integration (Week 3)

### Step 3.1: ML Module Refactoring

**Prompt for New Chat:**
```
I have ML code for potentiometer QA that needs refactoring:
- ml_threshold_optimizer.py (determines optimal thresholds)
- Predictive failure analysis
- Drift detection

Refactor into a clean ML module with:
1. MLEngine class that manages all ML operations
2. Separate model classes for each ML task
3. Model versioning and storage
4. Automated retraining pipeline
5. Feature engineering pipeline
6. Model performance tracking

Current issues: ML code mixed with business logic, no model management, hardcoded parameters
```

### Step 3.2: Real-time ML Integration

**Prompt for New Chat:**
```
I need to integrate ML predictions in real-time during analysis:

1. As each file is processed, get ML predictions
2. Flag anomalies immediately
3. Suggest threshold adjustments
4. Predict failure probability
5. Compare to historical patterns

Create an MLPredictor class that:
- Loads pre-trained models on startup
- Provides fast inference
- Caches predictions
- Updates models periodically
- Handles missing features gracefully
```

## Phase 4: GUI Modernization (Week 4)

### Step 4.1: Main Window Implementation

**Prompt for New Chat:**
```
I need to implement a modern GUI for a QA analysis tool using tkinter. Requirements:

1. Main window with:
   - Sidebar navigation (Home, Analysis, Historical, ML Tools, AI Insights, Settings)
   - Header with quick actions and status
   - Content area with stacked frames
   - Status bar with connection indicators

2. Modern styling:
   - Card-based layouts
   - Consistent color scheme
   - Responsive design
   - Loading animations

3. Key features:
   - Drag-and-drop file selection
   - Real-time progress updates
   - Interactive charts (matplotlib)
   - Export functionality

Base it on this design: [paste qa_professional_gui.py overview]
```

### Step 4.2: Custom Widgets

**Prompt for New Chat:**
```
Create custom tkinter widgets for a QA application:

1. MetricCard widget:
   - Shows title, value, trend arrow, sparkline
   - Configurable colors based on thresholds
   - Click for details

2. FileAnalysisWidget:
   - Shows file info, progress bar, status
   - Expandable to show track details
   - Action buttons (view plot, export)

3. ChartWidget:
   - Wrapper for matplotlib charts
   - Zoom, pan, export functionality
   - Multiple chart types (line, bar, scatter)

4. AlertBanner:
   - Shows critical alerts
   - Dismissible
   - Action buttons

Style: Modern, clean, professional
```

## Phase 5: AI Integration (Week 5)

### Step 5.1: AI Client Implementation

**Prompt for New Chat:**
```
Implement an AI client for QA analysis insights. Support multiple providers:

1. Anthropic Claude API
2. OpenAI GPT-4 API
3. Local LLM (Ollama)

Requirements:
- Unified interface regardless of provider
- Retry logic and error handling
- Token counting and cost tracking
- Response caching
- Streaming responses
- Prompt templates for different analysis types

Example use cases:
- Analyze failure patterns
- Suggest process improvements
- Generate QA reports
- Answer questions about data
```

### Step 5.2: Intelligent Analysis Features

**Prompt for New Chat:**
```
Create AI-powered analysis features for potentiometer QA:

1. Automatic insight generation:
   - After each analysis run, generate insights
   - Identify unusual patterns
   - Compare to best practices

2. Interactive QA assistant:
   - Answer questions about the data
   - Explain complex metrics
   - Suggest next steps

3. Report generation:
   - Create professional PDF reports
   - Include AI-generated summaries
   - Actionable recommendations

4. Predictive maintenance:
   - Predict when equipment needs calibration
   - Identify drift patterns
   - Alert on anomalies
```

## Phase 6: Integration and Testing (Week 6)

### Step 6.1: System Integration

**Prompt for New Chat:**
```
I need to integrate all components of my refactored QA system:

1. Data flow:
   - GUI triggers analysis
   - Processor handles files
   - Analytics engine computes metrics
   - ML makes predictions
   - Results saved to database
   - AI generates insights
   - GUI displays results

2. Event system:
   - Progress updates
   - Real-time alerts
   - Status changes

3. Configuration management:
   - User preferences
   - API keys
   - Processing options
   - ML model paths

Create integration code that ties everything together cleanly.
```

### Step 6.2: Testing Strategy

**Prompt for New Chat:**
```
Create a comprehensive testing strategy for a QA analysis system:

1. Unit tests for:
   - Data models
   - Processing logic
   - Analytics calculations
   - ML predictions

2. Integration tests for:
   - File processing pipeline
   - Database operations
   - API communications

3. GUI tests:
   - User workflows
   - Error handling
   - Performance

4. Test data:
   - Sample files for each system type
   - Edge cases
   - Performance benchmarks

Provide pytest fixtures and test examples.
```

## Phase 7: Deployment and Documentation (Week 7)

### Step 7.1: Packaging and Distribution

**Prompt for New Chat:**
```
Package a Python QA application for distribution:

1. Create installer for Windows using:
   - PyInstaller or cx_Freeze
   - Include all dependencies
   - Handle data files and configs

2. Configuration:
   - First-run setup wizard
   - Database initialization
   - API key configuration

3. Updates:
   - Auto-update mechanism
   - Version checking
   - Rollback capability

4. Licensing:
   - License key validation
   - Feature flags
   - Usage tracking
```

### Step 7.2: Documentation

**Prompt for New Chat:**
```
Create comprehensive documentation for a QA analysis system:

1. User Guide:
   - Getting started
   - Feature walkthrough
   - Troubleshooting
   - FAQ

2. Administrator Guide:
   - Installation
   - Configuration
   - Database management
   - ML model updates

3. Developer Guide:
   - Architecture overview
   - API reference
   - Plugin development
   - Contributing guidelines

4. Video tutorials:
   - Script for intro video
   - Feature demonstrations
   - Best practices

Format: MkDocs with Material theme
```

## Migration Strategy

### Step 1: Parallel Development
- Keep existing system running
- Build new system alongside
- Test with subset of data

### Step 2: Data Migration
```python
# Migration script prompt
"""
Create a data migration script that:
1. Reads from old SQLite database
2. Transforms data to new schema
3. Validates data integrity
4. Imports to new database
5. Verifies completeness
"""
```

### Step 3: Gradual Rollout
- Beta test with power users
- Run both systems in parallel
- Compare results
- Full switchover

## Quick Reference Prompts

### For Architecture Questions:
```
I'm building a QA analysis system with Python. Current stack: tkinter GUI, SQLite database, pandas/numpy for analysis, matplotlib for plots. I need advice on [specific architecture question].
```

### For Implementation Help:
```
I'm implementing [specific feature] for a potentiometer QA system. Current code: [paste relevant code]. Issues: [describe problems]. Need: [desired outcome].
```

### For Debugging:
```
In my laser trim analyzer, [describe issue]. Error: [paste error]. Relevant code: [paste code]. Context: [explain what should happen].
```

### For Optimization:
```
My QA analysis app has performance issues with [describe scenario]. Current approach: [paste code]. Data size: [typical volumes]. Need to optimize for [specific goals].
```

## Success Metrics

1. **Performance**: Process 1000 files in < 5 minutes
2. **Reliability**: 99.9% uptime, < 0.1% analysis errors  
3. **Usability**: New users productive in < 1 hour
4. **Insights**: AI generates actionable insights for 95% of issues
5. **Maintenance**: Add new features in < 1 day

## Next Chat Prompt

When starting a new chat for the next phase:

```
I'm refactoring a laser trim analyzer app following this plan: [paste relevant phase from above]. 

Current status: [what's completed]
Working on: [current phase]
Specific need: [what you need help with]

Here's my current code: [paste relevant code]
Here's what I'm trying to achieve: [specific goals]
```

This plan gives you a clear roadmap and specific prompts for each phase. Save this document and refer to it as you progress through the refactoring.
