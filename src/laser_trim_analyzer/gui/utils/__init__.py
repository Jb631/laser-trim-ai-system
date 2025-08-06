"""GUI Utilities for Laser Trim Analyzer"""

def apply_saved_settings_to_config(config, settings_manager):
    """Apply saved settings from settings_manager to the config object."""
    
    # Processing settings
    if hasattr(config, 'processing'):
        workers = settings_manager.get('performance.thread_pool_size')
        if workers is not None:
            config.processing.max_workers = workers
            
        generate_plots = settings_manager.get('display.include_charts')
        if generate_plots is not None:
            config.processing.generate_plots = generate_plots
            
        cache_enabled = settings_manager.get('performance.enable_caching')
        if cache_enabled is not None:
            config.processing.cache_enabled = cache_enabled
    
    # Database settings
    if hasattr(config, 'database'):
        db_enabled = settings_manager.get('data.auto_backup')
        if db_enabled is not None:
            config.database.enabled = db_enabled
    
    # ML settings
    if hasattr(config, 'ml'):
        ml_enabled = settings_manager.get('analysis.enable_ml_predictions')
        if ml_enabled is not None:
            config.ml.enabled = ml_enabled
            
        ml_insights = settings_manager.get('notifications.notification_types.ml_insights')
        if ml_insights is not None and hasattr(config.ml, 'failure_prediction_enabled'):
            config.ml.failure_prediction_enabled = ml_insights
            
        threshold_opt = settings_manager.get('analysis.threshold_optimization')
        if threshold_opt is not None and hasattr(config.ml, 'threshold_optimization_enabled'):
            config.ml.threshold_optimization_enabled = threshold_opt
    
    # GUI settings
    if hasattr(config, 'gui'):
        theme = settings_manager.get('theme.mode')
        if theme is not None and hasattr(config.gui, 'theme'):
            config.gui.theme = theme
            
        experimental = settings_manager.get('advanced.experimental_features')
        if experimental is not None:
            if hasattr(config.gui, 'show_historical_tab'):
                config.gui.show_historical_tab = experimental
            if hasattr(config.gui, 'show_ml_insights'):
                config.gui.show_ml_insights = experimental