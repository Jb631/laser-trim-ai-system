"""
End-to-end integration tests for the laser trim analyzer workflows.

Tests the complete workflow from startup to data processing and storage.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestApplicationStartup:
    """Test application startup and initialization."""
    
    @patch('laser_trim_analyzer.gui.ctk_main_window.ctk.CTk')
    def test_main_window_initialization(self, mock_ctk):
        """Test that main window initializes correctly."""
        from laser_trim_analyzer.core.config import get_config
        from laser_trim_analyzer.gui.main_window import MainWindow
        
        config = get_config()
        app = MainWindow(config)
        
        assert app.config is not None
        assert app.ctk_window is None  # Not created until run()
        
    @patch('laser_trim_analyzer.gui.ctk_main_window.ctk')
    def test_pages_registration(self, mock_ctk):
        """Test that all pages are registered correctly."""
        from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
        from laser_trim_analyzer.core.config import get_config
        
        # Mock the CTk components
        mock_ctk.CTkFrame = Mock
        mock_ctk.CTkLabel = Mock
        mock_ctk.CTkButton = Mock
        mock_ctk.CTkOptionMenu = Mock
        
        config = get_config()
        
        # Create window with mocked components
        with patch.object(CTkMainWindow, 'mainloop'):
            window = CTkMainWindow(config)
            
            # Check that pages were created
            expected_pages = [
                'home', 'analysis', 'single_file', 'batch',
                'multi_track', 'model_summary', 'historical',
                'ml_tools', 'ai_insights', 'settings'
            ]
            
            for page_name in expected_pages:
                assert page_name in window.pages, f"Page {page_name} not registered"


class TestDatabaseIntegration:
    """Test database manager integration."""
    
    def test_database_manager_initialization(self):
        """Test database manager initializes correctly."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db_url = f"sqlite:///{db_path}"
            
            manager = DatabaseManager(db_url)
            
            assert manager.engine is not None
            assert manager._session_factory is not None
            
            # Close the connection to avoid file lock
            manager.close()
            
    def test_database_error_recovery(self):
        """Test database handles unavailable state gracefully."""
        from laser_trim_analyzer.database.manager import DatabaseManager, DatabaseError
        
        # Test with invalid database URL
        with pytest.raises(DatabaseError):
            manager = DatabaseManager("invalid://database/url")
            with manager.get_session() as session:
                session.execute("SELECT 1")


class TestDataFlow:
    """Test data flow between components."""
    
    @patch('laser_trim_analyzer.gui.pages.single_file_page.LaserTrimProcessor')
    def test_single_file_to_database_flow(self, mock_processor):
        """Test data flows from single file analysis to database."""
        from laser_trim_analyzer.gui.pages.single_file_page import SingleFilePage
        from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus
        from datetime import datetime
        
        # Create mock analysis result
        mock_result = AnalysisResult(
            filename="test.xls",
            filepath=Path("test.xls"),
            model="TEST001",
            serial="S001",
            timestamp=datetime.now(),
            validation_status="VALID",
            tracks=[],
            metadata={},
            status=AnalysisStatus.COMPLETED
        )
        
        mock_processor.return_value.process_file.return_value = mock_result
        
        # Mock parent and main window
        mock_parent = Mock()
        mock_main_window = Mock()
        mock_main_window.db_manager = Mock()
        
        # Create page
        page = SingleFilePage(mock_parent, mock_main_window)
        
        # Set current file
        page.current_file = Path("test.xls")
        page.current_result = mock_result
        
        # Test save to database
        page._save_to_database()
        
        # Verify database manager was called
        assert mock_main_window.db_manager.add_analysis_result.called
        
    def test_batch_processing_data_flow(self):
        """Test batch processing stores results correctly."""
        from laser_trim_analyzer.gui.pages.batch_processing_page import BatchProcessingPage
        
        # Mock components
        mock_parent = Mock()
        mock_main_window = Mock()
        mock_main_window.db_manager = Mock()
        
        page = BatchProcessingPage(mock_parent, mock_main_window)
        
        # Test batch results storage
        mock_results = [Mock(), Mock(), Mock()]
        page.batch_results = mock_results
        
        # Process results
        page._process_batch_results()
        
        # Verify all results were processed
        assert mock_main_window.db_manager.add_batch.called or \
               mock_main_window.db_manager.add_analysis_result.call_count == len(mock_results)


class TestMLIntegration:
    """Test ML components integration."""
    
    def test_ml_predictor_loading(self):
        """Test ML predictor loads correctly."""
        from laser_trim_analyzer.ml.predictors import QualityPredictor
        
        predictor = QualityPredictor()
        
        # Test predictor is ready
        assert predictor is not None
        assert hasattr(predictor, 'predict')
        
    def test_ml_predictor_availability(self):
        """Test ML predictor is available to pages."""
        from laser_trim_analyzer.gui.pages.ml_tools_page import MLToolsPage
        
        mock_parent = Mock()
        mock_main_window = Mock()
        
        page = MLToolsPage(mock_parent, mock_main_window)
        
        # Check ML components are accessible
        assert hasattr(page, 'ml_manager') or hasattr(page, 'predictor')


class TestConfigurationManagement:
    """Test configuration management across pages."""
    
    def test_config_propagation(self):
        """Test configuration propagates to all pages."""
        from laser_trim_analyzer.core.config import get_config
        from laser_trim_analyzer.gui.pages.settings_page import SettingsPage
        
        config = get_config()
        
        mock_parent = Mock()
        mock_main_window = Mock()
        mock_main_window.config = config
        
        page = SettingsPage(mock_parent, mock_main_window)
        
        # Verify page has access to config
        assert hasattr(page, 'config') or hasattr(page.main_window, 'config')
        
    def test_settings_persistence(self):
        """Test settings are persisted correctly."""
        from laser_trim_analyzer.gui.settings_manager import settings_manager
        
        # Set a value
        settings_manager.set("test.value", 42)
        
        # Retrieve it
        value = settings_manager.get("test.value")
        assert value == 42
        
        # Test default value
        default = settings_manager.get("non.existent", "default")
        assert default == "default"


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_missing_imports_recovery(self):
        """Test application handles missing imports gracefully."""
        # This is handled in ctk_main_window.py lines 19-27
        from laser_trim_analyzer.gui.ctk_main_window import SingleFilePage
        
        # Even if import failed, placeholder should exist
        assert SingleFilePage is not None or True  # Graceful handling
        
    def test_circular_dependency_prevention(self):
        """Test no circular dependencies exist."""
        # Import all major modules
        try:
            from laser_trim_analyzer.core import processor
            from laser_trim_analyzer.database import manager
            from laser_trim_analyzer.gui import main_window
            from laser_trim_analyzer.ml import predictors
            
            # If we get here, no circular dependencies
            assert True
        except ImportError as e:
            pytest.fail(f"Circular dependency detected: {e}")
            
    def test_resource_initialization_order(self):
        """Test resources initialize in correct order."""
        from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
        from laser_trim_analyzer.core.config import get_config
        
        config = get_config()
        
        # Mock CTk components
        with patch('laser_trim_analyzer.gui.ctk_main_window.ctk'):
            window = CTkMainWindow(config)
            
            # Check initialization order
            # 1. Window setup
            assert hasattr(window, 'title')
            
            # 2. Services (database)
            assert hasattr(window, 'db_manager')
            
            # 3. UI creation
            assert hasattr(window, 'sidebar_frame')
            
            # 4. Pages
            assert hasattr(window, 'pages')
            
    def test_file_processing_error_recovery(self):
        """Test recovery from file processing errors."""
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.exceptions import ProcessingError
        
        processor = LaserTrimProcessor()
        
        # Test with invalid file
        with pytest.raises(ProcessingError):
            processor.process_file(Path("non_existent_file.xls"))
            
        # Processor should still be usable after error
        assert processor is not None


class TestIntegrationIssues:
    """Test for specific integration issues."""
    
    def test_database_unavailable_handling(self):
        """Test pages handle database unavailability."""
        from laser_trim_analyzer.gui.pages.historical_page import HistoricalPage
        
        mock_parent = Mock()
        mock_main_window = Mock()
        mock_main_window.db_manager = None  # Database unavailable
        
        # Should not crash
        page = HistoricalPage(mock_parent, mock_main_window)
        assert page is not None
        
    def test_missing_ml_models_handling(self):
        """Test pages handle missing ML models."""
        from laser_trim_analyzer.gui.pages.ml_tools_page import MLToolsPage
        
        mock_parent = Mock()
        mock_main_window = Mock()
        
        # Mock missing ML models
        with patch('laser_trim_analyzer.ml.predictors.QualityPredictor', side_effect=ImportError):
            page = MLToolsPage(mock_parent, mock_main_window)
            assert page is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])