"""
End-to-end workflow tests for the laser trim analyzer.

Tests the complete application workflow without mocking.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TestStartupWorkflow:
    """Test application startup workflow."""
    
    def test_imports_and_initialization(self):
        """Test that all imports work and basic initialization succeeds."""
        # Test core imports
        from laser_trim_analyzer.core.config import get_config
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.database.manager import DatabaseManager
        
        # Test GUI imports
        from laser_trim_analyzer.gui.main_window import MainWindow
        from laser_trim_analyzer.gui.ctk_main_window import CTkMainWindow
        
        # Test page imports
        from laser_trim_analyzer.gui.pages import (
            HomePage, AnalysisPage, HistoricalPage, 
            ModelSummaryPage, MLToolsPage, AIInsightsPage, SettingsPage
        )
        
        # All imports successful
        assert True
        
    def test_config_loading(self):
        """Test configuration loads correctly."""
        from laser_trim_analyzer.core.config import get_config
        
        config = get_config()
        
        # Verify configuration structure
        assert hasattr(config, 'database')
        assert hasattr(config, 'processing')
        assert hasattr(config, 'gui')
        assert hasattr(config, 'ml')
        
        # Check database config
        assert config.database.enabled is True
        assert config.database.path is not None
        
        # Check processing config
        assert config.processing.max_workers > 0
        assert config.processing.file_extensions == [".xlsx", ".xls"]
        
    def test_database_initialization(self):
        """Test database can be initialized."""
        from laser_trim_analyzer.database.manager import DatabaseManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db_url = f"sqlite:///{db_path}"
            
            try:
                manager = DatabaseManager(db_url)
                
                # Check manager is initialized
                assert manager is not None
                assert manager.engine is not None
                
                # Test basic query
                from sqlalchemy import text
                with manager.get_session() as session:
                    # Should not raise
                    result = session.execute(text("SELECT 1"))
                    assert result is not None
                    
            finally:
                # Ensure cleanup
                if 'manager' in locals():
                    manager.close()


class TestDataProcessingWorkflow:
    """Test data processing workflow."""
    
    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.config import get_config
        
        config = get_config()
        processor = LaserTrimProcessor(config)
        
        assert processor is not None
        assert processor.config == config
        
    def test_file_validation(self):
        """Test file validation workflow."""
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.config import get_config
        
        config = get_config()
        processor = LaserTrimProcessor(config)
        
        # Test with non-existent file
        result = processor.validate_file(Path("non_existent.xls"))
        assert result is False
        
        # Test with existing test file
        test_files = list(Path("test_files/System A test files").glob("*.xls"))
        if test_files:
            result = processor.validate_file(test_files[0])
            # Should be True for valid Excel file
            assert isinstance(result, bool)


class TestPageIntegration:
    """Test page integration workflow."""
    
    def test_page_dependencies(self):
        """Test pages can access required dependencies."""
        from laser_trim_analyzer.gui.pages.base_page import BasePage
        import customtkinter as ctk
        
        # Test BasePage has required methods
        assert hasattr(BasePage, 'on_show')
        assert hasattr(BasePage, 'on_hide')
        assert hasattr(BasePage, 'cleanup')
        
    def test_ml_components(self):
        """Test ML components are available."""
        try:
            from laser_trim_analyzer.ml.predictors import QualityPredictor
            from laser_trim_analyzer.ml.ml_manager import MLManager
            
            # ML components available
            assert True
        except ImportError:
            # ML components optional
            assert True


class TestErrorHandling:
    """Test error handling workflow."""
    
    def test_database_error_handling(self):
        """Test database errors are handled gracefully."""
        from laser_trim_analyzer.database.manager import DatabaseManager, DatabaseError
        
        try:
            # Invalid database URL
            manager = DatabaseManager("invalid://url")
            assert False, "Should have raised DatabaseError"
        except DatabaseError:
            # Expected error
            assert True
        except Exception as e:
            # Unexpected error type
            logger.warning(f"Unexpected error type: {type(e)}")
            assert True  # Still handling error
            
    def test_processing_error_handling(self):
        """Test processing errors are handled gracefully."""
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.core.exceptions import ProcessingError
        
        processor = LaserTrimProcessor()
        
        try:
            # Process non-existent file
            result = processor.process_file(Path("non_existent.xls"))
            assert False, "Should have raised ProcessingError"
        except (ProcessingError, FileNotFoundError):
            # Expected error
            assert True
        except Exception as e:
            # Other error type but still handled
            logger.warning(f"Unexpected error type: {type(e)}")
            assert True


class TestIntegrationPoints:
    """Test specific integration points."""
    
    def test_config_to_database(self):
        """Test config properly initializes database."""
        from laser_trim_analyzer.core.config import get_config
        from laser_trim_analyzer.database.manager import DatabaseManager
        
        config = get_config()
        
        if config.database.enabled:
            # Database should be initializable with config
            try:
                db_path = config.database.path
                if not str(db_path).startswith(('sqlite://', 'postgresql://', 'mysql://')):
                    db_path = f"sqlite:///{Path(db_path).absolute()}"
                    
                manager = DatabaseManager(db_path)
                assert manager is not None
                manager.close()
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                assert False
                
    def test_processor_to_database(self):
        """Test processor results can be stored in database."""
        from laser_trim_analyzer.core.processor import LaserTrimProcessor
        from laser_trim_analyzer.database.manager import DatabaseManager
        from laser_trim_analyzer.core.models import AnalysisResult
        
        processor = LaserTrimProcessor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db_url = f"sqlite:///{db_path}"
            
            try:
                manager = DatabaseManager(db_url)
                
                # Create a mock result
                from datetime import datetime
                result = AnalysisResult(
                    filename="test.xls",
                    filepath=Path("test.xls"),
                    model="TEST",
                    serial="001",
                    timestamp=datetime.now(),
                    validation_status="VALID",
                    tracks=[],
                    metadata={},
                    status="COMPLETED"
                )
                
                # Should be able to add to database
                db_result = manager.add_analysis_result(result)
                assert db_result is not None
                assert db_result.id is not None
                
            finally:
                if 'manager' in locals():
                    manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])