"""
Cached processor implementation that integrates caching with the main processor.

This module extends the FileProcessor with comprehensive caching capabilities
for improved performance and reduced redundant calculations.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .processor import FileProcessor
from .cache_manager import get_cache_manager, CacheManager
from .models import (
    ProcessingResult, AnalysisResult, FileMetadata,
    TrackData, ValidationResult
)
from .exceptions import ProcessingError, CacheError
from ..utils.file_utils import calculate_file_hash

logger = logging.getLogger(__name__)


class CachedFileProcessor(FileProcessor):
    """
    Extended file processor with integrated caching capabilities.
    
    This processor caches:
    - Extracted data from Excel files
    - Analysis results
    - ML predictions
    - Validation results
    """
    
    def __init__(self, config, output_dir: Path):
        super().__init__(config, output_dir)
        self.cache = get_cache_manager()
        self._cache_hits = 0
        self._cache_misses = 0
        
    async def process_file(
        self, 
        file_path: Path,
        force_reprocess: bool = False
    ) -> ProcessingResult:
        """
        Process file with caching support.
        
        Args:
            file_path: Path to the Excel file
            force_reprocess: Force reprocessing even if cached
            
        Returns:
            ProcessingResult with cached or fresh data
        """
        # Generate cache key based on file content
        file_hash = calculate_file_hash(file_path)
        cache_key = f"process:{file_path.name}:{file_hash}"
        
        # Check cache unless forced to reprocess
        if not force_reprocess:
            cached_result = self.cache.get(cache_key, cache_type='analysis_results')
            if cached_result:
                logger.info(f"Cache hit for file: {file_path.name}")
                self._cache_hits += 1
                return self._deserialize_processing_result(cached_result)
        
        logger.info(f"Cache miss for file: {file_path.name}, processing...")
        self._cache_misses += 1
        
        # Process file normally
        result = await super().process_file(file_path)
        
        # Cache the result
        try:
            serialized = self._serialize_processing_result(result)
            self.cache.set(
                cache_key, 
                serialized, 
                cache_type='analysis_results',
                ttl=86400  # 24 hour TTL
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
        
        return result
    
    def _extract_data(self, file_path: Path) -> Tuple[pd.DataFrame, FileMetadata]:
        """Extract data with caching of parsed Excel content."""
        # Cache key for extracted data
        file_hash = calculate_file_hash(file_path)
        cache_key = f"excel_data:{file_path.name}:{file_hash}"
        
        # Check cache
        cached_data = self.cache.get(cache_key, cache_type='file_content')
        if cached_data:
            logger.debug(f"Using cached Excel data for {file_path.name}")
            return (
                pd.DataFrame(cached_data['data']),
                FileMetadata(**cached_data['metadata'])
            )
        
        # Extract data normally
        data, metadata = super()._extract_data(file_path)
        
        # Cache the extracted data
        try:
            cache_data = {
                'data': data.to_dict('records'),
                'metadata': metadata.dict()
            }
            self.cache.set(
                cache_key,
                cache_data,
                cache_type='file_content',
                ttl=3600  # 1 hour TTL
            )
        except Exception as e:
            logger.warning(f"Failed to cache Excel data: {e}")
        
        return data, metadata
    
    async def _run_ml_predictions(
        self,
        analysis_result: AnalysisResult,
        metadata: FileMetadata
    ) -> Dict[str, Any]:
        """Run ML predictions with caching."""
        if not self.ml_predictor or not self.ml_predictor.is_initialized:
            return {}
        
        # Create cache key from analysis data
        cache_key = self._create_ml_cache_key(analysis_result, metadata)
        
        # Check cache
        cached_predictions = self.cache.get_model_prediction(
            "laser_trim_predictor",
            cache_key
        )
        if cached_predictions:
            logger.debug("Using cached ML predictions")
            return cached_predictions
        
        # Run predictions normally
        predictions = await super()._run_ml_predictions(analysis_result, metadata)
        
        # Cache predictions
        if predictions:
            self.cache.cache_model_prediction(
                "laser_trim_predictor",
                cache_key,
                predictions,
                ttl=7200  # 2 hour TTL
            )
        
        return predictions
    
    def _validate_data(
        self,
        data: pd.DataFrame,
        metadata: FileMetadata
    ) -> ValidationResult:
        """Validate data with caching of validation results."""
        # Create cache key
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(data).values.tobytes()
        ).hexdigest()[:16]
        cache_key = f"validation:{metadata.model_number}:{data_hash}"
        
        # Check cache
        cached_validation = self.cache.get(cache_key, cache_type='general')
        if cached_validation:
            logger.debug("Using cached validation result")
            return ValidationResult(**cached_validation)
        
        # Validate normally
        validation_result = super()._validate_data(data, metadata)
        
        # Cache validation result
        try:
            self.cache.set(
                cache_key,
                validation_result.dict(),
                cache_type='general',
                ttl=1800  # 30 minute TTL
            )
        except Exception as e:
            logger.warning(f"Failed to cache validation result: {e}")
        
        return validation_result
    
    def clear_file_cache(self, file_path: Path) -> None:
        """Clear all cached data for a specific file."""
        file_name = file_path.name
        
        # Clear different cache types
        patterns = [
            f"process:{file_name}:*",
            f"excel_data:{file_name}:*",
            f"validation:*{file_path.stem}*"
        ]
        
        total_cleared = 0
        for pattern in patterns:
            for cache_type in ['analysis_results', 'file_content', 'general']:
                cleared = self.cache.invalidate_pattern(pattern, cache_type)
                total_cleared += cleared
        
        logger.info(f"Cleared {total_cleared} cache entries for {file_name}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        stats = {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_stats': self.cache.get_stats()
        }
        
        return stats
    
    def _serialize_processing_result(self, result: ProcessingResult) -> Dict[str, Any]:
        """Serialize ProcessingResult for caching."""
        return {
            'file_path': str(result.file_path),
            'metadata': result.metadata.dict(),
            'validation_result': result.validation_result.dict() if result.validation_result else None,
            'analysis_result': result.analysis_result.dict() if result.analysis_result else None,
            'ml_predictions': result.ml_predictions,
            'processing_time': result.processing_time,
            'error': result.error
        }
    
    def _deserialize_processing_result(self, data: Dict[str, Any]) -> ProcessingResult:
        """Deserialize cached data to ProcessingResult."""
        from .models import ProcessingResult, FileMetadata, ValidationResult, AnalysisResult
        
        return ProcessingResult(
            file_path=Path(data['file_path']),
            metadata=FileMetadata(**data['metadata']),
            validation_result=ValidationResult(**data['validation_result']) if data['validation_result'] else None,
            analysis_result=AnalysisResult(**data['analysis_result']) if data['analysis_result'] else None,
            ml_predictions=data['ml_predictions'],
            processing_time=data['processing_time'],
            error=data['error']
        )
    
    def _create_ml_cache_key(
        self,
        analysis_result: AnalysisResult,
        metadata: FileMetadata
    ) -> str:
        """Create cache key for ML predictions."""
        # Use key features that affect predictions
        key_data = {
            'model': metadata.model_number,
            'sigma': float(analysis_result.sigma_analysis.overall_sigma),
            'linearity': float(analysis_result.linearity_analysis.overall_linearity),
            'track_count': len(analysis_result.tracks),
            'resistance_range': (
                float(analysis_result.resistance_analysis.min_resistance),
                float(analysis_result.resistance_analysis.max_resistance)
            )
        }
        
        # Create deterministic hash
        import json
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class CachedBatchProcessor:
    """Batch processor with caching support."""
    
    def __init__(self, config, output_dir: Path):
        self.processor = CachedFileProcessor(config, output_dir)
        self.cache = get_cache_manager()
        
    async def process_batch(
        self,
        file_paths: List[Path],
        force_reprocess: bool = False,
        progress_callback: Optional[callable] = None
    ) -> List[ProcessingResult]:
        """
        Process batch of files with caching.
        
        Args:
            file_paths: List of files to process
            force_reprocess: Force reprocessing of all files
            progress_callback: Callback for progress updates
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Check if result is already cached
                if not force_reprocess:
                    file_hash = calculate_file_hash(file_path)
                    cache_key = f"process:{file_path.name}:{file_hash}"
                    cached = self.cache.get(cache_key, cache_type='analysis_results')
                    
                    if cached:
                        result = self.processor._deserialize_processing_result(cached)
                        results.append(result)
                        
                        if progress_callback:
                            progress_callback(i + 1, len(file_paths), f"Loaded from cache: {file_path.name}")
                        continue
                
                # Process file
                result = await self.processor.process_file(file_path, force_reprocess)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(file_paths), f"Processed: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Create error result
                from .models import ProcessingResult
                error_result = ProcessingResult(
                    file_path=file_path,
                    metadata=None,
                    validation_result=None,
                    analysis_result=None,
                    ml_predictions=None,
                    processing_time=0,
                    error=str(e)
                )
                results.append(error_result)
        
        return results
    
    def clear_batch_cache(self, file_paths: List[Path]) -> None:
        """Clear cache for multiple files."""
        for file_path in file_paths:
            self.processor.clear_file_cache(file_path)
    
    def preload_cache(self, file_paths: List[Path]) -> int:
        """
        Preload cache by processing files in background.
        
        Returns:
            Number of files preloaded
        """
        preloaded = 0
        
        for file_path in file_paths:
            file_hash = calculate_file_hash(file_path)
            cache_key = f"process:{file_path.name}:{file_hash}"
            
            # Skip if already cached
            if self.cache.exists(cache_key, cache_type='analysis_results'):
                continue
            
            try:
                # Process file to populate cache
                import asyncio
                asyncio.run(self.processor.process_file(file_path))
                preloaded += 1
            except Exception as e:
                logger.warning(f"Failed to preload {file_path}: {e}")
        
        return preloaded


def create_cached_processor(config, output_dir: Path) -> CachedFileProcessor:
    """Factory function to create cached processor."""
    return CachedFileProcessor(config, output_dir)


def create_cached_batch_processor(config, output_dir: Path) -> CachedBatchProcessor:
    """Factory function to create cached batch processor."""
    return CachedBatchProcessor(config, output_dir)