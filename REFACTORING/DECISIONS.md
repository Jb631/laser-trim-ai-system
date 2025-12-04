# Architectural Decision Records (ADR)

**Purpose**: Document all significant architectural decisions made during refactoring

**Format**: Each decision gets a unique ID (ADR-XXX) and follows this template

---

## ADR-001: Use Feature Flags for Major Changes

**Date**: 2025-01-25
**Status**: Accepted
**Context**: Aggressive refactoring with 6 processors → 1 unified processor requires careful rollout
**Decision**: Use feature flags to enable/disable new architecture
**Consequences**:
- **Positive**: Can deploy with new code but old behavior (safe)
- **Positive**: Easy rollback if issues discovered
- **Positive**: A/B testing possible
- **Negative**: Temporary code duplication
- **Mitigation**: Remove flags after 1 stable release (v3.1)

**Implementation**:
```yaml
# config/development.yaml
refactoring:
  use_unified_processor: false  # Default OFF
  use_ml_failure_predictor: false
  use_split_chart_modules: false
```

**Related**: CLAUDE_REFACTOR_RULES.md Section 7

---

## ADR-002: Incremental Processing via Database Tracking

**Date**: 2025-01-25
**Status**: Accepted
**Context**: Users add 100-200 new files daily to folders with 1000+ existing files. Processing all files takes 45+ minutes.
**Decision**: Add `processed_files` table to track already-analyzed files. Skip files that exist in DB.
**Alternatives Considered**:
1. File system timestamps (rejected - unreliable)
2. Separate cache file (rejected - out of sync risk)
3. Database table (chosen - authoritative source)

**Consequences**:
- **Positive**: 10x faster for incremental updates (6 min vs 60 min)
- **Positive**: Authoritative record in database
- **Positive**: Can detect duplicates via file hash
- **Negative**: Need migration for existing databases
- **Negative**: Extra DB query per file (minimal overhead)

**Implementation**:
```python
class ProcessedFile(Base):
    filename = Column(String, index=True)
    file_hash = Column(String(64), index=True)  # SHA-256
    processed_date = Column(DateTime)
```

**Related**: Phase 1, Day 2

---

## ADR-003: Remove AnalyticsEngine (1,052 lines)

**Date**: 2025-01-25
**Status**: Accepted
**Context**: AnalyticsEngine class exists but is never imported or used anywhere in codebase.
**Decision**: Delete `analysis/analytics_engine.py` completely.
**Alternatives Considered**:
1. Keep for future use (rejected - YAGNI principle)
2. Archive outside codebase (rejected - git history is archive)
3. Delete (chosen - reduce maintenance burden)

**Consequences**:
- **Positive**: -1,052 lines (cleaner codebase)
- **Positive**: Less confusion about what's used
- **Positive**: Faster searches/navigation
- **Negative**: None (never used)
- **Recovery**: Available in git history if ever needed

**Verification**:
```bash
grep -r "AnalyticsEngine" src/ --include="*.py"
# Result: Only in analytics_engine.py itself
```

**Related**: Phase 1, Day 3

---

## ADR-004: Unify 6 Processors Using Strategy Pattern

**Date**: 2025-12-04
**Status**: Accepted (Design Complete)
**Context**: 6 different processor classes with 36% duplicated logic (2,100 lines). Users confused about which processor to use. Maintenance burden: fixes must be applied to multiple places.

**Decision**: Create UnifiedProcessor with pluggable strategies for different scenarios.

**Processor Analysis** (Day 4 findings):

| Processor | Lines | Duplicated | Unique | Primary Purpose |
|-----------|-------|------------|--------|-----------------|
| LaserTrimProcessor | 2,682 | ~800 | ~1,882 | Core analysis logic |
| FastProcessor | 1,499 | ~800 | ~699 | Parallel processing |
| LargeScaleProcessor | 1,189 | ~200 | ~989 | Memory/chunking |
| CachedFileProcessor | 383 | ~300 | ~83 | File caching |
| CachedBatchProcessor | (same file) | - | - | Batch caching |
| SecureFileProcessor | ~234 | ~50 | ~184 | Security wrapper |
| **Total** | **5,753** | **~2,100** | **~3,653** | |

**Key Duplicated Methods** (1,280 lines):
- `_analyze_sigma()` / `_analyze_sigma_fast()` (~80 lines x2)
- `_analyze_linearity()` / `_analyze_linearity_fast()` (~70 lines x2)
- `_extract_trim_data()` / `_extract_trim_data_fast()` (~180 lines x2)
- `_determine_overall_status()` (~20 lines x2)
- `_calculate_failure_prediction()` (~70 lines x2)
- Plus 5 more duplicated methods...

**Detailed Design**:

```python
# Base processor with all shared logic
class UnifiedProcessor:
    """Single processor that replaces all 6 existing processors."""

    def __init__(
        self,
        config: Config,
        strategy: str = 'auto',  # 'auto', 'standard', 'turbo', 'memory_safe'
        enable_caching: bool = False,
        enable_security: bool = True,
        incremental: bool = True,  # Skip already-processed files
    ):
        self.config = config
        self.strategy = self._create_strategy(strategy, config)
        self.caching = CachingLayer() if enable_caching else None
        self.security = SecurityLayer() if enable_security else None
        self.incremental = incremental

        # Shared components (initialized once)
        self.sigma_analyzer = SigmaAnalyzer()
        self.linearity_analyzer = LinearityAnalyzer()
        self.resistance_analyzer = ResistanceAnalyzer()
        self.ml_engine = MLEngine(config)
        self.db_manager = DatabaseManager(config)

    async def process_file(self, file_path: str) -> AnalysisResult:
        """Process a single file through the pipeline."""
        # 1. Security validation (optional layer)
        if self.security:
            self.security.validate(file_path)

        # 2. Check incremental (skip if already processed)
        if self.incremental and self.db_manager.is_file_processed(file_path):
            return self.db_manager.get_cached_result(file_path)

        # 3. Check cache (optional layer)
        if self.caching and (cached := self.caching.get(file_path)):
            return cached

        # 4. Delegate to strategy for actual processing
        result = await self.strategy.process(file_path, self)

        # 5. Store result
        if self.caching:
            self.caching.set(file_path, result)
        await self.db_manager.save_result(result)
        await self.db_manager.mark_file_processed(file_path)

        return result

    async def process_batch(
        self,
        files: List[str],
        progress_callback: Optional[Callable] = None,
    ) -> AsyncGenerator[AnalysisResult, None]:
        """Process multiple files with progress tracking."""
        # Filter already-processed if incremental
        if self.incremental:
            files = self.db_manager.get_unprocessed_files(files)

        # Delegate to strategy
        async for result in self.strategy.process_batch(files, self, progress_callback):
            yield result

    # Shared analysis methods (used by all strategies)
    def _analyze_sigma(self, data: TrackData) -> SigmaResult:
        return self.sigma_analyzer.analyze(data)

    def _analyze_linearity(self, data: TrackData) -> LinearityResult:
        return self.linearity_analyzer.analyze(data)

    def _analyze_resistance(self, data: TrackData) -> ResistanceResult:
        return self.resistance_analyzer.analyze(data)

    def _determine_overall_status(self, analyses: Dict) -> str:
        # Single implementation (not duplicated)
        ...

    def _create_strategy(self, strategy: str, config: Config) -> ProcessingStrategy:
        if strategy == 'auto':
            return AutoStrategy(config)
        elif strategy == 'turbo':
            return TurboStrategy(config)
        elif strategy == 'memory_safe':
            return MemorySafeStrategy(config)
        else:
            return StandardStrategy(config)


# Strategy interface
class ProcessingStrategy(ABC):
    """Base class for all processing strategies."""

    @abstractmethod
    async def process(self, file_path: str, processor: UnifiedProcessor) -> AnalysisResult:
        """Process a single file."""
        pass

    @abstractmethod
    async def process_batch(
        self,
        files: List[str],
        processor: UnifiedProcessor,
        progress_callback: Optional[Callable],
    ) -> AsyncGenerator[AnalysisResult, None]:
        """Process multiple files."""
        pass


# Concrete strategies
class StandardStrategy(ProcessingStrategy):
    """Default sequential processing. Replaces LaserTrimProcessor."""

    async def process(self, file_path: str, processor: UnifiedProcessor) -> AnalysisResult:
        # Extract data
        data = await self._extract_data(file_path)

        # Run analyses using shared methods
        sigma = processor._analyze_sigma(data)
        linearity = processor._analyze_linearity(data)
        resistance = processor._analyze_resistance(data)

        # Aggregate results
        return self._create_result(file_path, sigma, linearity, resistance)


class TurboStrategy(ProcessingStrategy):
    """Parallel processing with ProcessPoolExecutor. Replaces FastProcessor."""

    def __init__(self, config: Config):
        self.max_workers = config.get('turbo_workers', 4)
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    async def process_batch(self, files, processor, progress_callback):
        # Submit all files to process pool
        futures = [self.executor.submit(self._process_single, f) for f in files]

        for future in as_completed(futures):
            result = future.result()
            if progress_callback:
                progress_callback(result)
            yield result


class MemorySafeStrategy(ProcessingStrategy):
    """Memory-managed chunked processing. Replaces LargeScaleProcessor."""

    def __init__(self, config: Config):
        self.chunk_size = config.get('chunk_size', 100)
        self.memory_threshold = config.get('memory_threshold_mb', 1000)

    async def process_batch(self, files, processor, progress_callback):
        # Process in chunks with memory monitoring
        for chunk in self._chunk_files(files, self.chunk_size):
            # Check memory before each chunk
            self._enforce_memory_limit()

            for file_path in chunk:
                yield await processor.strategy.process(file_path, processor)

            # GC after each chunk
            gc.collect()


class AutoStrategy(ProcessingStrategy):
    """Auto-select strategy based on file count and system resources."""

    def __init__(self, config: Config):
        self.standard = StandardStrategy(config)
        self.turbo = TurboStrategy(config)
        self.memory_safe = MemorySafeStrategy(config)

    async def process_batch(self, files, processor, progress_callback):
        file_count = len(files)
        available_memory = psutil.virtual_memory().available / (1024**2)

        # Select strategy based on conditions
        if file_count > 1000 or available_memory < 500:
            strategy = self.memory_safe
        elif file_count > 10:
            strategy = self.turbo
        else:
            strategy = self.standard

        async for result in strategy.process_batch(files, processor, progress_callback):
            yield result
```

**Migration Path** (Phase 2 Implementation):

1. **Day 1-2**: Create UnifiedProcessor with StandardStrategy
   - Extract shared methods from LaserTrimProcessor
   - Create ProcessingStrategy interface
   - Implement StandardStrategy

2. **Day 3**: Add TurboStrategy
   - Extract parallel logic from FastProcessor
   - Implement ProcessPoolExecutor integration
   - Test with 100, 500, 1000 files

3. **Day 4**: Add MemorySafeStrategy
   - Extract chunking from LargeScaleProcessor
   - Implement memory monitoring
   - Add progress/recovery support

4. **Day 5**: Add layers and migration
   - Implement CachingLayer (from Cached*Processor)
   - Implement SecurityLayer (from SecureFileProcessor)
   - Update GUI pages to use UnifiedProcessor
   - Add feature flag for rollback

**Caller Migration**:

```python
# Before (6 different entry points)
if turbo_mode:
    processor = FastProcessor(config)
elif file_count > 1000:
    processor = LargeScaleProcessor(config)
else:
    processor = LaserTrimProcessor(config)

# After (single entry point)
processor = UnifiedProcessor(
    config,
    strategy='auto',  # Auto-selects best strategy
    enable_caching=True,
    incremental=True,
)
```

**Expected Results**:
- **Lines Removed**: ~2,000 (36% duplication eliminated)
- **Code Complexity**: 6 classes → 1 class + 4 strategies
- **Testing**: Simpler (test shared base + each strategy)
- **Maintenance**: Single place for bug fixes

**Consequences**:
- **Positive**: Single entry point (clear API)
- **Positive**: -36% code duplication (~2,000 lines)
- **Positive**: Easier testing (test strategies independently)
- **Positive**: Composable (layers can wrap any strategy)
- **Positive**: Auto-selection removes user confusion
- **Negative**: Requires refactor of all callers (7 GUI pages, CLI)
- **Mitigation**: Feature flag for gradual rollout (ADR-001)
- **Mitigation**: Keep old processors during transition

**Verification**:
```bash
# After Phase 2, verify:
pytest tests/                           # All tests pass
python scripts/benchmark_processing.py --files 1000  # No regression
grep -r "LaserTrimProcessor" src/gui/   # Should be 0 (migrated)
wc -l src/laser_trim_analyzer/core/*.py # Should be ~3,500 lines (down from 5,753)
```

**Related**:
- Phase 1, Day 4 (this analysis)
- Phase 2 (implementation)
- ADR-001 (feature flags)

---

## ADR-005: Wire ML Models to Processing Pipeline

**Date**: TBD (Phase 3)
**Status**: Proposed
**Context**: ML models (FailurePredictor, DriftDetector) exist but processing uses hardcoded formulas.
**Decision**: Replace hardcoded formulas with ML predictions, with formula as fallback.

**Priority Order**:
1. Check for ML prediction
2. Fall back to formula if ML not available
3. Log which method used for debugging

**Implementation Pattern**:
```python
def _predict_failure(self, ...):
    # Try ML first
    if self.ml_engine and self.ml_engine.has_model('failure_predictor'):
        prediction = self.ml_engine.predict_failure(data)
        logger.info("Using ML failure prediction")
        return prediction

    # Fallback to formula
    logger.info("Using formula-based failure prediction (ML not available)")
    return self._calculate_failure_prediction(...)
```

**Consequences**:
- **Positive**: Better accuracy (ML learns from data)
- **Positive**: Graceful degradation (formula fallback)
- **Positive**: Clear logging of which method used
- **Negative**: Slight performance overhead for ML
- **Mitigation**: Batch predictions to reduce overhead

**Related**: Phase 3

---

## ADR-006: Split Large Files by Logical Modules

**Date**: TBD (Phase 4)
**Status**: Proposed
**Context**: 4 files >3000 lines each (chart_widget.py: 4,266 lines, historical_page.py: 4,261 lines, etc.)
**Decision**: Extract logical modules, keep original file as orchestrator/facade.

**Example: chart_widget.py**:
```
Before:
chart_widget.py (4,266 lines)

After:
chart_widget.py (500 lines) - Orchestrator
charts/
  ├── line_charts.py
  ├── bar_charts.py
  ├── validation.py
  └── theme.py
```

**Consequences**:
- **Positive**: Easier navigation (find specific chart type)
- **Positive**: Easier testing (test modules independently)
- **Positive**: Easier maintenance (change one chart type)
- **Positive**: -75% in original file
- **Negative**: More files to manage
- **Mitigation**: Clear directory structure and naming

**Related**: Phase 4

---

## Template for New ADRs

```markdown
## ADR-XXX: [Decision Title]

**Date**: YYYY-MM-DD
**Status**: Proposed / Accepted / Deprecated / Superseded by ADR-YYY
**Context**: [What is the issue we're seeing that is motivating this decision?]
**Decision**: [What is the change we're proposing/have agreed to?]
**Alternatives Considered**:
1. Alternative 1 (rejected - reason)
2. Alternative 2 (rejected - reason)
3. Chosen alternative (selected - reason)

**Consequences**:
- **Positive**: [Good outcomes]
- **Negative**: [Tradeoffs/costs]
- **Mitigation**: [How we address negatives]

**Implementation**: [Code examples, config changes]
**Related**: [Phase, related ADRs, related files]
**Verification**: [How to verify this decision was correct]
```

---

## Decision Index

| ID | Title | Status | Phase | Date |
|----|-------|--------|-------|------|
| ADR-001 | Feature Flags for Major Changes | Accepted | All | 2025-01-25 |
| ADR-002 | Incremental Processing via DB | Accepted | 1 | 2025-01-25 |
| ADR-003 | Remove AnalyticsEngine | Accepted | 1 | 2025-01-25 |
| ADR-004 | Unify Processors (Strategy Pattern) | **Accepted (Design Complete)** | 2 | 2025-12-04 |
| ADR-005 | Wire ML Models to Pipeline | Proposed | 3 | TBD |
| ADR-006 | Split Large Files by Modules | Proposed | 4 | TBD |

---

**Last Updated**: 2025-12-04 (ADR-004 Design Complete)
**Next Review**: After each phase completion
