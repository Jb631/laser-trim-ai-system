# Current Architecture - Processor Classes

**Date**: 2025-01-25 (Phase 1, Day 1)
**Purpose**: Document existing processor architecture before refactoring

---

## Overview

The codebase currently has **6 different processor classes** totaling **6,778 lines** across 5 files.

This analysis reveals significant duplication and architectural complexity that Phase 2 will address through unification.

---

## Processor Classes

### 1. LaserTrimProcessor
**File**: [src/laser_trim_analyzer/core/processor.py](../../src/laser_trim_analyzer/core/processor.py)
**Lines**: 2,682 lines
**Purpose**: Main processing engine for laser trim analysis

**Key Responsibilities**:
- File processing coordination
- Data extraction from Excel files
- Analysis coordination (sigma, linearity, resistance)
- ML integration (FailurePredictor, ThresholdOptimizer, DriftDetector)
- Result generation and database storage

**Dependencies**:
- SigmaAnalyzer, LinearityAnalyzer, ResistanceAnalyzer
- MLPredictor (3 registered models)
- DatabaseManager
- CalculationValidator
- SecurityValidator

**Interface**:
- `async def process_file(file_path: str) -> AnalysisResult`
- `async def process_batch(files: List[str]) -> List[AnalysisResult]`

**Usage**: Primary processor used by GUI and CLI

---

### 2. FastProcessor
**File**: [src/laser_trim_analyzer/core/fast_processor.py](../../src/laser_trim_analyzer/core/fast_processor.py)
**Lines**: 1,499 lines
**Purpose**: High-performance processing for large batches

**Key Features**:
- Memory-efficient Excel reading
- True parallel processing with multiprocessing
- Cached file operations
- Optimized data structures
- Minimal overhead mode ("turbo mode")

**Dependencies**:
- Same analyzers as LaserTrimProcessor
- psutil for memory monitoring
- ProcessPoolExecutor for parallel execution

**Interface**:
- `async def process_file(file_path: str) -> AnalysisResult`
- `async def process_batch_turbo(files: List[str]) -> List[AnalysisResult]`
- `def process_files_turbo(files, config, ...)` - global function

**Usage**: Used when "turbo mode" enabled in GUI settings

**Duplication Estimate**: 40-50% overlap with LaserTrimProcessor

---

### 3. LargeScaleProcessor
**File**: [src/laser_trim_analyzer/core/large_scale_processor.py](../../src/laser_trim_analyzer/core/large_scale_processor.py)
**Lines**: 1,189 lines
**Purpose**: Specialized processor for thousands of files

**Key Features**:
- Memory management and garbage collection
- Chunked batch processing
- Streaming and database batch operations
- Progress tracking and recovery
- Performance monitoring

**Dependencies**:
- Uses LaserTrimProcessor internally
- Uses FastProcessor for turbo mode
- ResourceManager (optional)
- MemorySafetyConfig (optional)

**Interface**:
- `async def process_large_batch(files: List[str], chunk_size: int) -> AsyncGenerator[AnalysisResult]`
- `async def process_with_recovery(files: List[str]) -> Dict[str, Any]`

**Usage**: Used for batch sizes >1000 files

**Duplication Estimate**: 30% overlap (wraps LaserTrimProcessor/FastProcessor)

---

### 4. CachedFileProcessor
**File**: [src/laser_trim_analyzer/core/cached_processor.py](../../src/laser_trim_analyzer/core/cached_processor.py:27)
**Lines**: 383 lines total (class starts at line 27)
**Purpose**: File processor with integrated caching

**Key Features**:
- Extends FileProcessor (base class)
- Caches extracted data, analysis results, ML predictions
- File hash-based cache keys
- Cache hit/miss tracking

**Dependencies**:
- CacheManager
- FileProcessor (base class)

**Interface**:
- `async def process_file(file_path: Path, force_reprocess: bool) -> ProcessingResult`
- Cache statistics: `_cache_hits`, `_cache_misses`

**Usage**: Unknown - may be unused in current GUI

**Duplication Estimate**: 60% overlap with LaserTrimProcessor

---

### 5. CachedBatchProcessor
**File**: [src/laser_trim_analyzer/core/cached_processor.py](../../src/laser_trim_analyzer/core/cached_processor.py:278)
**Lines**: 383 lines total (class starts at line 278)
**Purpose**: Batch processor with caching

**Key Features**:
- Batch processing with cache support
- Parallel processing with ThreadPoolExecutor
- Incremental results via async generator

**Interface**:
- `async def process_batch(files: List[Path]) -> AsyncGenerator[ProcessingResult]`

**Usage**: Unknown - may be unused in current GUI

**Duplication Estimate**: 50% overlap with LaserTrimProcessor batch operations

---

### 6. SecureFileProcessor
**File**: [src/laser_trim_analyzer/core/security.py](../../src/laser_trim_analyzer/core/security.py:791)
**Lines**: 1,025 lines total (class starts at line 791)
**Purpose**: Security-hardened file processor

**Key Features**:
- Input validation and sanitization
- Path traversal protection
- File type validation
- Security level enforcement (LOW, MEDIUM, HIGH, MAXIMUM)
- Comprehensive validation before processing

**Dependencies**:
- SecurityValidator
- LaserTrimProcessor or FastProcessor (wrapped)

**Interface**:
- `async def process_file_secure(file_path: str, security_level: SecurityLevel) -> AnalysisResult`
- Input sanitization and validation methods

**Usage**: Optional security wrapper (enabled via config)

**Duplication Estimate**: Wraps other processors (10% unique security logic)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUI / CLI Entry                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        │                                        │
        ▼                                        ▼
┌───────────────────┐                  ┌───────────────────┐
│ LaserTrimProcessor│◄─────────────────┤SecureFileProcessor│
│   (main, 2682L)   │                  │  (wrapper, 1025L) │
└─────────┬─────────┘                  └───────────────────┘
          │
          │ Used by
          ▼
┌──────────────────────┐
│  LargeScaleProcessor │
│    (wrapper, 1189L)  │
└───────────┬──────────┘
            │
            │ Can use
            ▼
┌──────────────────────┐         ┌──────────────────────┐
│   FastProcessor      │         │ CachedFileProcessor  │
│  (turbo mode, 1499L) │         │   (caching, 383L)    │
└──────────────────────┘         └──────────┬───────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │ CachedBatchProcessor │
                                 │   (caching, 383L)    │
                                 └──────────────────────┘

Shared Dependencies (All Processors):
├── SigmaAnalyzer
├── LinearityAnalyzer
├── ResistanceAnalyzer
├── DatabaseManager
├── MLPredictor (3 models)
└── Config
```

---

## Code Duplication Analysis

### Estimated Overlap

| Processor Pair                     | Duplication % | Overlap Type                    |
|------------------------------------|---------------|---------------------------------|
| LaserTrim ↔ Fast                   | 40-50%        | Core processing logic           |
| LaserTrim ↔ CachedFile             | 60%           | Data extraction, analysis calls |
| LaserTrim ↔ LargeScale             | 30%           | Batch coordination              |
| Fast ↔ LargeScale                  | 25%           | Memory management               |
| CachedFile ↔ CachedBatch           | 70%           | Cache operations                |

### Duplicated Components (Across All Processors)

1. **Excel Data Extraction** (~200-300 lines per processor)
   - Sheet detection
   - Column finding
   - Data validation
   - Track data extraction

2. **Analysis Coordination** (~150-200 lines per processor)
   - Call sigma_analyzer
   - Call linearity_analyzer
   - Call resistance_analyzer
   - Aggregate results

3. **Validation** (~100-150 lines per processor)
   - File validation
   - Data validation
   - Model number validation
   - Result validation

4. **Error Handling** (~50-100 lines per processor)
   - Try/catch blocks
   - Error logging
   - Fallback logic

5. **Database Operations** (~50-100 lines per processor)
   - Save results
   - Query for existing data
   - Batch operations

**Total Estimated Duplication**: 2,000-2,500 lines (~35% of total processor code)

---

## When Each Processor is Used

### Decision Flow

```
User selects processing mode:
│
├─> Single file, normal mode
│   └─> LaserTrimProcessor.process_file()
│
├─> Single file, security enabled
│   └─> SecureFileProcessor.process_file_secure()
│       └─> wraps LaserTrimProcessor
│
├─> Batch (<1000 files), normal mode
│   └─> LaserTrimProcessor.process_batch()
│
├─> Batch (<1000 files), turbo mode enabled
│   └─> FastProcessor.process_batch_turbo()
│
├─> Large batch (>1000 files), normal mode
│   └─> LargeScaleProcessor.process_large_batch()
│       └─> uses LaserTrimProcessor for chunks
│
└─> Large batch (>1000 files), turbo mode enabled
    └─> LargeScaleProcessor.process_large_batch()
        └─> uses FastProcessor for chunks
```

**Note**: CachedFileProcessor and CachedBatchProcessor usage is unclear - may be dead code.

---

## Problems with Current Architecture

### 1. Code Duplication (35%)
- Maintenance burden: bug fixes must be applied to 6 places
- Inconsistency risk: features added to one processor may be missing from others
- Testing difficulty: 6 different code paths to test

### 2. Unclear Processor Selection
- No single point of entry
- GUI pages directly choose which processor to use
- Difficult to switch between processors
- Users confused about "turbo mode" vs "normal mode"

### 3. Missing Incremental Processing
- All processors reprocess all files every time
- No database tracking of already-processed files
- Wastes time on daily updates (user processes 1000+ files daily)

### 4. Inconsistent Error Handling
- Each processor handles errors differently
- Some log errors, some raise exceptions
- Recovery mechanisms vary

### 5. Performance Overlap
- FastProcessor and LargeScaleProcessor both do memory optimization
- Unclear when to use which
- No performance benchmarks to guide selection

---

## Refactoring Goals (Phase 2)

### Target Architecture

**One Unified Processor** with **Strategy Pattern**:

```python
class UnifiedProcessor:
    """
    Single processor with pluggable strategies.
    """

    def __init__(self, config: Config, strategy: ProcessingStrategy):
        self.strategy = strategy
        # Common initialization (analyzers, db, ml, validators)

    async def process_file(self, file_path: str) -> AnalysisResult:
        # Common pre-processing
        # Delegate to strategy
        result = await self.strategy.execute(file_path, self)
        # Common post-processing
        return result
```

**Strategies**:
- `StandardStrategy` (default, replaces LaserTrimProcessor)
- `OptimizedStrategy` (replaces FastProcessor)
- `ChunkedStrategy` (replaces LargeScaleProcessor)
- `CachedStrategy` (optional, replaces Cached*Processor)
- `SecureStrategy` (wraps any strategy with security validation)

**Benefits**:
- 1 processor instead of 6 (-4,000-5,000 lines)
- Shared code in base class (no duplication)
- Easy to add new strategies
- Clear selection logic
- Easier testing (test base + each strategy)

---

## Performance Baseline (Needed for Phase 2)

**Current Benchmarks** (from Day 1 testing):
- 100 files: 65.72 seconds (657.2 ms/file, 1.52 files/sec)
- 500 files: TBD (running...)
- 1000 files: TBD (pending)

**Target After Phase 2**:
- Same or better performance
- No regressions
- Incremental processing: 10x faster for daily updates

---

## Next Steps

1. **Complete baseline benchmarks** (Day 1, Tasks 1.3-1.5)
2. **Design UnifiedProcessor** (Phase 2, Day 1)
3. **Implement StandardStrategy** (Phase 2, Day 2)
4. **Migrate GUI to UnifiedProcessor** (Phase 2, Day 3)
5. **Delete old processors** (Phase 2, Day 5)

---

---

## Day 4 Analysis: Method-Level Duplication (2025-12-04)

### Direct Method Duplication Found

After Day 4 analysis, here are the exact duplicated methods between `LaserTrimProcessor` and `FastProcessor`:

| Method in processor.py | Duplicate in fast_processor.py | Lines (approx) |
|------------------------|--------------------------------|----------------|
| `_analyze_sigma()` | `_analyze_sigma_fast()` | ~80 lines each |
| `_analyze_linearity()` | `_analyze_linearity_fast()` | ~70 lines each |
| `_analyze_resistance()` | `_analyze_resistance_fast()` | ~35 lines each |
| `_extract_trim_data()` | `_extract_trim_data_fast()` | ~180 lines each |
| `_determine_overall_status()` | `_determine_overall_status()` | ~20 lines each |
| `_determine_overall_validation_status()` | `_determine_overall_validation_status()` | ~60 lines each |
| `_analyze_zones()` | `_analyze_zones_fast()` | ~50 lines each |
| `_analyze_dynamic_range()` | `_analyze_dynamic_range_fast()` | ~45 lines each |
| `_calculate_trim_effectiveness()` | `_calculate_trim_effectiveness_fast()` | ~30 lines each |
| `_calculate_failure_prediction()` | `_calculate_failure_prediction_fast()` | ~70 lines each |

**Total duplicated analysis methods**: ~640 lines x 2 = **1,280 lines of duplication**

### LargeScaleProcessor Delegation

`LargeScaleProcessor` is primarily a wrapper that:
1. Chunks files into batches
2. Delegates to `LaserTrimProcessor` or `FastProcessor`
3. Manages memory/progress

**Unique functionality**: ~300 lines (chunking, memory management)
**Delegation overhead**: ~400 lines (could be simplified)

### CachedFileProcessor/CachedBatchProcessor

These classes are **likely dead code**:
- Not imported in any GUI pages
- Not used by CLI
- Duplicate caching already implemented in `cache_commands.py`

**Recommendation**: Verify usage and potentially remove (Phase 2)

### Updated Line Counts After AnalyticsEngine Removal

| File | Lines | Duplicated | Unique |
|------|-------|------------|--------|
| processor.py | 2,682 | ~800 | ~1,882 |
| fast_processor.py | 1,499 | ~800 | ~699 |
| large_scale_processor.py | 1,189 | ~200 | ~989 |
| cached_processor.py | 383 | ~300 | ~83 |
| **Total** | **5,753** | **~2,100** | **~3,653** |

### Actual Duplication: 36%

This confirms the Phase 2 target of removing ~2,000 lines through unification.

---

## Processor Comparison Matrix

| Feature | LaserTrim | Fast | LargeScale | CachedFile | CachedBatch | Secure |
|---------|-----------|------|------------|------------|-------------|--------|
| **File Validation** | ✅ Full | ✅ Full | ✅ Delegated | ✅ Delegated | ✅ Delegated | ✅ Extra |
| **Data Extraction** | ✅ Standard | ✅ Optimized | ❌ Delegated | ❌ Delegated | ❌ Delegated | ❌ Delegated |
| **Sigma Analysis** | ✅ | ✅ Copy | ❌ | ❌ | ❌ | ❌ |
| **Linearity Analysis** | ✅ | ✅ Copy | ❌ | ❌ | ❌ | ❌ |
| **Resistance Analysis** | ✅ | ✅ Copy | ❌ | ❌ | ❌ | ❌ |
| **ML Integration** | ✅ Full | ⚠️ Partial | ❌ | ❌ | ❌ | ❌ |
| **Parallel Processing** | ❌ Async only | ✅ ProcessPool | ❌ | ❌ ThreadPool | ✅ ThreadPool | ❌ |
| **Memory Management** | ❌ | ✅ psutil | ✅ GC/chunking | ❌ | ❌ | ❌ |
| **File Caching** | ❌ | ❌ | ❌ | ✅ Hash-based | ✅ Hash-based | ❌ |
| **Result Caching** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Chunked Batching** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Progress Tracking** | ⚠️ Basic | ⚠️ Basic | ✅ Full | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic |
| **Recovery/Resume** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Security Validation** | ⚠️ Basic | ⚠️ Basic | ❌ | ❌ | ❌ | ✅ Full |
| **Database Storage** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Turbo Mode** | ❌ | ✅ Native | ✅ Delegates | ❌ | ❌ | ❌ |

**Legend**: ✅ Full support | ⚠️ Partial/Basic | ❌ Not supported | Copy = Duplicated code

### Key Observations from Matrix

1. **Core Analysis is Duplicated**: Sigma, linearity, resistance analysis exists in both LaserTrim and Fast
2. **Wrappers are Thin**: LargeScale, Cached*, Secure all delegate to LaserTrim/Fast
3. **Memory Management Scattered**: Different approaches in Fast (psutil) vs LargeScale (GC/chunking)
4. **Caching Separate**: Cached* processors could be a strategy/layer instead of separate classes
5. **Security is a Wrapper**: SecureFileProcessor just validates then delegates

### Common Code Patterns (Extract to Base)

These patterns appear in multiple processors and should be unified:

```python
# 1. File validation (appears in all processors)
def _validate_file(self, file_path: Path) -> ValidationResult:
    # Check exists, extension, size, permissions

# 2. Analysis orchestration (LaserTrim, Fast)
def _run_analyses(self, data: TrackData) -> Dict[str, Any]:
    sigma = self._analyze_sigma(data)
    linearity = self._analyze_linearity(data)
    resistance = self._analyze_resistance(data)

# 3. Status determination (LaserTrim, Fast)
def _determine_status(self, analyses: Dict) -> str:
    # Aggregate pass/fail from all analyses

# 4. Database save (LaserTrim, Fast, LargeScale)
async def _save_results(self, result: AnalysisResult) -> None:
    # Store in database
```

---

**Document Status**: Updated - Day 4 Complete
**Last Updated**: 2025-12-04
**Author**: Claude Code (Refactoring Agent)
