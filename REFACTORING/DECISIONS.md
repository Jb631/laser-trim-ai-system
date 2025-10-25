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

**Date**: TBD (Phase 2)
**Status**: Proposed
**Context**: 6 different processor classes with 40-60% duplicated logic. Confusing which to use when.
**Decision**: Create UnifiedProcessor with pluggable strategies for different scenarios.

**Processor Analysis**:
1. `LaserTrimProcessor` (2,682 lines) - Main, standard processing
2. `FastProcessor` (1,499 lines) - Turbo mode (parallel processing)
3. `LargeScaleProcessor` (1,189 lines) - Large batches (chunking)
4. `CachedFileProcessor` (383 lines) - File-level caching
5. `CachedBatchProcessor` (383 lines) - Batch-level caching
6. `SecureFileProcessor` (unknown) - Security validation

**Proposed Design**:
```python
class UnifiedProcessor:
    def __init__(self, config, strategy='auto'):
        self.strategy = self._select_strategy(strategy, config)

    def process(self, files):
        return self.strategy.process(files)

class StandardStrategy:  # Default
class TurboStrategy:     # Parallel processing
class LargeScaleStrategy:  # Chunking for 1000+ files
class CachedStrategy:    # Caching layer
```

**Consequences**:
- **Positive**: Single entry point (clear API)
- **Positive**: -40% code duplication
- **Positive**: Easier testing (test strategies independently)
- **Positive**: Composable (can combine strategies)
- **Negative**: Requires refactor of all callers
- **Mitigation**: Feature flag for gradual rollout

**Related**: Phase 2

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
| ADR-004 | Unify Processors (Strategy Pattern) | Proposed | 2 | TBD |
| ADR-005 | Wire ML Models to Pipeline | Proposed | 3 | TBD |
| ADR-006 | Split Large Files by Modules | Proposed | 4 | TBD |

---

**Last Updated**: 2025-01-25 (Initial ADRs)
**Next Review**: After each phase completion
