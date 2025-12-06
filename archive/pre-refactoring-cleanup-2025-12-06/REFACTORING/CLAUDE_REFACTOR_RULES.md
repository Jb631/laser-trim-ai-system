# Claude Code Refactoring Rules
**STRICT ENFORCEMENT - These override normal development rules during refactoring**

## Integration with CLAUDE.md
- All rules from `/CLAUDE.md` STILL APPLY
- These refactoring rules ADD additional constraints
- In case of conflict, refactoring rules take precedence during refactoring phases

---

## Refactoring-Specific Commandments

### 1. READ-ONLY FIRST
- **ALWAYS** read all related files before making changes
- **NEVER** assume you know the code without reading it
- **VERIFY** dependencies by searching imports before modifying
- Use parallel Read/Grep calls to investigate efficiently

**Example:**
```python
# WRONG: Assume and edit
Edit file without reading it first

# CORRECT: Read, then edit
Read file completely → Understand context → Make targeted changes
```

### 2. ONE CHANGE PER COMMIT
- Each commit changes ONE thing (one file type, one feature, one refactor)
- Commit format: `[PHASE-X.Y] CATEGORY: Description`
- Categories: **CLEANUP**, **PERF**, **ML**, **SPLIT**, **GUI**, **TEST**, **DOC**, **ARCH**

**Commit Message Format:**
```
[PHASE-X.Y] CATEGORY: Brief description

Detailed description of changes made.

Files changed:
- path/to/file.py: What changed (before: X lines → after: Y lines)
- path/to/other.py: What changed

Performance impact: (if applicable)
- Before: X ms/seconds/files
- After: Y ms/seconds/files
- Improvement: +Z%

Tests: (if applicable)
- Added: X tests
- Modified: Y tests
- All passing: ✅ XX/XX

Related:
- REFACTORING/DECISIONS.md: ADR-00X
- REFACTORING/PHASES/phase-X/CHECKLIST.md: Task X.Y

Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Examples:**
```bash
# Good commits
[PHASE-1.2] CLEANUP: Remove AnalyticsEngine dead code (1,052 lines)
[PHASE-2.3] ARCH: Implement UnifiedProcessor with strategy pattern
[PHASE-3.2] ML: Wire FailurePredictor to processing pipeline
[PHASE-4.1] SPLIT: Extract chart_widget.py into modules (-3,374 lines)

# Bad commits (too broad)
[PHASE-1] Cleanup and refactoring  # What cleanup? What refactoring?
Fixed stuff  # No context
Various improvements  # Too vague
```

### 3. TEST BEFORE AND AFTER
- **BEFORE**: Run `pytest tests/` and record results in session log
- **AFTER**: Run `pytest tests/` and verify same or better results
- If tests fail, **DO NOT COMMIT** - fix first or rollback
- Add regression tests for refactored code

**Testing Protocol:**
```bash
# Before making changes
pytest tests/ -v > test_before.txt
# Record: XX/YY passing

# Make changes

# After changes
pytest tests/ -v > test_after.txt
# Compare: Must be XX/YY or better

# If new tests added
pytest tests/test_new_feature.py -v
# Record: All new tests passing
```

### 4. MEASURE EVERYTHING
- **Performance changes**: Include benchmark in commit message
- **Code reduction**: Include line count diff in commit message
- **Memory changes**: Include memory usage diff if significant

**Measurement Format:**
```
Performance:
- Before: 1000 files in 45 minutes (2.7 seconds/file)
- After: 1000 files in 12 minutes (0.72 seconds/file)
- Improvement: +73% faster

Code Size:
- Before: 4,266 lines
- After: 892 lines
- Reduction: -3,374 lines (-79%)

Memory:
- Before: 2.5 GB peak
- After: 1.1 GB peak
- Reduction: -56%
```

### 5. DOCUMENT AS YOU GO
- Update `REFACTORING/PROGRESS.md` after each significant task
- Update `REFACTORING/PHASES/phase-X/PROGRESS.md` daily (minimum)
- Update `CHANGELOG.md` for user-facing changes only
- Update `REFACTORING/DECISIONS.md` for architectural changes

**Documentation Checklist (after each task):**
- [ ] Session log updated with task completion
- [ ] Phase PROGRESS.md updated
- [ ] Overall PROGRESS.md updated (if phase progress changed)
- [ ] CHANGELOG.md updated (if user-facing)
- [ ] DECISIONS.md updated (if architectural decision)

### 6. BACKWARD COMPATIBILITY
- **NEVER** break existing APIs during refactoring
- Use deprecation warnings for 1 version before removal
- Maintain old code paths with feature flags during transition
- Document migration path for deprecated features

**Feature Flag Pattern:**
```python
# Example: Unified processor with feature flag
if config.refactoring.use_unified_processor:
    # New unified processor (Phase 2)
    processor = UnifiedProcessor(config, strategy='auto')
else:
    # Old LaserTrimProcessor (deprecated, will remove in v3.1)
    logger.warning("Using deprecated LaserTrimProcessor. "
                   "Enable use_unified_processor for better performance.")
    processor = LaserTrimProcessor(config)
```

**Deprecation Warning Pattern:**
```python
import warnings

def old_function():
    warnings.warn(
        "old_function() is deprecated and will be removed in v3.1. "
        "Use new_unified_function() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Old implementation still works
    return old_implementation()
```

### 7. FEATURE FLAGS FOR MAJOR CHANGES
- New architecture changes hidden behind feature flags
- Default: **OFF** (safe, old behavior)
- Enable after validation: **ON**
- Remove flag after 1 stable release

**Feature Flag Lifecycle:**
```
Phase 2, Day 3: Add UnifiedProcessor with flag=OFF (default)
Phase 2, Day 4: Test UnifiedProcessor, validate
Phase 2, Day 5: Enable flag=ON, deprecate old processor
Phase 6, Day 5: Release v3.0 with flag=ON
v3.1 (future): Remove flag, delete old processor code
```

**Config Example:**
```yaml
# config/development.yaml
refactoring:
  use_unified_processor: true      # Phase 2+
  use_ml_failure_predictor: true   # Phase 3+
  use_split_chart_modules: true    # Phase 4+
```

### 8. NO SCOPE CREEP
- **ONLY** work on current phase tasks from checklist
- New ideas go to `REFACTORING/IDEAS.md` (review after phase complete)
- Do NOT add new features during refactoring
- Do NOT "fix while you're there" - log it, fix in dedicated commit

**Handling Scope Creep:**
```markdown
# REFACTORING/IDEAS.md

## Ideas Discovered During Refactoring

### Phase 1 - Foundation
- [ ] Idea: Add database connection pooling (discovered during baseline)
  - Benefit: Potential 10% performance gain
  - Effort: 2 hours
  - Decision: Add to Phase 6 performance optimization

- [ ] Idea: Refactor Excel export to use streaming (discovered in file reading)
  - Benefit: Lower memory usage for large exports
  - Effort: 4 hours
  - Decision: Post-refactoring backlog item
```

### 9. PARALLEL TOOL USAGE
- Use multiple Read/Grep calls in parallel when investigating
- Stage all changes at once before commit (`git add -A`)
- Never run sequential bash commands if they can be parallel
- Batch independent operations

**Example:**
```python
# GOOD: Parallel reads
Read file1, Read file2, Read file3 (all at once)

# BAD: Sequential reads
Read file1 → wait → Read file2 → wait → Read file3

# GOOD: Parallel search
Grep pattern1, Grep pattern2, Grep pattern3 (all at once)

# BAD: Sequential search
Grep pattern1 → wait → Grep pattern2 → wait → Grep pattern3
```

### 10. SESSION DISCIPLINE
- **ALWAYS** follow SESSION_PROTOCOL.md at start/end
- Session start: 5-10 minutes (read context, plan work)
- Session end: 5-10 minutes (commit, update trackers, summarize)
- Log all work in session file throughout session
- 30-minute progress checks during long sessions

**Session Boundaries:**
```
START SESSION
├─ Copy template to new session log
├─ Review: PROGRESS.md, CHECKLIST.md, last session
├─ Verify: git status clean, tests passing
├─ Plan: Set session goals
└─ BEGIN WORK

DURING SESSION (every 30 min)
├─ Check: Still on task?
├─ Check: Any blockers?
└─ Update: Session log with progress

END SESSION
├─ Save: Commit or stash all work
├─ Update: All progress trackers
├─ Test: Run test suite
├─ Measure: Benchmarks if applicable
├─ Summarize: Session log completion
└─ Cleanup: Push commits, close files
```

---

## Error Recovery

If something breaks during refactoring:

1. **STOP IMMEDIATELY** - Do not continue making changes
2. **ASSESS**: Run tests to understand scope of breakage
   ```bash
   pytest tests/ -v
   git diff  # See what changed
   ```
3. **ROLLBACK** (if needed):
   ```bash
   git reset --hard HEAD~1  # Undo last commit
   # OR
   git stash  # Save work-in-progress
   git reset --hard HEAD  # Reset to clean state
   ```
4. **LOG**: Document in `REFACTORING/RISKS.md`
   ```markdown
   ### Issue: Test failures after processor change
   - Date: 2025-01-26
   - Phase: 2.3
   - Severity: High
   - Impact: 15 tests failing
   - Root Cause: UnifiedProcessor missing edge case handling
   - Resolution: Rolled back, added edge case tests, re-implemented
   - Prevention: Add edge case tests BEFORE refactoring
   ```
5. **PLAN**: Create recovery strategy
6. **FIX**: Implement fix in isolated commit
7. **VERIFY**: Full test suite + manual smoke test before continuing

---

## Phase Completion Criteria

A phase is NOT complete until ALL of these are checked:

- [ ] All checklist tasks completed (100%)
- [ ] All tests passing (no regressions)
- [ ] Performance equal or better than baseline
- [ ] All tracking docs updated:
  - [ ] REFACTORING/PROGRESS.md
  - [ ] REFACTORING/PHASES/phase-X/PROGRESS.md
  - [ ] REFACTORING/PHASES/phase-X/RESULTS.md
  - [ ] CHANGELOG.md (if user-facing changes)
- [ ] Code review checkpoint passed (self-review minimum)
- [ ] Final session log for phase completed
- [ ] All work committed and pushed
- [ ] Phase marked complete in PROGRESS.md
- [ ] Next phase prerequisites documented

---

## Measurement Standards

### Performance Benchmarks
Always use consistent test data:
```bash
# Standard benchmark sizes
pytest tests/test_performance.py --files=100    # Small batch
pytest tests/test_performance.py --files=500    # Medium batch
pytest tests/test_performance.py --files=1000   # Large batch
pytest tests/test_performance.py --files=5000   # Stress test (Phase 6)
```

Record in `REFACTORING/MEASUREMENTS.md`:
```markdown
### Baseline (2025-01-25)
- 100 files: 3.2 seconds (32 ms/file)
- 500 files: 18.5 seconds (37 ms/file)
- 1000 files: 45.3 seconds (45 ms/file)

### Phase 1 Complete (2025-01-XX)
- 100 files: 3.0 seconds (30 ms/file) [+6% faster]
- 500 files: 17.1 seconds (34 ms/file) [+8% faster]
- 1000 files: 42.8 seconds (43 ms/file) [+6% faster]
```

### Code Metrics
```bash
# Count lines before/after
wc -l src/laser_trim_analyzer/**/*.py

# Find largest files
find src/laser_trim_analyzer -name "*.py" -exec wc -l {} + | sort -rn | head -20
```

---

## Common Patterns

### When Splitting Large Files
1. Read entire file first
2. Identify logical modules (classes, related functions)
3. Create new module files
4. Move code to new modules (one module per commit)
5. Update imports
6. Test after each module extraction
7. Delete empty sections from original file
8. Final commit: Update all imports throughout codebase

### When Unifying Duplicate Code
1. Find all duplicate code locations (Grep)
2. Read all duplicates, understand variations
3. Design unified interface
4. Create unified implementation
5. Add tests for unified version
6. Replace one duplicate at a time (one commit each)
7. Final commit: Remove old duplicate code

### When Wiring ML Models
1. Read existing ML model code
2. Test ML model in isolation
3. Design integration point in processing pipeline
4. Add feature flag
5. Implement integration (flag=OFF)
6. Test integration
7. Enable flag (flag=ON)
8. Deprecate old hardcoded logic

---

## Emergency Procedures

### Session Interrupted
```bash
# Quick save
git add -A
git stash save "EMERGENCY WIP: [description of what you were doing]"
echo "Resume at: [file:line] - [task description]" >> REFACTORING/SESSIONS/$(date +%Y-%m-%d)_session-X.md
```

### Tests Won't Pass
```bash
# Don't force it - investigate
pytest tests/ -v --tb=short  # See short traceback
pytest tests/test_failing.py -vv  # Verbose on specific test
pytest tests/ --pdb  # Debug failing test

# If can't fix quickly
git stash  # Save changes
# Document in RISKS.md
# Plan fix for next session
```

### Performance Regression
```bash
# Identify bottleneck
python -m cProfile -o profile.stats scripts/benchmark.py
python -m pstats profile.stats
# (pstats) sort time
# (pstats) stats 20

# If significant regression (>10% slower)
git reset --hard HEAD~1  # Rollback
# Redesign approach
# Document in DECISIONS.md why previous approach failed
```

---

## Remember

- **Quality over speed** - Take time to do it right
- **Document everything** - Future you will thank you
- **Test religiously** - Catch regressions early
- **Measure constantly** - Track progress objectively
- **Communicate clearly** - Session logs tell the story

---

**These rules ensure the refactoring is systematic, traceable, and successful. Follow them strictly.**
