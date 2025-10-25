# Risk Register & Mitigation Strategies

**Purpose**: Track all known risks, issues, and mitigation strategies during refactoring

---

## Risk Categories

- 🔴 **High**: Could block progress or cause data loss
- 🟡 **Medium**: Could delay timeline or cause temporary issues
- 🟢 **Low**: Minor impact, easily resolved

---

## Active Risks

### RISK-008: System Crash at 1000 Files (CRITICAL - DISCOVERED DAY 1)
**Category**: 🔴 **HIGH - CRITICAL**
**Probability**: High (confirmed - happened twice)
**Impact**: **CRITICAL** - Full system crash, potential data loss

**Description**: Processing 1000 files causes **complete system crash** (not just app crash).
- First attempt: Exit code 139 (segfault)
- Second attempt: **Entire computer crashed**
- 100 files: Stable
- 500 files: Stable
- 1000 files: **System crash**

**Root Cause** (Suspected):
- Memory leak or excessive memory allocation
- Possible Excel library (openpyxl/xlrd) memory issue
- No chunking or memory management for large batches
- Database operations accumulating in memory

**Mitigation Strategies**:
1. 🔴 **URGENT**: Document as critical baseline finding
2. ⏸️ Phase 1: Implement incremental processing (avoid 1000-file batch)
3. ⏸️ Phase 2: Add memory management to UnifiedProcessor
4. ⏸️ Phase 4: Use streaming Excel readers (not load entire file)
5. ⏸️ Phase 6: Fix root cause - comprehensive memory profiling
6. ⏸️ Phase 6: Add batch chunking (process 100-200 at a time with memory cleanup)

**Monitoring**:
- Phase 6: Re-run 1000-file benchmark to verify fix
- Memory usage tracking during large batches
- System stability monitoring

**Recovery Plan**:
- **Immediate**: Avoid >500 file batches until Phase 6
- **User workaround**: Process files in smaller batches (≤500)
- **Phase 6**: Implement robust large-batch handling

**Status**: 🔴 **CRITICAL ACTIVE** - Blocks reliable 1000+ file processing

**Target Resolution**: Phase 6 (Performance & Stability) - Week 5-6

---

### RISK-001: Breaking Existing Functionality
**Category**: 🔴 High
**Probability**: Medium
**Impact**: High - Users cannot process files

**Description**: Aggressive refactoring could break existing workflows without realizing it

**Mitigation Strategies**:
1. ✅ Run full test suite before/after every change
2. ✅ Use feature flags for major changes (can disable if broken)
3. ✅ Maintain backward compatibility (deprecated but working)
4. ✅ Manual smoke testing after each phase
5. ⏸️ Create integration test suite (Phase 6)

**Monitoring**:
- Test pass rate (must stay 100%)
- User reports (if deployed)

**Recovery Plan**:
- Feature flag OFF → revert to old behavior
- Git rollback to last known good commit
- Stash changes and fix in isolated branch

**Status**: Active (mitigations in place)

---

### RISK-002: Performance Regression on Large Batches
**Category**: 🟡 Medium
**Probability**: Medium
**Impact**: Medium - Processing slower than before

**Description**: Unified processor or ML integration could be slower than current optimized paths

**Mitigation Strategies**:
1. ✅ Benchmark after every performance-related change
2. ✅ Set performance targets (1000 files <5 min)
3. ⏸️ Profile before/after major changes
4. ⏸️ Add performance tests to test suite

**Monitoring**:
- MEASUREMENTS.md tracking
- Benchmark script results

**Recovery Plan**:
- Identify bottleneck via profiling
- Optimize critical path
- If unfixable: revert change, try different approach

**Status**: Active (benchmarking in place)

---

### RISK-003: Lost Work Due to Session Interruption
**Category**: 🟢 Low
**Probability**: Medium
**Impact**: Low - Lost time, not lost code (git stash available)

**Description**: Session interrupted before commit, work not saved properly

**Mitigation Strategies**:
1. ✅ SESSION_PROTOCOL.md emergency end procedure
2. ✅ Git stash with clear messages
3. ✅ Session logs document where to resume
4. Commit frequently (every complete subtask)

**Monitoring**:
- Session logs completeness
- Git stash list

**Recovery Plan**:
- Check git stash list
- Review last session log
- Decide: continue or rollback stash

**Status**: Active (protocol in place)

---

### RISK-004: Scope Creep - Adding Features During Refactoring
**Category**: 🟡 Medium
**Probability**: High
**Impact**: Medium - Timeline delay, increased complexity

**Description**: Temptation to add new features or "fix while you're there" during refactoring

**Mitigation Strategies**:
1. ✅ CLAUDE_REFACTOR_RULES.md Rule 8: No Scope Creep
2. ✅ REFACTORING/IDEAS.md for future improvements
3. ✅ Strict adherence to phase checklists
4. ⏸️ Code review to catch scope creep

**Monitoring**:
- Commit messages (should match phase tasks)
- Session logs (should align with checklist)

**Recovery Plan**:
- Identify non-refactoring work
- Move to IDEAS.md or separate branch
- Refocus on phase checklist

**Status**: Active (rules in place)

---

### RISK-005: Database Migration Issues
**Category**: 🔴 High
**Probability**: Low
**Impact**: High - Could corrupt production database

**Description**: Adding `processed_files` table or other schema changes could fail

**Mitigation Strategies**:
1. ✅ Test migrations on development database first
2. ✅ Backup database before migration
3. ⏸️ Create rollback migration script
4. ⏸️ Test with production data copy

**Monitoring**:
- Migration script success logs
- Database integrity checks

**Recovery Plan**:
- Run rollback migration
- Restore from backup
- Fix migration script and retry

**Status**: Active (will be critical in Phase 1, Day 2)

---

### RISK-006: ML Model Training Data Insufficient
**Category**: 🟡 Medium
**Probability**: Medium
**Impact**: Medium - ML models less accurate than formulas

**Description**: Not enough historical data to train accurate ML models

**Mitigation Strategies**:
1. ✅ Formula fallback always available
2. ⏸️ Check data quantity before training (min 500 samples)
3. ⏸️ Validate ML accuracy vs formulas
4. ⏸️ Document minimum data requirements

**Monitoring**:
- ML model performance metrics
- Comparison to formula-based predictions

**Recovery Plan**:
- Continue using formula-based predictions
- Collect more data
- Retrain models when sufficient data available

**Status**: Active (will be critical in Phase 3)

---

### RISK-007: Test Suite Becomes Too Slow
**Category**: 🟢 Low
**Probability**: Medium
**Impact**: Low - Slower development iteration

**Description**: Adding many regression tests could make test suite slow

**Mitigation Strategies**:
1. ✅ Measure test suite execution time
2. ⏸️ Use pytest markers for slow tests (run separately)
3. ⏸️ Optimize slow tests
4. ⏸️ Parallel test execution

**Monitoring**:
- pytest --durations=10 output
- MEASUREMENTS.md test suite timing

**Recovery Plan**:
- Identify slow tests
- Optimize or mark as slow
- Run slow tests separately in CI

**Status**: Active (monitoring in place)

---

## Resolved Risks

(Risks that were identified but have been resolved)

**Example Format:**
```markdown
### RISK-XXX: [Risk Title]
**Status**: ✅ Resolved
**Resolution Date**: YYYY-MM-DD
**How Resolved**: [What we did to eliminate this risk]
```

---

## Issue Log

(Track issues encountered during refactoring)

### Example Issue Template
```markdown
### ISSUE-XXX: [Issue Title]
**Date**: YYYY-MM-DD
**Phase**: X
**Severity**: High / Medium / Low
**Status**: Open / In Progress / Resolved

**Description**: [What went wrong]

**Impact**: [What's affected]

**Root Cause**: [Why it happened]

**Workaround**: [Temporary solution]

**Resolution**: [Permanent fix or "See ADR-XXX"]

**Prevention**: [How to avoid in future]
```

---

## Risk Review Schedule

**Weekly Review**: Every Friday end-of-week session
- Review all active risks
- Update mitigation status
- Add new risks discovered
- Close resolved risks

**Phase Completion Review**: End of each phase
- Comprehensive risk review
- Update for next phase
- Document lessons learned

---

## Risk Metrics

**Total Risks**: 7
**Active**: 7
**Resolved**: 0

**By Category**:
- 🔴 High: 2
- 🟡 Medium: 4
- 🟢 Low: 1

**Trend**: N/A (just started)

---

**Last Updated**: 2025-01-25 (Initial Risks Identified)
**Next Review**: End of Phase 1 (or sooner if major issue discovered)
