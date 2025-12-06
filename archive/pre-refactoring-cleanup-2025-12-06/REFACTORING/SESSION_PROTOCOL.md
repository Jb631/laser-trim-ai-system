# Session Start/End Protocol
**MANDATORY - Execute at every session boundary**

---

## Session Start Checklist

### 1. Session Setup (2-3 minutes)

```bash
# Navigate to project
cd d:/UserFolders/Desktop/laser_trim_analyzer_v2

# Pull latest changes
git pull origin main

# Verify clean state
git status

# Create session log from template
cp REFACTORING/SESSIONS/template.md REFACTORING/SESSIONS/$(date +%Y-%m-%d)_session-N.md
# Note: Replace N with next session number
```

**Record in session log:**
- Session date/time
- Session number
- Current phase
- Planned work for this session
- Git commit hash at start: `git rev-parse HEAD`

---

### 2. Review Context (5 minutes)

Read in this order:

1. **REFACTORING/PROGRESS.md** - Where are we overall?
   - Current phase
   - Overall progress percentage
   - Recent completions

2. **REFACTORING/PHASES/phase-X/CHECKLIST.md** - What's next?
   - Find next unchecked task
   - Read task description and requirements
   - Note any dependencies

3. **REFACTORING/PHASES/phase-X/PROGRESS.md** - What was last done?
   - Review recent updates
   - Check for any issues noted
   - Understand context

4. **CHANGELOG.md** (first 20 lines) - What changed recently?
   - Any user-facing changes
   - Any breaking changes to be aware of

5. **Last session log** (`REFACTORING/SESSIONS/`) - What was in progress?
   - Read "Next Session" section
   - Check for any blockers
   - Review notes/observations

**Record in session log:**
```markdown
### Context Review
- Last Completed: [Task from previous session]
- Next Task: [What I'll work on]
- Current Branch: [git branch name]
- Starting Commit: [git commit hash]
- Tests Baseline: XX/YY passing

### Blockers/Risks Identified
- [Any issues from previous session]
- [Any new concerns]
```

---

### 3. Verify Environment (2 minutes)

```bash
# Activate virtual environment
.venv/Scripts/activate  # Windows
# OR
source .venv/bin/activate  # Linux/Mac

# Verify Python version
python --version
# Expected: Python 3.11 or later

# Verify key dependencies
pip list | grep -E "numpy|pandas|scikit-learn|pytest"

# Run quick test to ensure environment OK
pytest tests/test_calculations.py -v
# Should see: XX tests passed
```

**Record in session log:**
```markdown
### Environment Check
- Python: 3.11.x ✅
- Dependencies: All installed ✅
- Tests Baseline: 17/17 passing ✅
- Issues: None / [List any]
```

**If environment issues:**
```bash
# Reinstall dependencies
pip install -e .

# If still issues
pip install --upgrade -r requirements.txt

# Document in session log
```

---

### 4. Set Session Goals (1 minute)

**Record in session log:**
```markdown
### Session Goals
**Primary Goal**: [What MUST be completed]
- Example: "Complete Day 2 incremental processing implementation"

**Secondary Goals**: [Nice to have]
- Example: "Document incremental processing design in DECISIONS.md"

**Time Estimate**: [X hours]
- Based on task complexity and your availability

**Success Criteria**:
- [ ] [Specific measurable outcome]
- [ ] [Another measurable outcome]
```

---

### 5. Start Timer & Begin Work

- Note start time in session log
- Set reminder for progress check every 30 minutes
- Begin work on primary goal

**Optional: Use pomodoro technique**
```
Work: 25 minutes
Break: 5 minutes
Every 4 pomodoros: 15-minute break
```

---

## During Session

### 30-Minute Progress Check

Every 30 minutes, pause and ask:

1. **Am I still working on the planned task?**
   - Yes: Continue
   - No: Document why (blocker, scope change)

2. **Have I encountered any blockers?**
   - Yes: Document in session log, RISKS.md
   - No: Continue

3. **Is this taking longer than expected?**
   - Yes: Adjust estimates, re-scope if needed
   - No: Continue

4. **Should I update progress notes now?**
   - If significant milestone: Update session log
   - If small progress: Continue, update later

**Update session log:**
```markdown
### [HH:MM] Progress Check
- Task: [Current task]
- Status: On track / Behind schedule / Blocked
- Issue: [If any]
- Adjustment: [If needed]
```

---

### Logging Work As You Go

**After completing each subtask:**
```markdown
### [HH:MM] Task: [Task name]
**Status**: ✅ Complete

**Actions Taken**:
- [What I did step by step]
- [Files modified]
- [Commands run]

**Results**:
- [Outcome achieved]
- [Metrics if applicable: lines changed, tests added, performance]

**Notes**:
- [Important observations]
- [Issues encountered and resolved]
- [Ideas for future]
```

**Example:**
```markdown
### [14:30] Task: Add processed_files table to database

**Status**: ✅ Complete

**Actions Taken**:
- Created migration script: database/migrations/add_processed_files_table.py
- Added ProcessedFile model to database/models.py
- Updated DatabaseManager with get_processed_files() method
- Ran migration on dev database

**Results**:
- New table: processed_files (columns: id, filename, file_hash, processed_date)
- 0 rows initially (ready for population)
- Tests: Added test_processed_files_table.py (5 tests, all passing)

**Notes**:
- Used SHA-256 for file_hash (better than MD5)
- Added index on filename for fast lookup
- Added index on file_hash to detect duplicates
```

---

## Session End Checklist

### 1. Save All Work (2 minutes)

```bash
# Stage all changes
git add -A

# Check what's staged
git status
git diff --cached --stat

# If work is COMPLETE:
git commit -m "[PHASE-X.Y] CATEGORY: Description"
# Follow commit message format from CLAUDE_REFACTOR_RULES.md

# If work is IN PROGRESS:
git stash save "WIP: Phase X.Y - [description of what's incomplete]"
# Document where to resume in session log
```

**Record in session log:**
```markdown
### Work Saved
- Status: ✅ Complete / 🔄 In Progress / ⚠️ Blocked
- Commit: [commit hash] / Stash: [stash message]
- Resume at: [file:line or task description if in progress]
```

---

### 2. Update Progress Trackers (3 minutes)

**Update these files in order:**

#### A) Current session log (`REFACTORING/SESSIONS/YYYY-MM-DD_session-N.md`)
```markdown
## Session End

### Work Summary
**Completed Tasks**:
- [x] Task 1 - [Brief outcome]
- [x] Task 2 - [Brief outcome]

**In Progress Tasks**:
- [ ] Task 3 - [Status, where to resume]

**Blocked Tasks**:
- [ ] Task 4 - [Why blocked, what's needed]

### Next Session
**Start With**: [Specific task or file:line]
**Prerequisites**: [Anything needed]
**Estimated Time**: [X hours]
```

#### B) Phase progress (`REFACTORING/PHASES/phase-X/PROGRESS.md`)
```markdown
### 2025-01-DD - Session N
**Tasks Completed**:
- [x] Day X, Task Y: [Brief description]
- [x] Day X, Task Z: [Brief description]

**Status Updates**:
- Day 1: ✅ Complete
- Day 2: 🔄 60% (3/5 tasks done)
- Day 3: ⏸️ Not started

**Issues**: [Any blockers or problems]
**Next**: [What's coming next]
```

#### C) Phase checklist (`REFACTORING/PHASES/phase-X/CHECKLIST.md`)
```markdown
## Day 2: Incremental Processing
- [x] Add processed_files table to database
- [x] Modify batch processor to check DB
- [ ] Add --skip-existing flag to CLI  ← Next session starts here
- [ ] Add "Skip Processed" checkbox to GUI
```

#### D) Overall progress (`REFACTORING/PROGRESS.md`)
```markdown
| Phase | Name | Status | Progress | Start | End | Days |
|-------|------|--------|----------|-------|-----|------|
| 1 | Foundation | 🔄 In Progress | 40% (Day 2/5) | 2025-01-25 | TBD | 5 |

## Recent Activity
### 2025-01-DD - Session N
- Completed: Incremental processing database schema
- Progress: Phase 1 now 40% complete
- [Session log](SESSIONS/2025-01-DD_session-N.md)
```

---

### 3. Performance Measurements (if applicable)

**Only if you changed code affecting performance:**

```bash
# Run benchmark script
python scripts/benchmark_processing.py --files 100 500 1000

# Record in REFACTORING/MEASUREMENTS.md
```

**Format in MEASUREMENTS.md:**
```markdown
### Session N (2025-01-DD) - Incremental Processing Added

#### Benchmark: Skip Already Processed Files
- Scenario: 1000 files, 900 already processed, 100 new
- Before (reprocess all): 45.3 seconds
- After (skip processed): 4.8 seconds
- Improvement: **+90% faster** (9.4x speedup)

#### Benchmark: Full Processing (baseline check)
- 1000 new files
- Before: 45.3 seconds
- After: 45.1 seconds
- Impact: No regression ✅
```

**Record in session log:**
```markdown
### Performance Measurements
- Benchmarks run: ✅
- Results recorded in MEASUREMENTS.md ✅
- Performance change: [Summary]
```

---

### 4. Test Verification (2 minutes)

```bash
# Run full test suite
pytest tests/ -v

# Record results
```

**Record in session log:**
```markdown
### Test Results
- Tests run: pytest tests/ -v
- Passing: XX/YY
- New tests added: Z
- New test failures: [List if any, or "None"]
- Action needed: [If tests failing: describe fix plan]
```

**If tests failing:**
```markdown
### Test Failures
- test_file.py::test_name: [Brief description of failure]
- Root cause: [If known]
- Fix plan: [What needs to be done]
- Blocked: [Yes/No - is this blocking next session?]
```

---

### 5. Session Summary (2 minutes)

**Complete session log with:**

```markdown
## Session End

### Session Statistics
- Duration: [X hours Y minutes]
- Tasks completed: X
- Tasks in progress: Y
- Tasks blocked: Z

### Code Changes
- Lines added: +XXX
- Lines removed: -YYY
- Net change: ±ZZZ
- Files modified: N

### Metrics This Session
- Tests added: X
- Tests passing: XX/YY
- Performance change: [+X% / -X% / No change]
- Code reduction: [-X lines / +X lines / No change]

### Session Status
**Overall**: ✅ Productive / 🔄 Partial Progress / ⚠️ Encountered Issues

**Phase Progress**: X% → Y% (+Z%)

### Notes for Next Session
- [Important context to remember]
- [Any decisions made]
- [Resources needed]
```

---

### 6. Cleanup (1 minute)

```bash
# Push commits if appropriate (not WIP stashes)
git push origin [branch-name]

# Close unnecessary files/terminals
# Save editor workspace if applicable
```

**Final session log update:**
```markdown
### Session Complete
- Session log: ✅ Complete
- Progress trackers: ✅ Updated
- Tests: ✅ XX/YY passing
- Commits: ✅ Pushed / 🔄 Stashed
- Ready for next session: ✅
```

---

## Emergency Session End

**If you must end session abruptly:**

```bash
# Quick stash with context
git add -A
git stash save "EMERGENCY WIP: [what you were doing] - Resume at [file:line]"

# Quick note in session log
echo "## EMERGENCY END" >> REFACTORING/SESSIONS/$(date +%Y-%m-%d)_session-N.md
echo "Reason: [brief reason]" >> REFACTORING/SESSIONS/$(date +%Y-%m-%d)_session-N.md
echo "Resume at: [file:line or task]" >> REFACTORING/SESSIONS/$(date +%Y-%m-%d)_session-N.md
echo "Status: Work stashed, tests were [passing/failing]" >> REFACTORING/SESSIONS/$(date +%Y-%m-%d)_session-N.md
```

**On next session start:**
- Review emergency stash
- Decide: Continue or rollback?
- Update session log with resolution

---

## Helpful Commands Reference

### Git
```bash
git status                          # Check current state
git diff                            # See unstaged changes
git diff --cached                   # See staged changes
git log --oneline -10               # Recent commits
git stash list                      # List stashes
git stash show stash@{0}            # Show stash contents
git stash pop                       # Apply and remove stash
```

### Testing
```bash
pytest tests/ -v                    # All tests, verbose
pytest tests/test_file.py -v        # Specific file
pytest tests/ -k "test_name"        # Specific test by name
pytest tests/ -x                    # Stop on first failure
pytest tests/ --tb=short            # Short traceback
pytest tests/ --pdb                 # Debug on failure
```

### Performance
```bash
python -m cProfile script.py        # Profile script
time python script.py               # Simple timing
```

### Code Metrics
```bash
wc -l file.py                       # Count lines in file
find . -name "*.py" -exec wc -l {} + | sort -rn | head -20  # Largest files
git diff --stat                     # Changes summary
```

---

## Session Protocol Checklist (Quick Reference)

**START:**
- [ ] Create session log from template
- [ ] Pull latest changes
- [ ] Review: PROGRESS.md, CHECKLIST.md, last session
- [ ] Verify environment (Python, deps, tests)
- [ ] Set session goals
- [ ] Start timer

**DURING:**
- [ ] 30-minute progress checks
- [ ] Log work as you go
- [ ] Update session log with completed tasks

**END:**
- [ ] Save all work (commit or stash)
- [ ] Update all progress trackers (4 files minimum)
- [ ] Run test suite
- [ ] Run benchmarks (if performance-related)
- [ ] Complete session summary
- [ ] Cleanup and push

---

**Following this protocol ensures:**
- No lost work
- Clear context for next session
- Traceable progress
- Measurable outcomes
- Systematic refactoring

**Time investment: 10-15 minutes per session boundary**
**Payoff: Hours saved in context switching and rework**
