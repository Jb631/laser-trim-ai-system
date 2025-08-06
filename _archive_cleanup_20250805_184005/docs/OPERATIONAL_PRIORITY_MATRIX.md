# Operational Priority Matrix - Laser Trim Analyzer V2
## Single User Production Focus

**Date:** November 6, 2025  
**Objective:** Get application fully operational for reliable single-user production use  
**Timeline:** 4 weeks to operational stability

---

## Executive Summary

This matrix prioritizes fixes based on operational impact for a single-user deployment. Security and multi-user features are deprioritized in favor of core functionality and stability. The goal is a working, reliable application that can process laser trim data without crashes or data loss.

---

## Week 1: CRITICAL - Application Breaking Issues

These issues prevent basic operation and must be fixed immediately.

### 1. Memory Exhaustion Crashes (ResourceError-001) 🔴
**Impact:** Application unusable for production files  
**Symptoms:** 
- Files >50MB cause immediate crash
- Application freezes with "Not Responding"
- Forces user to kill process, risking data corruption

**Root Cause:** Entire Excel file loaded into memory at once  
**Location:** `src/laser_trim_analyzer/core/processor.py` - `_process_file_internal()`  
**Fix Strategy:**
```python
# Implement streaming Excel reader
def process_file_streaming(file_path: Path, chunk_size: int = 10000):
    for chunk in pd.read_excel(file_path, chunksize=chunk_size):
        yield process_chunk(chunk)
```
**Effort:** 16 hours  
**Success Metric:** Process 500MB files without crash

### 2. Database Corruption on Process Kill (DBError-004) 🔴
**Impact:** Total data loss requiring restart from scratch  
**Symptoms:**
- Kill process during save = corrupted database
- "database disk image is malformed" error
- All previous work lost

**Root Cause:** No Write-Ahead Logging (WAL) enabled  
**Location:** `src/laser_trim_analyzer/database/manager.py`  
**Fix Strategy:**
```python
# Enable WAL mode for crash recovery
engine = create_engine(
    f"sqlite:///{db_path}",
    connect_args={'check_same_thread': False},
    pool_pre_ping=True
)
# Execute after connection
connection.execute("PRAGMA journal_mode=WAL")
```
**Effort:** 8 hours  
**Success Metric:** Database survives process termination

### 3. Thread Deadlocks (ThreadError-001) 🔴
**Impact:** Application hangs permanently  
**Symptoms:**
- UI completely frozen
- Must force quit application
- Happens during batch operations

**Root Cause:** Poor thread synchronization, no timeout handling  
**Location:** `src/laser_trim_analyzer/gui/pages/batch_processing_page.py`  
**Fix Strategy:**
```python
# Add timeout to all locks
with self.processing_lock.acquire(timeout=30):
    # Process with automatic release
    pass
```
**Effort:** 12 hours  
**Success Metric:** No permanent hangs in 24-hour test

### 4. Model Prediction Failures (MLError-002, MLError-006) 🔴
**Impact:** Wrong analysis results or crashes  
**Symptoms:**
- Silent wrong predictions
- "Model version mismatch" errors
- Missing feature crashes

**Root Cause:** No model versioning or validation  
**Location:** `src/laser_trim_analyzer/ml/predictors.py`  
**Fix Strategy:**
```python
# Add model version checking
MODEL_VERSION = "2.0.0"
def load_model(path):
    model = joblib.load(path)
    if model.version != MODEL_VERSION:
        raise ModelVersionError(f"Expected {MODEL_VERSION}, got {model.version}")
    return model
```
**Effort:** 12 hours  
**Success Metric:** Clear errors on version mismatch, no silent failures

---

## Week 2: HIGH PRIORITY - Core Workflow Issues

These issues severely impact productivity but have workarounds.

### 5. Batch Processing Hangs (ResourceError-003, BatchError-001) 🟡
**Impact:** Cannot process multiple files efficiently  
**Symptoms:**
- UI freeze with >100 files
- Progress bar stops updating
- Memory usage grows unbounded

**Root Cause:** No resource management or batch size limits  
**Location:** `src/laser_trim_analyzer/gui/pages/batch_processing_page.py`  
**Fix Strategy:**
```python
# Adaptive batch sizing based on available memory
def calculate_batch_size(file_count: int, avg_file_size: float) -> int:
    available_memory = psutil.virtual_memory().available
    safe_memory = available_memory * 0.5  # Use only 50%
    return min(file_count, int(safe_memory / avg_file_size / 3))
```
**Effort:** 16 hours  
**Success Metric:** Process 1000 files without hanging

### 6. File Corruption in Concurrent Access (FSError-003) 🟡
**Impact:** Results files corrupted when accessed during write  
**Symptoms:**
- Excel files become unreadable
- "File format is not valid" errors
- Partial data in output files

**Root Cause:** No file locking mechanism  
**Location:** `src/laser_trim_analyzer/utils/file_utils.py`  
**Fix Strategy:**
```python
# Implement file locking
import fcntl  # Unix/Linux
import msvcrt  # Windows

def safe_file_write(path: Path, data: bytes):
    with open(path, 'wb') as f:
        # Platform-specific locking
        if sys.platform == 'win32':
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, len(data))
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        f.write(data)
```
**Effort:** 12 hours  
**Success Metric:** No corruption in concurrent access test

### 7. UI Freezing Issues (UIError-001, UIError-003) 🟡
**Impact:** Poor user experience, appears crashed  
**Symptoms:**
- Click button, no response for 30+ seconds
- Can't cancel operations
- Progress updates flood UI

**Root Cause:** Heavy operations on main thread  
**Location:** `src/laser_trim_analyzer/gui/pages/single_file_page.py`  
**Fix Strategy:**
```python
# Move processing to background thread
def analyze_file(self):
    # Disable UI
    self.set_ui_enabled(False)
    
    # Process in background
    self.worker_thread = threading.Thread(
        target=self._process_in_background,
        daemon=True
    )
    self.worker_thread.start()
```
**Effort:** 16 hours  
**Success Metric:** UI responsive during all operations

---

## Week 3-4: MEDIUM PRIORITY - Performance & Stability

These issues impact efficiency but don't prevent operation.

### 8. Database Query Performance 🟢
**Impact:** Slow file lookups and statistics  
**Symptoms:**
- 5+ seconds to load file history
- Model statistics timeout
- Batch results slow to display

**Root Cause:** Missing database indexes  
**Location:** `src/laser_trim_analyzer/database/models.py`  
**Fix Strategy:**
```sql
-- Add missing indexes
CREATE INDEX idx_file_hash ON analysis_results(file_hash);
CREATE INDEX idx_failure_prob ON track_results(failure_probability DESC);
CREATE INDEX idx_model_date ON analysis_results(model, analysis_date DESC);
```
**Effort:** 4 hours  
**Success Metric:** <1 second query response

### 9. Memory Leaks in Plot Generation (ResourceError-008) 🟢
**Impact:** Requires restart after ~8 hours use  
**Symptoms:**
- Memory usage grows continuously
- Application becomes sluggish
- Eventually crashes with OOM

**Root Cause:** Matplotlib figures not properly closed  
**Location:** `src/laser_trim_analyzer/utils/plotting_utils.py`  
**Fix Strategy:**
```python
# Ensure all plots are closed
def create_plot_safe(data):
    fig = plt.figure()
    try:
        # Create plot
        ax = fig.add_subplot(111)
        ax.plot(data)
        
        # Save or display
        return fig
    finally:
        plt.close(fig)  # Always close
        plt.clf()
        plt.cla()
        gc.collect()  # Force cleanup
```
**Effort:** 8 hours  
**Success Metric:** Stable memory after 24-hour run

### 10. Pickle Serialization (APIError-003) 🟢
**Impact:** Security risk IF using API features  
**Symptoms:**
- Potential remote code execution
- Only affects AI analysis features

**Root Cause:** Using pickle for API response caching  
**Location:** `src/laser_trim_analyzer/api/client.py`  
**Fix Strategy:**
```python
# Replace pickle with JSON
def cache_response(key: str, response: dict):
    cache_data = {
        'timestamp': time.time(),
        'data': response  # Must be JSON-serializable
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
```
**Effort:** 6 hours  
**Success Metric:** No pickle usage in codebase

---

## Implementation Strategy

### Week 1 Focus: Core Stability
1. **Monday-Tuesday:** Memory exhaustion fix (streaming)
2. **Wednesday:** Database WAL mode
3. **Thursday:** Thread deadlock fixes
4. **Friday:** Model validation

**Testing:** Each fix includes 4-hour stress test

### Week 2 Focus: Workflow Reliability  
1. **Monday-Tuesday:** Batch processing stability
2. **Wednesday-Thursday:** File locking implementation
3. **Friday:** UI responsiveness fixes

**Testing:** Full workflow testing with production data

### Week 3-4 Focus: Polish & Performance
1. **Week 3:** Database optimization, memory leak fixes
2. **Week 4:** Final testing, pickle removal if needed

**Testing:** 48-hour continuous operation test

---

## Success Metrics

### Week 1 Complete When:
- ✅ Can process 500MB Excel files
- ✅ Database survives forced shutdown
- ✅ No application hangs
- ✅ ML predictions work correctly

### Week 2 Complete When:
- ✅ Can process 1000-file batches
- ✅ No file corruption issues
- ✅ UI remains responsive

### Week 3-4 Complete When:
- ✅ All queries < 1 second
- ✅ 48-hour uptime achieved
- ✅ Memory usage stable

---

## Risk Mitigation

### Backup Strategy
- Daily database backups before fixes
- Keep previous version available
- Document all configuration changes

### Testing Protocol
1. Unit test for each fix
2. Integration test with real data
3. 4-hour stress test minimum
4. User acceptance testing

### Rollback Plan
- Each fix in separate commit
- Feature flags where possible
- Previous version on standby

---

## Post-Implementation

### After Week 4:
1. **Documentation Update**
   - Update user manual with new limits
   - Document known working configurations
   - Create troubleshooting guide

2. **Monitoring Setup**
   - Memory usage alerts
   - Database size monitoring
   - Performance baselines

3. **Future Considerations**
   - Plan for multi-user support
   - Security hardening roadmap
   - Cloud deployment options

---

## Conclusion

This prioritized approach focuses on making the Laser Trim Analyzer v2 a reliable single-user production tool. Security and multi-user features are intentionally deferred to achieve operational stability first. The 4-week timeline provides a working application that can handle real production workloads without crashes or data loss.

**Key Message:** Stability first, features later. A working tool is better than a feature-rich but unreliable one.