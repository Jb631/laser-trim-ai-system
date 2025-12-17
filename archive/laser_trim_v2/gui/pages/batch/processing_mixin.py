"""
Batch Processing Mixin

Contains the core batch processing logic extracted from BatchProcessingPage.
This includes file processing, memory management, turbo mode, and cancellation handling.
"""

import gc
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from laser_trim_analyzer.core.exceptions import ProcessingError, ValidationError
from laser_trim_analyzer.core.models import AnalysisResult
from laser_trim_analyzer.utils.file_utils import ensure_directory

logger = logging.getLogger(__name__)


class ProcessingMixin:
    """Mixin containing batch processing logic for BatchProcessingPage.

    Required attributes from main class:
    - analyzer_config: Application configuration
    - processor: File processor instance
    - _db_manager: Database manager instance
    - _state_lock: Threading lock
    - _stop_event: Threading event for stop
    - _processing_cancelled: Bool flag
    - _stop_requested: Bool flag
    - progress_dialog: Progress dialog widget
    - batch_logger: BatchProcessingLogger instance
    - resource_manager: ResourceManager instance
    - resource_optimizer: ResourceOptimizer instance
    - Various UI variables (generate_plots_var, save_to_db_var, etc.)

    Required methods from main class:
    - _safe_after(): Thread-safe UI callback
    - _is_processing_cancelled(): Check if cancelled
    - _clear_results(): Clear previous results
    - _set_controls_state(): Enable/disable controls
    - _handle_batch_success(): Handle successful completion
    - _handle_batch_error(): Handle error
    - _handle_batch_cancelled(): Handle cancellation
    - _save_batch_to_database(): Save results to DB
    """

    def _start_processing(self):
        """Start batch processing."""
        from tkinter import messagebox
        import threading

        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return

        with self._state_lock:
            if self.is_processing:
                messagebox.showwarning("Warning", "Processing already in progress")
                return

        # Check if validation was run
        if not self.validation_results:
            reply = messagebox.askyesno(
                "No Validation",
                "Batch validation hasn't been run. Proceed anyway?"
            )
            if not reply:
                return

        # Check resources before starting
        if self.resource_manager:
            status = self.resource_manager.get_current_status()
            if status.memory_critical:
                reply = messagebox.askyesno(
                    "Low Memory Warning",
                    "System memory is critically low.\n\n"
                    f"Available: {status.available_memory_mb:.0f}MB\n"
                    f"Recommended: Wait or close other applications\n\n"
                    "Continue anyway?"
                )
                if not reply:
                    return

        # Filter to only valid files if validation was run
        processable_files = []
        if self.validation_results:
            for file_path in self.selected_files:
                if self.validation_results.get(str(file_path), True):
                    processable_files.append(file_path)
        else:
            processable_files = self.selected_files.copy()

        if not processable_files:
            messagebox.showerror("Error", "No valid files to process")
            return

        # Incremental processing: Skip already-processed files (Phase 1 Day 2)
        skipped_count = 0
        if self.skip_processed_var.get() and self._db_manager:
            try:
                original_count = len(processable_files)
                processable_files = self._db_manager.get_unprocessed_files(processable_files)
                skipped_count = original_count - len(processable_files)
                if skipped_count > 0:
                    logger.info(f"Incremental mode: Skipping {skipped_count} already-processed files")
                    if len(processable_files) == 0:
                        messagebox.showinfo(
                            "All Files Already Processed",
                            f"All {original_count} selected files have already been processed.\n\n"
                            "Uncheck 'Skip Already Processed' to reprocess them."
                        )
                        return
            except Exception as e:
                logger.warning(f"Could not filter already-processed files: {e}")

        if not processable_files:
            messagebox.showerror("Error", "No valid files to process")
            return

        # Clear previous results
        self._clear_results()

        # Reset error tracking
        self._file_error_count = 0

        # Disable controls
        self._set_controls_state("disabled")

        # Import here to avoid circular imports
        from laser_trim_analyzer.gui.widgets.progress_widgets_ctk import BatchProgressDialog

        # Show progress dialog with skipped count info
        title = "Batch Processing"
        if skipped_count > 0:
            title = f"Batch Processing ({skipped_count} skipped)"
        self.progress_dialog = BatchProgressDialog(
            self,
            title=title,
            total_files=len(processable_files)
        )
        self.progress_dialog.show()

        # Start processing in thread (thread-safe)
        with self._state_lock:
            self.is_processing = True

        # Track processing start time for summary
        self.processing_start_time = time.time()
        self._skipped_processed_count = skipped_count

        self.processing_thread = threading.Thread(
            target=self._run_batch_processing,
            args=(processable_files,),
            daemon=True
        )
        self.processing_thread.start()

        if skipped_count > 0:
            logger.info(f"Started batch processing of {len(processable_files)} files ({skipped_count} already-processed files skipped)")
        else:
            logger.info(f"Started batch processing of {len(processable_files)} files")

    def _run_batch_processing(self, file_paths: List[Path]):
        """Run batch processing in background thread with performance optimizations and stop handling."""
        from tkinter import messagebox
        from laser_trim_analyzer.utils.batch_logging import setup_batch_logging

        try:
            # Reset stop flags (thread-safe)
            self._stop_event.clear()
            with self._state_lock:
                self._processing_cancelled = False
            self._stop_requested = False

            # Performance tracking
            start_time = time.time()
            last_gc_time = start_time
            last_progress_update = 0
            processed_count = 0

            # Create output directory if plots requested
            output_dir = None
            if self.generate_plots_var.get():
                base_dir = self.analyzer_config.data_directory if hasattr(self.analyzer_config, 'data_directory') else Path.home() / "LaserTrimResults"
                output_dir = base_dir / "batch_processing" / datetime.now().strftime("%Y%m%d_%H%M%S")
                ensure_directory(output_dir)
                self.last_output_dir = output_dir

            # Throttled progress callback to prevent UI flooding
            def progress_callback(message: str, progress: float):
                nonlocal last_progress_update, processed_count

                if self._is_processing_cancelled():
                    return False

                current_time = time.time()

                # Adaptive throttling based on batch size
                if len(file_paths) > 1000:
                    update_interval = 2.0
                elif len(file_paths) > 500:
                    update_interval = 1.0
                elif len(file_paths) > 100:
                    update_interval = 0.5
                else:
                    update_interval = 0.25

                update_every_n_files = max(10, len(file_paths) // 100)
                should_update = (current_time - last_progress_update >= update_interval) or \
                               (processed_count % update_every_n_files == 0)

                if should_update:
                    last_progress_update = current_time
                    if self.progress_dialog:
                        self._safe_after(0, lambda m=message, p=progress: self.progress_dialog.update_progress(m, p))

                    if processed_count % 50 == 0:
                        self._safe_after(0, self.update)

                    if processed_count % 100 == 0:
                        time.sleep(0.001)

                return True

            # Enhanced progress callback with memory monitoring
            def enhanced_progress_callback(message: str, progress: float):
                nonlocal processed_count, last_gc_time

                if self._is_processing_cancelled():
                    return False

                current_time = time.time()
                processed_count = int(progress * len(file_paths))

                try:
                    if not progress_callback(message, progress):
                        return False
                except Exception as e:
                    logger.debug(f"Progress callback error (continuing): {e}")

                # Adaptive memory management based on batch size
                if len(file_paths) > 1000:
                    cleanup_interval = 25
                    gc_time_interval = 20
                elif len(file_paths) > 500:
                    cleanup_interval = 35
                    gc_time_interval = 25
                else:
                    cleanup_interval = 50
                    gc_time_interval = 30

                if (processed_count % cleanup_interval == 0 and processed_count > 0) or \
                   (current_time - last_gc_time > gc_time_interval):
                    logger.debug(f"Performing memory cleanup at file {processed_count}")
                    gc.collect()
                    last_gc_time = current_time

                    import matplotlib.pyplot as plt
                    plt.close('all')

                    if hasattr(self.processor, '_file_cache') and len(self.processor._file_cache) > 50:
                        cache_keys = list(self.processor._file_cache.keys())
                        for key in cache_keys[:-25]:
                            del self.processor._file_cache[key]

                    if len(file_paths) > 500:
                        time.sleep(0.005)

                return True

            # Initialize batch logger
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = output_dir / "batch_logs" if output_dir else Path.home() / ".laser_trim_analyzer" / "batch_logs"
            self.batch_logger = setup_batch_logging(batch_id, log_dir, enable_performance=True)

            # Log batch start
            batch_config = {
                'generate_plots': self.generate_plots_var.get(),
                'save_to_db': self.save_to_db_var.get(),
                'output_directory': str(output_dir) if output_dir else 'None',
                'file_count': len(file_paths),
                'concurrent_workers': 4
            }
            self.batch_logger.log_batch_start(len(file_paths), batch_config)

            # Run batch processing with optimizations
            try:
                # Always using single worker to prevent CPU overload
                max_workers = 1

                processing_params = {
                    'generate_plots': self.generate_plots_var.get(),
                    'max_concurrent_files': max_workers,
                    'max_workers': max_workers
                }

                if self.resource_optimizer:
                    optimized_params = self.resource_optimizer.optimize_processing_params(
                        len(file_paths), processing_params
                    )
                    max_workers = optimized_params.get('max_workers', max_workers)

                    if not optimized_params.get('generate_plots') and self.generate_plots_var.get():
                        response = messagebox.askyesno(
                            "Resource Optimization",
                            "Plot generation uses significant memory for large batches.\n\n"
                            "Recommended: Disable plots to conserve memory.\n\n"
                            "Do you want to disable plot generation for this batch?"
                        )
                        if response:
                            self.generate_plots_var.set(False)

                    if optimized_params.get('resource_warnings'):
                        warnings_text = "\n".join(optimized_params['resource_warnings'])
                        self._safe_after(0, lambda: messagebox.showwarning(
                            "Resource Warnings",
                            f"Resource constraints detected:\n\n{warnings_text}\n\n"
                            "Processing will continue with optimized settings."
                        ))

                logger.info(f"Processing {len(file_paths)} files with {max_workers} workers (optimized)")
                if self.batch_logger:
                    self.batch_logger.main_logger.info(f"Processing {len(file_paths)} files with {max_workers} workers (optimized)")

                disable_plots_threshold = getattr(self.analyzer_config.processing, 'disable_plots_threshold', 200)

                if len(file_paths) > disable_plots_threshold and self.generate_plots_var.get():
                    user_response = {'value': None}

                    def ask_user():
                        response = messagebox.askyesno(
                            "Large Batch Detected",
                            f"Processing {len(file_paths)} files with plots enabled may cause performance issues.\n\n"
                            f"Plots are automatically disabled for batches over {disable_plots_threshold} files.\n\n"
                            "Do you want to disable plots for better performance?"
                        )
                        user_response['value'] = response

                    self._safe_after(0, ask_user)

                    timeout = 10
                    start_time = time.time()
                    while user_response['value'] is None and time.time() - start_time < timeout:
                        time.sleep(0.1)

                    if user_response['value'] is None or user_response['value']:
                        self.generate_plots_var.set(False)
                        logger.info(f"Disabled plot generation for large batch ({len(file_paths)} files)")

                # Check turbo mode threshold
                turbo_threshold = getattr(self.analyzer_config.processing, 'turbo_mode_threshold', 100)

                if len(file_paths) >= turbo_threshold:
                    logger.info(f"TURBO MODE ACTIVATED for {len(file_paths)} files (threshold: {turbo_threshold})")

                    if self.generate_plots_var.get():
                        self.generate_plots_var.set(False)
                        logger.info("Plots automatically disabled for turbo mode")

                    try:
                        self._safe_after(0, lambda: messagebox.showinfo(
                            "Turbo Mode",
                            "Turbo mode enabled for large batch.\n\n"
                            "For performance, ML predictions are disabled and heuristic risk estimates are used."
                        ))
                    except Exception:
                        pass

                    results = self._process_with_turbo_mode(
                        file_paths=file_paths,
                        output_dir=output_dir,
                        progress_callback=enhanced_progress_callback
                    )
                else:
                    logger.info(f"STANDARD MODE: {len(file_paths)} files below turbo threshold of {turbo_threshold}")
                    results = self._process_with_memory_management(
                        file_paths=file_paths,
                        output_dir=output_dir,
                        progress_callback=enhanced_progress_callback,
                        max_workers=max_workers
                    )

            except Exception as process_error:
                error_msg = f"Batch processing failed: {str(process_error)}"
                if "Config object has no attribute" in str(process_error):
                    error_msg = f"Configuration error: {str(process_error)}. Please check your configuration settings."
                elif "No module named" in str(process_error):
                    error_msg = f"Missing dependency: {str(process_error)}. Please ensure all required packages are installed."
                elif "Permission denied" in str(process_error):
                    error_msg = f"File access error: {str(process_error)}. Please check file permissions."
                elif "Memory" in str(process_error) or "RAM" in str(process_error):
                    error_msg = f"Memory error: {str(process_error)}. Try processing fewer files at once."
                else:
                    error_msg = f"Processing error: {str(process_error)}"

                raise ProcessingError(error_msg)

            # Check for cancellation before database save
            if self._is_processing_cancelled():
                self._safe_after(0, lambda: self._handle_batch_cancelled(results))
                return

            # Save to database if requested
            if self.save_to_db_var.get() and self._db_manager:
                self._save_batch_to_database(results)

            # Final cleanup
            gc.collect()

            # Check for cancellation one final time
            if self._is_processing_cancelled():
                self._safe_after(0, lambda: self._handle_batch_cancelled(results))
            else:
                self._safe_after(0, lambda: self._handle_batch_success(results, output_dir))

        except ValidationError as e:
            logger.error(f"Batch validation error: {e}")
            error_msg = f"Batch validation failed:\n\n{str(e)}\n\nPlease check that all selected files are valid Excel files."
            self._safe_after(0, lambda: self._handle_batch_error(error_msg))

        except ProcessingError as e:
            logger.error(f"Batch processing error: {e}")
            error_msg = str(e)
            self._safe_after(0, lambda msg=error_msg: self._handle_batch_error(msg))

        except Exception as e:
            logger.error(f"Unexpected batch error: {e}")
            logger.error(traceback.format_exc())
            error_msg = f"An unexpected error occurred during batch processing:\n\n{str(e)}\n\n"
            error_msg += "Please check the log files for more details."
            if "MemoryError" in str(type(e).__name__):
                error_msg = f"Out of memory error:\n\n{str(e)}\n\nTry processing fewer files at once or disable plot generation."
            self._safe_after(0, lambda: self._handle_batch_error(error_msg))

        finally:
            with self._state_lock:
                self.is_processing = False

            if self.batch_logger:
                try:
                    summary = self.batch_logger.finalize_batch()
                    logger.info(f"Batch processing summary saved to: {self.batch_logger.log_dir}")
                except Exception as e:
                    logger.error(f"Failed to finalize batch logger: {e}")

            gc.collect()

    def _process_with_memory_management(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path],
        progress_callback: Callable[[str, float], None],
        max_workers: int
    ) -> Dict[str, AnalysisResult]:
        """Process files with enhanced memory management and cancellation support."""
        import concurrent.futures

        results = {}
        self.failed_files = []
        processed_files = 0
        total_files = len(file_paths)

        # Determine chunk size
        if self.resource_manager:
            chunk_size = self.resource_manager.get_adaptive_batch_size(total_files, 0)
        else:
            if total_files > 2000:
                chunk_size = 100
            elif total_files > 1000:
                chunk_size = 75
            elif total_files > 500:
                chunk_size = 50
            elif total_files > 100:
                chunk_size = 25
            else:
                chunk_size = 10

        chunk_size = max(1, chunk_size)
        logger.info(f"Processing {total_files} files in chunks of {chunk_size}")

        for chunk_start in range(0, total_files, chunk_size):
            if self._is_processing_cancelled():
                logger.info(f"Processing cancelled after {processed_files}/{total_files} files")
                return results

            chunk_end = min(chunk_start + chunk_size, total_files)
            chunk_files = file_paths[chunk_start:chunk_end]

            logger.debug(f"Processing chunk {chunk_start}-{chunk_end} ({len(chunk_files)} files)")

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            try:
                future_to_file = {}
                active_futures = set()
                file_index = 0

                while file_index < len(chunk_files) or active_futures:
                    if self._is_processing_cancelled():
                        break

                    while len(active_futures) < max_workers and file_index < len(chunk_files):
                        file_path = chunk_files[file_index]
                        future = executor.submit(self._process_single_file_safe, file_path, output_dir)
                        future_to_file[future] = file_path
                        active_futures.add(future)
                        file_index += 1
                        time.sleep(0.2)

                        if file_index % 5 == 0 and self.resource_manager:
                            status = self.resource_manager.get_current_status()
                            if status.cpu_percent > 70:
                                logger.debug(f"CPU at {status.cpu_percent}%, adding extra delay")
                                time.sleep(1.0)

                    if active_futures:
                        done, pending = concurrent.futures.wait(
                            active_futures,
                            timeout=0.5,
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        for future in done:
                            active_futures.remove(future)

                            if self._is_processing_cancelled():
                                logger.info("Cancellation requested, stopping file processing")
                                for remaining_future in future_to_file:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                break

                            file_path = future_to_file[future]

                        if future.cancelled():
                            logger.debug(f"Skipping cancelled future for {file_path}")
                            continue

                        processed_files += 1
                        file_start_time = time.time() if self.batch_logger else None

                        try:
                            result = future.result()
                            if result is not None:
                                results[str(file_path)] = result
                                if self.batch_logger and file_start_time:
                                    self.batch_logger.log_file_complete(file_path, file_start_time, result)

                                if total_files > 1000:
                                    update_interval = 50
                                elif total_files > 500:
                                    update_interval = 25
                                else:
                                    update_interval = 10

                                if len(results) % update_interval == 0:
                                    partial_results = results.copy()
                                    self._safe_after(0, lambda r=partial_results: self.batch_results_widget.display_results(r))
                            else:
                                self.failed_files.append((str(file_path), "Processing returned None"))
                                if self.batch_logger and file_start_time:
                                    self.batch_logger.log_file_complete(
                                        file_path, file_start_time,
                                        error=Exception("Processing returned None")
                                    )

                        except Exception as e:
                            logger.error(f"File processing failed for {file_path}: {e}")
                            self.failed_files.append((str(file_path), str(e)))
                            if self.batch_logger:
                                self.batch_logger.log_file_complete(file_path, file_start_time or time.time(), error=e)

                        progress_fraction = min(processed_files / total_files, 1.0)
                        message = f"Processing {file_path.name} ({processed_files}/{total_files})"

                        if self.batch_logger and processed_files % 10 == 0:
                            self.batch_logger.log_batch_progress()

                        if not progress_callback(message, progress_fraction):
                            logger.info("Progress callback signaled to stop processing")
                            break

                        time.sleep(0.1)

                        if processed_files % 3 == 0 and self.resource_manager:
                            status = self.resource_manager.get_current_status()
                            if status.cpu_percent > 60:
                                logger.debug(f"CPU at {status.cpu_percent}% after file {processed_files}, pausing")
                                time.sleep(0.5)

            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout processing files in chunk {chunk_start}-{chunk_end}")
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
            finally:
                executor.shutdown(wait=False)
                if self._is_processing_cancelled():
                    logger.info("Executor shut down due to cancellation")

            # Memory cleanup after each chunk
            if self.resource_manager:
                if self.resource_manager.should_pause_processing():
                    logger.info("Pausing for resource recovery...")

                    if progress_callback:
                        progress_fraction = min(processed_files / total_files, 1.0)
                        progress_callback("Pausing for memory recovery...", progress_fraction)

                    wait_start = time.time()
                    max_wait = 5.0

                    while time.time() - wait_start < max_wait:
                        if self._is_processing_cancelled():
                            logger.info("Cancellation detected during resource wait")
                            break

                        status = self.resource_manager.get_current_status()
                        if not status.memory_critical and status.available_memory_mb > self.resource_manager.MIN_FREE_MEMORY_MB:
                            logger.info("Resources available, resuming processing")
                            break

                        self.resource_manager.force_cleanup()
                        time.sleep(0.1)

                    if time.time() - wait_start >= max_wait:
                        logger.warning("Resource recovery timeout, continuing anyway")

                if chunk_end < total_files:
                    next_chunk_size = self.resource_manager.get_adaptive_batch_size(total_files, chunk_end)
                    next_chunk_size = max(1, next_chunk_size)
                    logger.debug(f"Next chunk size: {next_chunk_size}")

            gc.collect()

            if self.batch_logger:
                self.batch_logger.log_memory_cleanup(force_gc=True)

            # Delay between chunks
            if chunk_size > 50:
                time.sleep(2.0)
            elif chunk_size > 25:
                time.sleep(1.5)
            else:
                time.sleep(1.0)

            if self.resource_manager:
                status = self.resource_manager.get_current_status()
                if status.cpu_high:
                    logger.info(f"High CPU detected ({status.cpu_percent}%), pausing for recovery")
                    time.sleep(5.0)
                elif status.cpu_percent > 50:
                    logger.info(f"Moderate CPU usage ({status.cpu_percent}%), adding delay")
                    time.sleep(3.0)

        if self.failed_files:
            logger.warning(f"Processing completed with {len(self.failed_files)} failures")
            from tkinter import messagebox
            if len(self.failed_files) > 0:
                failure_summary = "The following files failed to process:\n\n"
                for file_path, error in self.failed_files[:10]:
                    file_name = Path(file_path).name
                    failure_summary += f"• {file_name}: {error}\n"

                if len(self.failed_files) > 10:
                    failure_summary += f"\n... and {len(self.failed_files) - 10} more files"

                self._safe_after(0, lambda: messagebox.showwarning(
                    "Processing Failures",
                    failure_summary + "\n\nCheck the log files for complete details."
                ))

        return results

    def _process_with_turbo_mode(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path],
        progress_callback: Callable[[str, float], None]
    ) -> Dict[str, AnalysisResult]:
        """Process files using turbo mode with FastProcessor directly."""
        logger.info(f"TURBO MODE: Initializing processing for {len(file_paths)} files")

        from laser_trim_analyzer.core.fast_processor import FastProcessor

        fast_processor = FastProcessor(self.analyzer_config, turbo_mode=True)
        logger.info(f"FastProcessor created with turbo_mode={fast_processor.turbo_mode}")

        self.stats = {
            'total_files': len(file_paths),
            'processed_files': 0,
            'failed_files': 0,
            'start_time': time.time()
        }

        def turbo_progress_callback(message: str, progress: float):
            """Progress callback for turbo mode."""
            if self._is_processing_cancelled():
                logger.info("Turbo mode processing cancelled by user")
                return False

            self.stats['processed_files'] = int(progress * len(file_paths))

            def update_ui():
                progress_callback(message, progress)

            try:
                self._safe_after(0, update_ui)
            except Exception as e:
                logger.debug(f"Could not schedule UI update: {e}")

            return True

        try:
            results_list = fast_processor.process_batch_fast(
                file_paths,
                output_dir,
                turbo_progress_callback
            )

            logger.info(f"TURBO MODE completed: {len(results_list)} files processed")

            results = {}
            for result in results_list:
                if result and hasattr(result, 'metadata') and hasattr(result.metadata, 'file_path'):
                    results[str(result.metadata.file_path)] = result

            # Save to database if enabled
            if self.save_to_db_var.get() and self._db_manager and results:
                try:
                    batch_size = 100
                    all_results = list(results.values())
                    for i in range(0, len(all_results), batch_size):
                        batch = all_results[i:i + batch_size]
                        if hasattr(self._db_manager, 'save_analysis_batch'):
                            force_overwrite = self.force_reprocess_var.get()
                            self._db_manager.save_analysis_batch(batch, force_overwrite=force_overwrite)
                        else:
                            for result in batch:
                                self._db_manager.save_analysis_result(result)
                    logger.info(f"Saved {len(results)} results to database")
                except Exception as e:
                    logger.error(f"Failed to save turbo mode results to database: {e}")

            return results

        except Exception as e:
            logger.error(f"Turbo mode processing failed: {e}")
            raise ProcessingError(f"Turbo mode failed: {e}")

    def _process_single_file_safe(self, file_path: Path, output_dir: Optional[Path]) -> Optional[AnalysisResult]:
        """Safely process a single file with error handling."""
        from tkinter import messagebox

        try:
            if self._is_processing_cancelled():
                return None

            processor_config = self.analyzer_config.copy() if hasattr(self.analyzer_config, 'copy') else self.analyzer_config

            if hasattr(processor_config, 'processing'):
                processor_config.processing.generate_plots = self.generate_plots_var.get()
                if hasattr(processor_config.processing, 'comprehensive_validation'):
                    processor_config.processing.comprehensive_validation = self.comprehensive_validation_var.get()

            result = self.processor.process_file_sync(
                file_path,
                output_dir=output_dir
            )

            if self._is_processing_cancelled():
                return None

            return result

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            error_msg = f"Failed to process {file_path.name}:\n\n{str(e)}"
            if "Permission denied" in str(e):
                error_msg += "\n\nPlease ensure the file is not open in another program."
            elif "No such file" in str(e):
                error_msg += "\n\nThe file may have been moved or deleted."
            elif "Invalid file format" in str(e):
                error_msg += "\n\nPlease ensure this is a valid Excel file."

            if not hasattr(self, '_file_error_count'):
                self._file_error_count = 0

            self._file_error_count += 1
            if self._file_error_count <= 3:
                self._safe_after(0, lambda: messagebox.showerror("File Processing Error", error_msg))
            elif self._file_error_count == 4:
                self._safe_after(0, lambda: messagebox.showwarning(
                    "Multiple File Errors",
                    "Multiple files have failed to process.\n\n"
                    "Further individual error dialogs will be suppressed.\n"
                    "Check the final summary for all errors."
                ))
            return None

    def _handle_batch_cancelled(self, partial_results: Dict[str, AnalysisResult]):
        """Handle cancelled batch processing."""
        from tkinter import messagebox

        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None

        self.batch_results = partial_results
        self._update_batch_status("Processing Cancelled", "orange")
        self._set_controls_state("normal")
        self.start_button.configure(text="Start Processing")

        if partial_results:
            self.batch_results_widget.display_results(partial_results)
            self.export_excel_button.configure(state="normal")
            self.export_html_button.configure(state="normal")
            self.export_csv_button.configure(state="normal")

            if hasattr(self, 'last_output_dir') and self.last_output_dir:
                self.output_folder_button.configure(state="normal")

        processed_count = len(partial_results)
        total_selected = len(self.selected_files)

        messagebox.showinfo(
            "Processing Cancelled",
            f"Batch processing was cancelled.\n\n"
            f"Processed: {processed_count} files\n"
            f"Remaining: {total_selected - processed_count} files\n\n"
            f"Partial results are available for export."
        )

        logger.info(f"Batch processing cancelled. Processed {processed_count} of {total_selected} files")
