"""
AI Insights Page for Laser Trim Analyzer

Provides interface for AI-powered analysis, chat assistant,
and automated report generation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from datetime import datetime
import threading
import json
from typing import Optional, Dict, List, Any

from laser_trim_analyzer.core.models import AnalysisResult
# from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage  # Using CTkFrame instead
import logging
import os
import sys
# Removed alert_banner import to prevent glitching issues
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.api.client import QAAIAnalyzer, AIProvider
# from laser_trim_analyzer.gui.widgets import add_mousewheel_support  # Not used


class AIInsightsPage(ctk.CTkFrame):
    """AI-powered insights and analysis page."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add BasePage-like functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        
        # Page-specific attributes
        self.ai_client = None
        self.chat_history = []
        self.chat_sessions = []  # Store all chat sessions
        self.current_session_id = None
        self.current_analysis = None
        
        # Thread safety
        self._ai_lock = threading.Lock()
        self._notification_timer = None
        
        # Create the page
        self._create_page()
        
        # Load chat history
        self._load_chat_sessions()
        
        # Initialize AI client in background to prevent blocking
        thread = threading.Thread(target=self._initialize_ai_client, daemon=True)
        thread.start()

    def _create_page(self):
        """Create AI insights page content with consistent theme (matching batch processing)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create simple alert frame at the top of main container to prevent glitching
        self.alert_frame = ctk.CTkFrame(self.main_container)
        self.alert_frame.pack(fill='x', pady=(0, 10))
        self.alert_message = ctk.CTkLabel(
            self.alert_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.alert_message.pack(padx=10, pady=5)
        self.alert_frame.pack_forget()  # Initially hidden
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_insights_section()
        self._create_chat_section()
        self._create_report_section()
        
        # Apply hover fixes after creation
        self.after(100, self._apply_hover_fixes)

    def _create_header(self):
        """Create header section (matching batch processing theme)."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="AI-Powered Insights",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)

    def _create_insights_section(self):
        """Create automatic insights section (matching batch processing theme)."""
        self.insights_frame = ctk.CTkFrame(self.main_container)
        self.insights_frame.pack(fill='x', pady=(0, 20))

        self.insights_label = ctk.CTkLabel(
            self.insights_frame,
            text="Automatic Insights:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.insights_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Insights container
        self.insights_container = ctk.CTkFrame(self.insights_frame)
        self.insights_container.pack(fill='x', padx=15, pady=(0, 15))

        # Controls row
        controls_frame = ctk.CTkFrame(self.insights_container)
        controls_frame.pack(fill='x', padx=10, pady=(10, 10))

        type_label = ctk.CTkLabel(controls_frame, text="Analysis Type:")
        type_label.pack(side='left', padx=10, pady=10)

        self.analysis_type_var = tk.StringVar(value="failures")
        self.analysis_type_combo = ctk.CTkComboBox(
            controls_frame,
            variable=self.analysis_type_var,
            values=["failures", "improvements", "report", "trends"],
            width=150,
            height=30
        )
        self.analysis_type_combo.pack(side='left', padx=(0, 20), pady=10)

        self.generate_insights_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Insights",
            command=self._generate_insights,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        self.generate_insights_btn.pack(side='right', padx=10, pady=10)

        # Insights display
        self.insights_display = ctk.CTkTextbox(
            self.insights_container,
            height=200,
            state="disabled"
        )
        self.insights_display.pack(fill='x', padx=10, pady=(0, 10))

    def _create_chat_section(self):
        """Create QA assistant chat interface (matching batch processing theme)."""
        self.chat_frame = ctk.CTkFrame(self.main_container)
        self.chat_frame.pack(fill='both', expand=True, pady=(0, 20))

        # Chat header with label and controls
        chat_header_frame = ctk.CTkFrame(self.chat_frame)
        chat_header_frame.pack(fill='x', padx=15, pady=(15, 10))

        self.chat_label = ctk.CTkLabel(
            chat_header_frame,
            text="QA Assistant Chat:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.chat_label.pack(side='left')

        # Chat history controls
        history_frame = ctk.CTkFrame(chat_header_frame)
        history_frame.pack(side='right')

        self.chat_history_combo = ctk.CTkComboBox(
            history_frame,
            values=["New Chat"],
            command=self._select_chat_session,
            width=200,
            height=30
        )
        self.chat_history_combo.pack(side='left', padx=(0, 10))
        self.chat_history_combo.set("New Chat")

        self.clear_history_btn = ctk.CTkButton(
            history_frame,
            text="Clear History",
            command=self._clear_chat_history,
            width=100,
            height=30,
            fg_color="red",
            hover_color="darkred"
        )
        self.clear_history_btn.pack(side='left')

        # Chat container
        self.chat_container = ctk.CTkFrame(self.chat_frame)
        self.chat_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Chat display
        self.chat_display = ctk.CTkTextbox(
            self.chat_container,
            height=300,
            state="disabled"
        )
        self.chat_display.pack(fill='both', expand=True, padx=10, pady=(10, 10))

        # Chat input row
        input_frame = ctk.CTkFrame(self.chat_container)
        input_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.chat_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Ask the AI assistant about quality analysis...",
            height=40
        )
        self.chat_entry.pack(side='left', fill='x', expand=True, padx=(10, 10), pady=10)

        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self._send_chat_message,
            width=80,
            height=40
        )
        self.send_btn.pack(side='right', padx=(0, 10), pady=10)

        # Bind Enter key to send message
        self.chat_entry.bind('<Return>', lambda e: self._send_chat_message())

        # Initialize chat with welcome message
        if self.ai_client:
            self._add_chat_message("AI Assistant", "Hello! I'm your QA assistant. I can help you understand your analysis data, suggest improvements, and answer questions about quality metrics.", is_user=False)
        else:
            self._add_chat_message("System", "AI features are not configured. To enable AI assistance, please configure API settings in the Settings page.", is_user=False)

    def _create_report_section(self):
        """Create automated report generation section (matching batch processing theme)."""
        self.report_frame = ctk.CTkFrame(self.main_container)
        self.report_frame.pack(fill='x', pady=(0, 20))

        self.report_label = ctk.CTkLabel(
            self.report_frame,
            text="Automated Report Generation:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.report_label.pack(anchor='w', padx=15, pady=(15, 10))

        # Report container
        self.report_container = ctk.CTkFrame(self.report_frame)
        self.report_container.pack(fill='x', padx=15, pady=(0, 15))

        # Report controls
        controls_frame = ctk.CTkFrame(self.report_container)
        controls_frame.pack(fill='x', padx=10, pady=(10, 10))

        report_type_label = ctk.CTkLabel(controls_frame, text="Report Type:")
        report_type_label.pack(side='left', padx=10, pady=10)

        self.report_type_var = tk.StringVar(value="summary")
        self.report_type_combo = ctk.CTkComboBox(
            controls_frame,
            variable=self.report_type_var,
            values=["summary", "detailed", "trends", "failures"],
            width=120,
            height=30
        )
        self.report_type_combo.pack(side='left', padx=(0, 20), pady=10)

        days_label = ctk.CTkLabel(controls_frame, text="Days:")
        days_label.pack(side='left', padx=10, pady=10)

        self.days_var = tk.StringVar(value="30")
        self.days_combo = ctk.CTkComboBox(
            controls_frame,
            variable=self.days_var,
            values=["7", "30", "90", "365"],
            width=80,
            height=30
        )
        self.days_combo.pack(side='left', padx=(0, 20), pady=10)

        self.generate_report_btn = ctk.CTkButton(
            controls_frame,
            text="Generate Report",
            command=self._generate_report,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.generate_report_btn.pack(side='right', padx=10, pady=10)

        # Report status
        self.report_status = ctk.CTkLabel(
            self.report_container,
            text="Ready to generate reports",
            font=ctk.CTkFont(size=12)
        )
        self.report_status.pack(padx=10, pady=(0, 10))

    def _initialize_ai_client(self):
        """Initialize AI client if configured with timeout to prevent freezing."""
        import time
        
        try:
            start_time = time.time()
            timeout_duration = 10  # 10 second timeout
            
            # Check if API is configured
            if not hasattr(self.main_window, 'config') or not hasattr(self.main_window.config, 'api'):
                self.logger.info("API configuration not available")
                if self.winfo_exists():
                    self.after(500, lambda: self.show_notification(
                        'AI configuration not found. AI features disabled.', 'info'
                    ))
                return
                
            if not self.main_window.config.api.enabled:
                self.logger.info("API is disabled in configuration")
                if self.winfo_exists():
                    self.after(500, lambda: self.show_notification(
                        'AI is disabled in configuration.', 'info'
                    ))
                return

            # Determine provider
            api_key = getattr(self.main_window.config.api, 'api_key', None)
            base_url = getattr(self.main_window.config.api, 'base_url', '')
            
            if api_key:
                if 'claude' in base_url.lower():
                    provider = AIProvider.ANTHROPIC
                else:
                    provider = AIProvider.OPENAI
            else:
                provider = AIProvider.OLLAMA

            # Quick connection test with timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_duration - 2:  # If already took too long, skip
                raise TimeoutError("Initialization taking too long")

            self.ai_client = QAAIAnalyzer(
                provider=provider,
                api_key=api_key,
                cache_ttl_hours=24,
                max_retries=1,  # Reduced retries for faster failure
                timeout=5  # 5 second timeout per request
            )

            # Show success notification
            self.logger.info(f'Connected to {provider.value} AI service')
            if self.winfo_exists():
                self.after(500, lambda: self.show_notification(
                    f'Connected to {provider.value} AI service', 'success'
                ))

        except TimeoutError:
            self.logger.warning("AI client initialization timed out - disabling AI features")
            if self.winfo_exists():
                self.after(500, lambda: self.show_notification(
                    'AI connection timed out. AI features disabled.', 'warning'
                ))
        except Exception as e:
            self.logger.error(f"Failed to initialize AI client: {e}")
            # Show error notification
            if self.winfo_exists():
                self.after(500, lambda: self.show_notification(
                    'AI connection failed. AI features unavailable.', 'warning'
                ))

    def _generate_insights(self):
        """Generate AI insights based on selected type."""
        if not self.ai_client:
            self.show_notification("AI is not configured. Please check API settings.", "warning")
            self.insights_display.configure(state="normal")
            self.insights_display.delete('1.0', 'end')
            self.insights_display.insert('1.0', "AI Features Not Available\n\nTo use AI features:\n1. Configure API settings in Settings page\n2. Ensure API key is valid\n3. Check internet connection\n\nYou can still use all other features without AI.")
            self.insights_display.configure(state="disabled")
            return

        # Clear previous insights
        self.insights_display.configure(state="normal")
        self.insights_display.delete('1.0', 'end')
        self.insights_display.insert('1.0', "Generating insights...\n")
        self.insights_display.configure(state="disabled")
        self.update()

        # Get analysis type
        analysis_type = self.analysis_type_var.get()

        # Disable button during generation
        self.generate_insights_btn.configure(state="disabled")

        # Run in background thread
        thread = threading.Thread(
            target=self._run_insight_generation,
            args=(analysis_type,),
            daemon=True
        )
        thread.start()

    def _run_insight_generation(self, analysis_type: str):
        """Run insight generation in background."""
        try:
            # Check if database is available
            if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
                self.winfo_toplevel().after(0, lambda: self._display_insights(
                    "Database not connected. Please ensure database is configured and connected.",
                    is_error=True
                ))
                return
                
            # Get recent data
            results = self.main_window.db_manager.get_historical_data(
                days_back=30,
                limit=100,
                include_tracks=True
            )

            if not results:
                self.winfo_toplevel().after(0, lambda: self._display_insights(
                    "No recent data available for analysis."
                ))
                return

            # Convert to format for AI
            data_summary = self._prepare_data_summary(results)

            # Generate insights based on type
            if analysis_type == "failures":
                response = self.ai_client.analyze_failures(data_summary)
            elif analysis_type == "improvements":
                response = self.ai_client.suggest_improvements(data_summary)
            elif analysis_type == "report":
                response = self.ai_client.generate_report(data_summary)
            else:  # trends
                response = self.ai_client.custom_analysis(
                    system_prompt="You are a trend analysis expert for potentiometer QA.",
                    user_prompt="Analyze the following data for trends and patterns:\n\n{data}",
                    data=data_summary
                )

            # Display insights
            self.winfo_toplevel().after(0, lambda: self._display_insights(response.content))

            # Show cost info
            cost_msg = f'Analysis Complete - Cost: ${response.cost:.4f} | Tokens: {sum(response.tokens_used.values())}'
            self.winfo_toplevel().after(0, lambda: self.show_notification(cost_msg, 'info'))

        except Exception as e:
            self.winfo_toplevel().after(0, lambda: self._display_insights(
                f"Error generating insights: {str(e)}",
                is_error=True
            ))

    def _prepare_data_summary(self, results) -> Dict[str, Any]:
        """Prepare data summary for AI analysis."""
        # Convert SQLAlchemy results to summary
        total = len(results)
        passed = sum(1 for r in results if r.overall_status.value == 'Pass')
        failed = total - passed

        # Model statistics
        model_stats = {}
        sigma_values = []
        risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}

        for result in results:
            model = result.model
            if model not in model_stats:
                model_stats[model] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'sigma_values': []
                }

            model_stats[model]['total'] += 1
            if result.overall_status.value == 'Pass':
                model_stats[model]['passed'] += 1
            else:
                model_stats[model]['failed'] += 1

            # Get track data
            if result.tracks:
                for track in result.tracks:
                    if track.sigma_gradient:
                        sigma_values.append(track.sigma_gradient)
                        model_stats[model]['sigma_values'].append(track.sigma_gradient)
                    if track.risk_category:
                        risk_counts[track.risk_category.value] += 1

        # Calculate statistics
        avg_sigma = sum(sigma_values) / len(sigma_values) if sigma_values else 0

        summary = {
            'total_units': total,
            'pass_rate': passed / total * 100 if total > 0 else 0,
            'failed_units': failed,
            'average_sigma_gradient': avg_sigma,
            'risk_distribution': risk_counts,
            'model_performance': {
                model: {
                    'pass_rate': (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                    'avg_sigma': sum(stats['sigma_values']) / len(stats['sigma_values']) if stats[
                        'sigma_values'] else 0,
                    'total_tested': stats['total']
                }
                for model, stats in model_stats.items()
            },
            'analysis_period': 'Last 30 days',
            'timestamp': datetime.now().isoformat()
        }

        return summary

    def _display_insights(self, content: str, is_error: bool = False):
        """Display AI insights with formatting in CTk textbox."""
        try:
            self.insights_display.configure(state="normal")
            self.insights_display.delete('1.0', 'end')

            if is_error:
                self.insights_display.insert('1.0', f"Error generating insights:\n{content}")
                self.insights_display.configure(state="disabled")
                return

            # Format the content nicely
            lines = content.split('\n')
            formatted_content = ""
            
            for line in lines:
                if not line.strip():
                    formatted_content += '\n'
                    continue

                if line.strip().startswith('#'):
                    # Heading
                    text = line.strip('#').strip()
                    formatted_content += f"{text}\n"
                elif line.strip().startswith('✓') or 'positive' in line.lower() or 'good' in line.lower():
                    # Positive point
                    formatted_content += f"{line}\n"
                elif line.strip().startswith('⚠') or 'warning' in line.lower() or 'concern' in line.lower():
                    # Warning point
                    formatted_content += f"{line}\n"
                elif line.strip().startswith('✗') or 'error' in line.lower() or 'critical' in line.lower():
                    # Error point
                    formatted_content += f"{line}\n"
                else:
                    # Normal text
                    formatted_content += f"{line}\n"

            self.insights_display.insert('1.0', formatted_content)
            self.insights_display.configure(state="disabled")

            # Save current analysis
            self.current_analysis = content

        except Exception as e:
            print(f"Error displaying insights: {e}")

    def _add_chat_message(self, sender: str, message: str, is_user: bool = True):
        """Add a message to the chat display using CTk textbox."""
        try:
            self.chat_display.configure(state="normal")
            
            # Add timestamp and sender
            timestamp = datetime.now().strftime("%H:%M")
            sender_prefix = "You" if is_user else sender
            
            # Add the message with formatting
            self.chat_display.insert('end', f"[{timestamp}] {sender_prefix}:\n")
            self.chat_display.insert('end', f"{message}\n\n")
            
            # Scroll to bottom
            self.chat_display.see('end')
            self.chat_display.configure(state="disabled")
            
            # Save to chat history
            self.chat_history.append({
                'sender': sender,
                'message': message,
                'is_user': is_user,
                'timestamp': timestamp
            })
            
        except Exception as e:
            print(f"Error adding chat message: {e}")

    def _send_chat_message(self):
        """Send user message to AI assistant."""
        message = self.chat_entry.get().strip()
        if not message:
            return

        if not self.ai_client:
            self._add_chat_message("System", "AI is not configured. Please configure API settings to use chat features.", is_user=False)
            self.chat_entry.delete(0, 'end')
            return

        # Add user message to chat
        self._add_chat_message("You", message, is_user=True)

        # Clear input
        self.chat_entry.delete(0, 'end')

        # Add thinking indicator
        self._add_chat_message("AI Assistant", "Thinking...", is_user=False)

        # Process in background
        thread = threading.Thread(
            target=self._process_chat_message,
            args=(message,),
            daemon=True
        )
        thread.start()

    def _process_chat_message(self, message: str):
        """Process chat message with AI."""
        try:
            # Get context data if available
            context_data = None
            if self.current_analysis:
                context_data = {
                    'recent_analysis': self.current_analysis,
                    'has_database': self.main_window.db_manager is not None
                }

            # Get AI response
            response = self.ai_client.interpret_data(
                data=context_data or "No specific data loaded",
                question=message
            )

            # Remove thinking message and add response
            self.winfo_toplevel().after(0, self._remove_last_message)
            self.winfo_toplevel().after(0, lambda: self._add_chat_message(
                "AI Assistant",
                response.content,
                is_user=False
            ))

        except Exception as e:
            self.winfo_toplevel().after(0, self._remove_last_message)
            self.winfo_toplevel().after(0, lambda: self._add_chat_message(
                "AI Assistant",
                f"I encountered an error: {str(e)}",
                is_user=False
            ))

    def _remove_last_message(self):
        """Remove the last message from chat (for removing 'thinking' indicator)."""
        try:
            self.chat_display.configure(state="normal")
            
            # Find and remove the last "Thinking..." message
            content = self.chat_display.get("1.0", "end-1c")
            lines = content.split('\n')
            
            # Find the last occurrence of "Thinking..."
            for i in range(len(lines) - 1, -1, -1):
                if "Thinking..." in lines[i]:
                    # Remove this line and the timestamp line before it
                    if i > 0:
                        lines.pop(i)  # Remove "Thinking..."
                        lines.pop(i-1)  # Remove timestamp line
                        # Also remove empty line after if exists
                        if i < len(lines) and not lines[i-1].strip():
                            lines.pop(i-1)
                    break
            
            # Update the display
            self.chat_display.delete("1.0", "end")
            self.chat_display.insert("1.0", '\n'.join(lines))
            self.chat_display.configure(state="disabled")
            
        except Exception as e:
            self.logger.debug(f"Error removing last message: {e}")

    def _use_suggestion(self, suggestion: str):
        """Use a suggested question."""
        self.chat_entry.delete('0', tk.END)
        self.chat_entry.insert('0', suggestion)
        self._send_chat_message()

    def _generate_report(self):
        """Generate automated AI report."""
        if not self.ai_client:
            self.show_notification("AI is not configured. Please check API settings.", "warning")
            self.report_status.configure(text="AI not available")
            return

        # Get parameters
        report_type = self.report_type_var.get()
        days = int(self.days_var.get())

        # Disable button and start progress
        self.generate_report_btn.configure(state="disabled")
        self.report_status.configure(text="Generating report...")

        # Run in thread
        thread = threading.Thread(
            target=self._run_report_generation,
            args=(report_type, days),
            daemon=True
        )
        thread.start()

    def _run_report_generation(self, report_type: str, days: int):
        """Run report generation in background."""
        try:
            # Check if database is available
            if not hasattr(self.main_window, 'db_manager') or not self.main_window.db_manager:
                self.winfo_toplevel().after(0, lambda: self.show_notification(
                    "Database not connected. Cannot generate report.", "error"
                ))
                return
                
            # Update status
            self.winfo_toplevel().after(0, lambda: self.report_status.configure(
                text="Loading data..."
            ))

            # Get data
            results = self.main_window.db_manager.get_historical_data(
                days_back=days,
                include_tracks=True
            )

            if not results:
                self.winfo_toplevel().after(0, lambda: messagebox.showwarning(
                    "No Data",
                    f"No data found for the last {days} days"
                ))
                return

            # Prepare data
            data_summary = self._prepare_data_summary(results)

            # Update status
            self.winfo_toplevel().after(0, lambda: self.report_status.configure(
                text="Generating AI content..."
            ))

            # Generate report content based on type
            if report_type == "detailed":
                response = self.ai_client.generate_report(data_summary)
            elif report_type == "summary":
                response = self.ai_client.custom_analysis(
                    system_prompt="You are an executive report writer. Create concise, high-level summaries.",
                    user_prompt="Create an executive summary of this QA data:\n\n{data}",
                    data=data_summary
                )
            elif report_type == "technical":
                response = self.ai_client.custom_analysis(
                    system_prompt="You are a technical report writer. Focus on detailed metrics and engineering insights.",
                    user_prompt="Create a technical analysis report:\n\n{data}",
                    data=data_summary
                )
            else:  # failures
                response = self.ai_client.analyze_failures(data_summary)

            # Save report
            self.winfo_toplevel().after(0, lambda: self._save_report(
                response.content,
                report_type,
                data_summary
            ))

        except Exception as e:
            self.winfo_toplevel().after(0, lambda: messagebox.showerror(
                "Report Generation Error",
                f"Failed to generate report:\n{str(e)}"
            ))

        finally:
            # Re-enable button and update status
            self.winfo_toplevel().after(0, lambda: self.generate_report_btn.configure(state='normal'))
            self.winfo_toplevel().after(0, lambda: self.report_status.configure(text="Ready to generate reports"))

    def _save_report(self, content: str, report_type: str, data_summary: Dict[str, Any]):
        """Save generated report to file."""
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[
                ('Text files', '*.txt'),
                ('Markdown files', '*.md'),
                ('All files', '*.*')
            ],
            initialfile=f'ai_report_{report_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        if not filename:
            return

        try:
            # Format report with metadata
            full_report = f"""AI-Generated QA Report
        {'=' * 50}
        Type: {report_type.title()}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Data Period: Last {self.days_var.get()} days
        Total Units Analyzed: {data_summary['total_units']}
        Overall Pass Rate: {data_summary['pass_rate']:.1f}%

        {'=' * 50}
        REPORT CONTENT
        {'=' * 50}

        {content}

        {'=' * 50}
        END OF REPORT
        """

            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_report)

            messagebox.showinfo(
                "Report Saved",
                f"Report saved successfully to:\n{filename}"
            )

            # Show notification
            self.show_notification(f'{report_type.title()} report saved successfully', 'success')

        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"Failed to save report:\n{str(e)}"
            )

    def _open_file(self, filename: str):
        """Open file in default application."""
        import os
        import platform

        try:
            if platform.system() == 'Windows':
                os.startfile(filename)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{filename}"')
            else:  # Linux
                os.system(f'xdg-open "{filename}"')
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{str(e)}")
    
    def _apply_hover_fixes(self):
        """Apply hover fixes to prevent glitching and shifting."""
        try:
            # Import hover fix utilities
            from laser_trim_analyzer.gui.widgets.hover_fix import fix_hover_glitches, stabilize_layout
            
            # Fix hover glitches on all widgets
            fix_hover_glitches(self)
            
            # Stabilize layout to prevent shifting
            stabilize_layout(self.main_container)
            
            # Specifically fix buttons that might have hover issues
            buttons_to_fix = [
                self.generate_insights_btn,
                self.send_btn,
                self.generate_report_btn
            ]
            
            for button in buttons_to_fix:
                if hasattr(button, '_hover_color'):
                    button._hover_color = None  # Disable default hover
                    
            self.logger.debug("Hover fixes applied successfully to AI Insights page")
        except Exception as e:
            self.logger.warning(f"Failed to apply hover fixes: {e}")
    
    def on_show(self):
        """Called when page is shown."""
        # Re-check AI client status
        if not self.ai_client and hasattr(self.main_window, 'config') and hasattr(self.main_window.config, 'api'):
            self._initialize_ai_client()
            
    def on_hide(self):
        """Called when page is hidden."""
        # Cancel any pending timers
        if self._notification_timer:
            self.after_cancel(self._notification_timer)
            self._notification_timer = None
        
        # Hide any visible alerts
        if hasattr(self, 'alert_frame'):
            self.alert_frame.pack_forget()
    
    def show_notification(self, message: str, alert_type: str = "info"):
        """Show notification using simple alert frame to prevent glitching."""
        try:
            # Color scheme for alert types
            colors = {
                'info': {'fg': "#3498db", 'bg': ("#e3f2fd", "#1976d2")},
                'success': {'fg': "#27ae60", 'bg': ("#e8f5e9", "#2e7d32")},
                'warning': {'fg': "#f39c12", 'bg': ("#fff3e0", "#f57c00")},
                'error': {'fg': "#e74c3c", 'bg': ("#ffebee", "#c62828")}
            }
            
            color = colors.get(alert_type, colors['info'])
            
            # Update alert message
            self.alert_frame.configure(fg_color=color['bg'])
            self.alert_message.configure(text=message, text_color=color['fg'])
            self.alert_frame.pack(fill='x', pady=(0, 10))
            
            # Cancel previous notification timer if exists
            if self._notification_timer:
                self.after_cancel(self._notification_timer)
            
            # Auto-hide after 5 seconds
            self._notification_timer = self.after(5000, lambda: self.alert_frame.pack_forget())
            
        except Exception as e:
            self.logger.error(f"Error showing notification: {e}")
    
    def _load_chat_sessions(self):
        """Load chat sessions from storage."""
        try:
            # Load from settings file
            if hasattr(self.main_window, 'settings_manager') and self.main_window.settings_manager:
                sessions = self.main_window.settings_manager.get_setting('ai_chat_sessions', [])
                self.chat_sessions = sessions
                
                # Update combo box values if it exists
                if hasattr(self, 'chat_history_combo'):
                    session_names = ["New Chat"]
                    for session in sessions:
                        preview = session.get('preview', 'Chat')
                        timestamp = session.get('timestamp', '')
                        name = f"{preview[:30]}... ({timestamp})" if len(preview) > 30 else f"{preview} ({timestamp})"
                        session_names.append(name)
                    self.chat_history_combo.configure(values=session_names)
        except Exception as e:
            self.logger.error(f"Error loading chat sessions: {e}")
            self.chat_sessions = []
    
    def _save_chat_session(self):
        """Save current chat session."""
        try:
            if not self.chat_history:
                return
                
            # Create session data
            session = {
                'id': datetime.now().timestamp(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'preview': self.chat_history[0]['message'] if self.chat_history else 'Empty chat',
                'messages': self.chat_history,
                'message_count': len(self.chat_history)
            }
            
            # Add to sessions list
            self.chat_sessions.append(session)
            
            # Limit to last 20 sessions
            if len(self.chat_sessions) > 20:
                self.chat_sessions = self.chat_sessions[-20:]
            
            # Save to settings
            if hasattr(self.main_window, 'settings_manager') and self.main_window.settings_manager:
                self.main_window.settings_manager.set_setting('ai_chat_sessions', self.chat_sessions)
                self.main_window.settings_manager.save_settings()
            
            # Reload sessions in combo box
            self._load_chat_sessions()
            
        except Exception as e:
            self.logger.error(f"Error saving chat session: {e}")
    
    def _select_chat_session(self, value: str):
        """Load selected chat session."""
        try:
            if value == "New Chat":
                # Save current session if it has messages
                if self.chat_history:
                    self._save_chat_session()
                
                # Clear current chat
                self.chat_history = []
                self.current_session_id = None
                self.chat_display.configure(state="normal")
                self.chat_display.delete('1.0', 'end')
                self.chat_display.configure(state="disabled")
                
                # Show welcome message
                if self.ai_client:
                    self._add_chat_message("AI Assistant", "Hello! I'm your QA assistant. How can I help you today?", is_user=False)
            else:
                # Find and load the selected session
                try:
                    session_index = self.chat_history_combo.cget('values').index(value) - 1  # -1 for "New Chat"
                    if 0 <= session_index < len(self.chat_sessions):
                        session = self.chat_sessions[session_index]
                    else:
                        self.logger.warning(f"Invalid session index: {session_index}")
                        return
                except ValueError:
                    self.logger.warning(f"Session not found: {value}")
                    return
                    self.current_session_id = session['id']
                    self.chat_history = session['messages']
                    
                    # Display the chat history
                    self.chat_display.configure(state="normal")
                    self.chat_display.delete('1.0', 'end')
                    
                    for msg in self.chat_history:
                        sender = msg['sender']
                        message = msg['message']
                        is_user = msg.get('is_user', sender == "You")
                        
                        # Style based on sender
                        if is_user:
                            self.chat_display.insert('end', f"You: {message}\n\n", "user")
                        else:
                            self.chat_display.insert('end', f"{sender}: {message}\n\n", "assistant")
                    
                    self.chat_display.configure(state="disabled")
                    self.chat_display.see('end')
                    
        except Exception as e:
            self.logger.error(f"Error selecting chat session: {e}")
    
    def _clear_chat_history(self):
        """Clear all chat history with confirmation."""
        try:
            # Confirm with user
            result = messagebox.askyesno(
                "Clear Chat History", 
                "Are you sure you want to clear all chat history?\nThis action cannot be undone."
            )
            
            if result:
                # Clear sessions
                self.chat_sessions = []
                self.chat_history = []
                self.current_session_id = None
                
                # Save empty sessions
                if hasattr(self.main_window, 'settings_manager') and self.main_window.settings_manager:
                    self.main_window.settings_manager.set_setting('ai_chat_sessions', [])
                    self.main_window.settings_manager.save_settings()
                
                # Reset UI
                self.chat_history_combo.configure(values=["New Chat"])
                self.chat_history_combo.set("New Chat")
                
                # Clear chat display
                self.chat_display.configure(state="normal")
                self.chat_display.delete('1.0', 'end')
                
                # Show welcome message
                if self.ai_client:
                    self._add_chat_message("AI Assistant", "Chat history cleared. How can I help you today?", is_user=False)
                    
                self.chat_display.configure(state="disabled")
                
                self.show_notification("Chat history cleared successfully", "success")
                
        except Exception as e:
            self.logger.error(f"Error clearing chat history: {e}")
            messagebox.showerror("Error", f"Failed to clear chat history: {str(e)}")
    
    def _apply_hover_fixes(self):
        """Apply hover fixes to prevent UI glitching."""
        try:
            # Import hover fix utilities
            from laser_trim_analyzer.gui.widgets.hover_fix import fix_hover_glitches, stabilize_layout
            
            # Fix hover glitches on all widgets
            fix_hover_glitches(self)
            
            # Stabilize layout to prevent shifting
            stabilize_layout(self.main_container)
            
            self.logger.debug("Hover fixes applied successfully")
        except ImportError:
            self.logger.warning("Hover fix utilities not available")
        except Exception as e:
            self.logger.warning(f"Failed to apply hover fixes: {e}")
    
    def cleanup(self):
        """Clean up resources when page is destroyed."""
        try:
            # Cancel any pending timers
            if self._notification_timer:
                self.after_cancel(self._notification_timer)
                self._notification_timer = None
            
            # Stop any running threads
            self._stop_requested = True
            
            # Clean up AI client
            if hasattr(self, 'ai_client') and self.ai_client:
                self.ai_client = None
            
            # Clear data
            self.chat_history = []
            self.current_analysis = None
            
            self.logger.debug("AI Insights page cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")