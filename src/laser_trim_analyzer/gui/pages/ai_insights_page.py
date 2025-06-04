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
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.alert_banner import AlertBanner, AlertStack
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.api.client import QAAIAnalyzer, AIProvider
from laser_trim_analyzer.gui.widgets import add_mousewheel_support


class AIInsightsPage(BasePage):
    """AI-powered insights and analysis page."""

    def __init__(self, parent, main_window):
        self.ai_client = None
        self.chat_history = []
        self.current_analysis = None
        super().__init__(parent, main_window)
        self._initialize_ai_client()

    def _create_page(self):
        """Create AI insights page content with consistent theme (matching batch processing)."""
        # Main scrollable container (matching batch processing theme)
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections in order (matching batch processing pattern)
        self._create_header()
        self._create_insights_section()
        self._create_chat_section()
        self._create_report_section()

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

        self.chat_label = ctk.CTkLabel(
            self.chat_frame,
            text="QA Assistant Chat:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.chat_label.pack(anchor='w', padx=15, pady=(15, 10))

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
        self._add_chat_message("AI Assistant", "Hello! I'm your QA assistant. I can help you understand your analysis data, suggest improvements, and answer questions about quality metrics.", is_user=False)

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
        """Initialize AI client if configured."""
        if not self.main_window.config.api.enabled:
            return

        try:
            # Determine provider
            if self.main_window.config.api.api_key:
                if 'claude' in self.main_window.config.api.base_url.lower():
                    provider = AIProvider.ANTHROPIC
                else:
                    provider = AIProvider.OPENAI
            else:
                provider = AIProvider.OLLAMA

            self.ai_client = QAAIAnalyzer(
                provider=provider,
                api_key=self.main_window.config.api.api_key,
                cache_ttl_hours=24,
                max_retries=self.main_window.config.api.max_retries
            )

            # Log success instead of using alert_stack
            self.logger.info(f'Connected to {provider.value} AI service')

        except Exception as e:
            self.logger.error(f"Failed to initialize AI client: {e}")
            # Log error instead of using alert_stack
            self.logger.error(f'AI Connection Failed: {str(e)}')

    def _generate_insights(self):
        """Generate AI insights based on selected type."""
        if not self.ai_client:
            messagebox.showerror("AI Not Available", "AI client is not configured or connected")
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
            self.winfo_toplevel().after(0, lambda: self.logger.info(
                f'Analysis Complete - Cost: ${response.cost:.4f} | Tokens: {sum(response.tokens_used.values())}'
            ))

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
            
        except Exception as e:
            print(f"Error adding chat message: {e}")

    def _send_chat_message(self):
        """Send user message to AI assistant."""
        message = self.chat_entry.get().strip()
        if not message:
            return

        if not self.ai_client:
            messagebox.showerror("AI Not Available", "AI client is not configured or connected")
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
        children = self.chat_messages_frame.winfo_children()
        if children:
            children[-1].destroy()
            self.chat_history.pop()

    def _use_suggestion(self, suggestion: str):
        """Use a suggested question."""
        self.chat_entry.delete('0', tk.END)
        self.chat_entry.insert('0', suggestion)
        self._send_chat_message()

    def _generate_report(self):
        """Generate automated AI report."""
        if not self.ai_client:
            messagebox.showerror("AI Not Available", "AI client is not configured or connected")
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
            # Update status
            self.winfo_toplevel().after(0, lambda: self.report_status.config(
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
            self.winfo_toplevel().after(0, lambda: self.report_status.config(
                text="Generating AI content..."
            ))

            # Generate report content based on type
            if report_type == "comprehensive":
                response = self.ai_client.generate_report(data_summary)
            elif report_type == "executive":
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
            else:  # maintenance
                response = self.ai_client.custom_analysis(
                    system_prompt="You are a maintenance planning expert. Focus on equipment and process maintenance needs.",
                    user_prompt="Create a maintenance report based on this QA data:\n\n{data}",
                    data=data_summary
                )

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
            # Re-enable button and stop progress
            self.winfo_toplevel().after(0, lambda: self.generate_report_btn.config(state='normal'))
            self.winfo_toplevel().after(0, lambda: self.report_status.config(text=""))

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

            # Show alert
            self.logger.info(f'{report_type.title()} report saved to {filename}')

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