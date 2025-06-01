"""
AI Insights Page for Laser Trim Analyzer

Provides interface for AI-powered analysis, chat assistant,
and automated report generation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import threading
import json
from typing import Optional, Dict, List, Any

from laser_trim_analyzer.core.models import AnalysisResult
from laser_trim_analyzer.gui.pages.base_page import BasePage
from laser_trim_analyzer.gui.widgets.alert_banner import AlertBanner, AlertStack
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.api.client import QAAIAnalyzer, AIProvider


class AIInsightsPage(BasePage):
    """AI-powered insights and analysis page."""

    def __init__(self, parent, main_window):
        self.ai_client = None
        self.chat_history = []
        self.current_analysis = None
        super().__init__(parent, main_window)
        self._initialize_ai_client()

    def _create_page(self):
        """Set up the AI insights page."""
        # Create scrollable frame
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create content in scrollable frame
        content_frame = scrollable_frame
        
        # Title
        title_frame = ttk.Frame(content_frame)
        title_frame.pack(fill='x', padx=20, pady=(20, 10))

        ttk.Label(
            title_frame,
            text="AI-Powered Insights",
            font=('Segoe UI', 24, 'bold')
        ).pack(side='left')

        # Alert stack for AI recommendations
        self.alert_stack = AlertStack(content_frame, max_alerts=3)
        self.alert_stack.pack(fill='x', padx=20, pady=(0, 10))

        # Create main sections in content_frame
        self._create_insights_section(content_frame)
        self._create_chat_section(content_frame)
        self._create_report_section(content_frame)

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

            # Show success alert
            self.alert_stack.add_alert(
                alert_type='success',
                title='AI Connected',
                message=f'Connected to {provider.value} AI service',
                auto_dismiss=5
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize AI client: {e}")
            self.alert_stack.add_alert(
                alert_type='error',
                title='AI Connection Failed',
                message=str(e),
                dismissible=True
            )

    def _create_insights_section(self, content_frame):
        """Create automatic insights section."""
        insights_frame = ttk.LabelFrame(
            content_frame,
            text="Automatic Insights",
            padding=15
        )
        insights_frame.pack(fill='x', padx=20, pady=10)

        # Controls
        controls_frame = ttk.Frame(insights_frame)
        controls_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(controls_frame, text="Analysis Type:").pack(side='left', padx=(0, 10))

        self.analysis_type_var = tk.StringVar(value="failures")
        analysis_types = [
            ("Failure Analysis", "failures"),
            ("Process Improvements", "improvements"),
            ("Quality Report", "report"),
            ("Trend Analysis", "trends")
        ]

        for text, value in analysis_types:
            ttk.Radiobutton(
                controls_frame,
                text=text,
                variable=self.analysis_type_var,
                value=value
            ).pack(side='left', padx=(0, 15))

        ttk.Button(
            controls_frame,
            text="Generate Insights",
            command=self._generate_insights,
            style='Primary.TButton'
        ).pack(side='right')

        # Insights display
        self.insights_frame = ttk.Frame(insights_frame)
        self.insights_frame.pack(fill='both', expand=True)

        # Insights text with scrollbar
        text_frame = ttk.Frame(self.insights_frame)
        text_frame.pack(fill='both', expand=True)

        self.insights_text = tk.Text(
            text_frame,
            height=12,
            wrap='word',
            font=('Segoe UI', 10)
        )
        scroll = ttk.Scrollbar(text_frame, command=self.insights_text.yview)
        self.insights_text.config(yscrollcommand=scroll.set)

        self.insights_text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        # Configure text tags for formatting
        self.insights_text.tag_configure('heading', font=('Segoe UI', 12, 'bold'))
        self.insights_text.tag_configure('positive', foreground='#27ae60')
        self.insights_text.tag_configure('warning', foreground='#f39c12')
        self.insights_text.tag_configure('error', foreground='#e74c3c')

    def _create_chat_section(self, content_frame):
        """Create QA assistant chat interface."""
        chat_frame = ttk.LabelFrame(
            content_frame,
            text="QA Assistant Chat",
            padding=15
        )
        chat_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Chat display
        chat_display_frame = ttk.Frame(chat_frame)
        chat_display_frame.pack(fill='both', expand=True)

        # Chat history with scrollbar
        chat_scroll_frame = ttk.Frame(chat_display_frame)
        chat_scroll_frame.pack(fill='both', expand=True)

        self.chat_canvas = tk.Canvas(
            chat_scroll_frame,
            bg='white',
            highlightthickness=1,
            highlightbackground='#e0e0e0'
        )
        chat_scrollbar = ttk.Scrollbar(
            chat_scroll_frame,
            orient='vertical',
            command=self.chat_canvas.yview
        )
        self.chat_canvas.configure(yscrollcommand=chat_scrollbar.set)

        self.chat_canvas.pack(side='left', fill='both', expand=True)
        chat_scrollbar.pack(side='right', fill='y')

        # Chat message frame
        self.chat_messages_frame = ttk.Frame(self.chat_canvas)
        self.chat_canvas_window = self.chat_canvas.create_window(
            (0, 0),
            window=self.chat_messages_frame,
            anchor='nw'
        )

        # Bind canvas resize
        self.chat_messages_frame.bind(
            '<Configure>',
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox('all'))
        )
        self.chat_canvas.bind(
            '<Configure>',
            lambda e: self.chat_canvas.itemconfig(
                self.chat_canvas_window,
                # Continue _create_chat_section
                width=e.width
            )
        )

        # Add welcome message
        self._add_chat_message(
            "AI Assistant",
            "Hello! I'm your QA assistant. I can help you understand your analysis results, "
            "identify patterns, and suggest improvements. What would you like to know?",
            is_user=False
        )

        # Chat input
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill='x', pady=(10, 0))

        self.chat_input = tk.Text(
            input_frame,
            height=3,
            wrap='word',
            font=('Segoe UI', 10)
        )
        self.chat_input.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Bind Enter key to send message
        self.chat_input.bind('<Return>', lambda e: self._send_chat_message() if not e.state else None)
        self.chat_input.bind('<Shift-Return>', lambda e: None)  # Allow Shift+Enter for new line

        send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._send_chat_message,
            style='Primary.TButton'
        )
        send_btn.pack(side='right')

        # Suggested questions
        suggestions_frame = ttk.Frame(chat_frame)
        suggestions_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(
            suggestions_frame,
            text="Suggested questions:",
            font=('Segoe UI', 9, 'italic')
        ).pack(anchor='w')

        suggestions = [
            "What is the current pass rate trend?",
            "Which models have the highest failure risk?",
            "How can I improve sigma gradient performance?",
            "Explain the recent anomalies"
        ]

        suggestion_btns_frame = ttk.Frame(suggestions_frame)
        suggestion_btns_frame.pack(fill='x', pady=(5, 0))

        for suggestion in suggestions:
            ttk.Button(
                suggestion_btns_frame,
                text=suggestion,
                command=lambda s=suggestion: self._use_suggestion(s)
            ).pack(side='left', padx=(0, 5))

    def _create_report_section(self, content_frame):
        """Create automated report generation section."""
        report_frame = ttk.LabelFrame(
            content_frame,
            text="AI Report Generation",
            padding=15
        )
        report_frame.pack(fill='x', padx=20, pady=(0, 20))

        # Report options
        options_frame = ttk.Frame(report_frame)
        options_frame.pack(fill='x')

        # Report type
        ttk.Label(options_frame, text="Report Type:").grid(
            row=0, column=0, sticky='w', padx=(0, 10), pady=5
        )

        self.report_type_var = tk.StringVar(value="comprehensive")
        report_types = [
            ("Comprehensive Analysis", "comprehensive"),
            ("Executive Summary", "executive"),
            ("Technical Details", "technical"),
            ("Maintenance Report", "maintenance")
        ]

        col = 1
        for text, value in report_types:
            ttk.Radiobutton(
                options_frame,
                text=text,
                variable=self.report_type_var,
                value=value
            ).grid(row=0, column=col, sticky='w', padx=(0, 15), pady=5)
            col += 1

        # Date range
        ttk.Label(options_frame, text="Data Range:").grid(
            row=1, column=0, sticky='w', padx=(0, 10), pady=5
        )

        self.report_range_var = tk.StringVar(value="30")
        range_frame = ttk.Frame(options_frame)
        range_frame.grid(row=1, column=1, columnspan=2, sticky='w', pady=5)

        ttk.Label(range_frame, text="Last").pack(side='left', padx=(0, 5))
        ttk.Entry(
            range_frame,
            textvariable=self.report_range_var,
            width=10
        ).pack(side='left', padx=(0, 5))
        ttk.Label(range_frame, text="days").pack(side='left')

        # Include options
        ttk.Label(options_frame, text="Include:").grid(
            row=2, column=0, sticky='w', padx=(0, 10), pady=5
        )

        include_frame = ttk.Frame(options_frame)
        include_frame.grid(row=2, column=1, columnspan=3, sticky='w', pady=5)

        self.include_charts = tk.BooleanVar(value=True)
        self.include_predictions = tk.BooleanVar(value=True)
        self.include_recommendations = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            include_frame,
            text="Charts & Visualizations",
            variable=self.include_charts
        ).pack(side='left', padx=(0, 15))

        ttk.Checkbutton(
            include_frame,
            text="Predictions",
            variable=self.include_predictions
        ).pack(side='left', padx=(0, 15))

        ttk.Checkbutton(
            include_frame,
            text="Recommendations",
            variable=self.include_recommendations
        ).pack(side='left')

        # Generate button and progress
        action_frame = ttk.Frame(report_frame)
        action_frame.pack(fill='x', pady=(15, 0))

        self.generate_report_btn = ttk.Button(
            action_frame,
            text="Generate Report",
            command=self._generate_report,
            style='Primary.TButton'
        )
        self.generate_report_btn.pack(side='left', padx=(0, 20))

        self.report_progress = ttk.Progressbar(
            action_frame,
            mode='indeterminate',
            length=200
        )
        self.report_progress.pack(side='left', padx=(0, 10))

        self.report_status_label = ttk.Label(action_frame, text="")
        self.report_status_label.pack(side='left')

    def _generate_insights(self):
        """Generate AI insights based on selected type."""
        if not self.ai_client:
            messagebox.showerror("Error", "AI service not connected")
            return

        analysis_type = self.analysis_type_var.get()

        # Get recent data from database
        if not self.main_window.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        # Clear previous insights
        self.insights_text.delete('1.0', tk.END)
        self.insights_text.insert('1.0', "Generating insights...\n", 'heading')
        self.update()

        # Run in thread
        thread = threading.Thread(
            target=self._run_insight_generation,
            args=(analysis_type,)
        )
        thread.daemon = True
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
                self.root.after(0, lambda: self._display_insights(
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
            self.root.after(0, lambda: self._display_insights(response.content))

            # Show cost info
            self.root.after(0, lambda: self.alert_stack.add_alert(
                alert_type='info',
                title='Analysis Complete',
                message=f'Cost: ${response.cost:.4f} | Tokens: {sum(response.tokens_used.values())}',
                auto_dismiss=10
            ))

        except Exception as e:
            self.root.after(0, lambda: self._display_insights(
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
        """Display AI insights with formatting."""
        self.insights_text.delete('1.0', tk.END)

        if is_error:
            self.insights_text.insert('1.0', content, 'error')
            return

        # Parse and format content
        lines = content.split('\n')
        for line in lines:
            if not line.strip():
                self.insights_text.insert(tk.END, '\n')
                continue

            # Detect formatting
            if line.strip().startswith('#'):
                # Heading
                text = line.strip('#').strip()
                self.insights_text.insert(tk.END, text + '\n', 'heading')
            elif line.strip().startswith('✓') or 'positive' in line.lower() or 'good' in line.lower():
                # Positive point
                self.insights_text.insert(tk.END, line + '\n', 'positive')
            elif line.strip().startswith('⚠') or 'warning' in line.lower() or 'concern' in line.lower():
                # Warning point
                self.insights_text.insert(tk.END, line + '\n', 'warning')
            elif line.strip().startswith('✗') or 'error' in line.lower() or 'critical' in line.lower():
                # Error point
                self.insights_text.insert(tk.END, line + '\n', 'error')
            else:
                # Normal text
                self.insights_text.insert(tk.END, line + '\n')

        # Save current analysis
        self.current_analysis = content

    def _add_chat_message(self, sender: str, message: str, is_user: bool = True):
        """Add a message to the chat display."""
        # Create message frame
        msg_frame = ttk.Frame(self.chat_messages_frame)
        msg_frame.pack(fill='x', padx=10, pady=5)

        # Message bubble
        bubble_frame = ttk.Frame(
            msg_frame,
            relief='solid',
            borderwidth=1,
            style='Card.TFrame'
        )

        if is_user:
            bubble_frame.pack(side='right', padx=(50, 0))
            bg_color = '#e3f2fd'
        else:
            bubble_frame.pack(side='left', padx=(0, 50))
            bg_color = '#f5f5f5'

        # Sender label
        sender_label = ttk.Label(
            bubble_frame,
            text=sender,
            font=('Segoe UI', 9, 'bold')
        )
        sender_label.pack(anchor='w', padx=10, pady=(5, 0))

        # Message text
        msg_text = tk.Text(
            bubble_frame,
            wrap='word',
            width=50,
            height=1,
            font=('Segoe UI', 10),
            bg=bg_color,
            relief='flat',
            padx=10,
            pady=5
        )
        msg_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        msg_text.insert('1.0', message)

        # Adjust height
        msg_text.update_idletasks()
        lines = int(msg_text.index('end-1c').split('.')[0])
        msg_text.config(height=lines, state='disabled')

        # Add to history
        self.chat_history.append({
            'sender': sender,
            'message': message,
            'timestamp': datetime.now(),
            'is_user': is_user
        })

        # Scroll to bottom
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def _send_chat_message(self):
        """Send user message to AI assistant."""
        message = self.chat_input.get('1.0', 'end-1c').strip()
        if not message:
            return

        if not self.ai_client:
            messagebox.showerror("Error", "AI service not connected")
            return

        # Add user message to chat
        self._add_chat_message("You", message, is_user=True)

        # Clear input
        self.chat_input.delete('1.0', tk.END)

        # Add thinking indicator
        self._add_chat_message("AI Assistant", "Thinking...", is_user=False)

        # Process in thread
        thread = threading.Thread(
            target=self._process_chat_message,
            args=(message,)
        )
        thread.daemon = True
        thread.start()

        return 'break'  # Prevent default Enter behavior

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
            self.root.after(0, self._remove_last_message)
            self.root.after(0, lambda: self._add_chat_message(
                "AI Assistant",
                response.content,
                is_user=False
            ))

        except Exception as e:
            self.root.after(0, self._remove_last_message)
            self.root.after(0, lambda: self._add_chat_message(
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
        self.chat_input.delete('1.0', tk.END)
        self.chat_input.insert('1.0', suggestion)
        self._send_chat_message()

    def _generate_report(self):
        """Generate AI-powered report."""
        if not self.ai_client:
            messagebox.showerror("Error", "AI service not connected")
            return

        if not self.main_window.db_manager:
            messagebox.showerror("Error", "Database not connected")
            return

        # Get parameters
        report_type = self.report_type_var.get()
        days = int(self.report_range_var.get())

        # Disable button and start progress
        self.generate_report_btn.config(state='disabled')
        self.report_progress.start(10)
        self.report_status_label.config(text="Generating report...")

        # Run in thread
        thread = threading.Thread(
            target=self._run_report_generation,
            args=(report_type, days)
        )
        thread.daemon = True
        thread.start()

    def _run_report_generation(self, report_type: str, days: int):
        """Run report generation in background."""
        try:
            # Update status
            self.root.after(0, lambda: self.report_status_label.config(
                text="Loading data..."
            ))

            # Get data
            results = self.main_window.db_manager.get_historical_data(
                days_back=days,
                include_tracks=True
            )

            if not results:
                self.root.after(0, lambda: messagebox.showwarning(
                    "No Data",
                    f"No data found for the last {days} days"
                ))
                return

            # Prepare data
            data_summary = self._prepare_data_summary(results)

            # Update status
            self.root.after(0, lambda: self.report_status_label.config(
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
            self.root.after(0, lambda: self._save_report(
                response.content,
                report_type,
                data_summary
            ))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Report Generation Error",
                f"Failed to generate report:\n{str(e)}"
            ))

        finally:
            # Re-enable button and stop progress
            self.root.after(0, lambda: self.generate_report_btn.config(state='normal'))
            self.root.after(0, self.report_progress.stop)
            self.root.after(0, lambda: self.report_status_label.config(text=""))

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
        Data Period: Last {self.report_range_var.get()} days
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
            self.alert_stack.add_alert(
                alert_type='success',
                title='Report Generated',
                message=f'{report_type.title()} report saved',
                auto_dismiss=5,
                actions=[{
                    'text': 'Open',
                    'command': lambda: self._open_file(filename)
                }]
            )

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