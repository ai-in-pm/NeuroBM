#!/usr/bin/env python3
"""
NeuroBM Automation Orchestrator.

This is the main orchestration script that coordinates all automated systems:
- Research monitoring
- Integration pipeline
- Version management
- Deployment automation

Features:
- Centralized control of all automation systems
- Scheduled execution of weekly release cycle
- Error handling and recovery
- Status monitoring and reporting
- Manual override capabilities

Usage:
    python automation/orchestrator.py --start-scheduler
    python automation/orchestrator.py --run-weekly-cycle
    python automation/orchestrator.py --status
    python automation/orchestrator.py --emergency-stop
"""

import asyncio
import logging
import json
import yaml
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import sys
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation/logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """System status information."""
    component: str
    status: str  # running, stopped, error, unknown
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class AutomationOrchestrator:
    """Main orchestrator for NeuroBM automation systems."""
    
    def __init__(self, config_path: str = "automation/config/orchestrator_config.yaml"):
        """Initialize the orchestrator."""
        self.config = self._load_config(config_path)
        self.system_status = {}
        self.is_running = False
        self.emergency_stop = False
        
        # Initialize component status
        self._init_system_status()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'schedule': {
                    'weekly_release_day': 'tuesday',
                    'weekly_release_time': '10:00',
                    'timezone': 'UTC'
                },
                'components': {
                    'research_monitor': {
                        'enabled': True,
                        'schedule': 'daily',
                        'time': '06:00'
                    },
                    'integration_pipeline': {
                        'enabled': True,
                        'schedule': 'on_demand',
                        'auto_integrate_threshold': 0.9
                    },
                    'version_manager': {
                        'enabled': True,
                        'schedule': 'weekly',
                        'day': 'monday',
                        'time': '09:00'
                    },
                    'deployment_manager': {
                        'enabled': True,
                        'schedule': 'weekly',
                        'day': 'tuesday',
                        'time': '10:00'
                    }
                },
                'error_handling': {
                    'max_retries': 3,
                    'retry_delay_minutes': 15,
                    'emergency_stop_on_critical': True
                },
                'notifications': {
                    'enabled': True,
                    'recipients': ['admin@neurobm.org']
                }
            }
    
    def _init_system_status(self):
        """Initialize system status tracking."""
        components = ['research_monitor', 'integration_pipeline', 'version_manager', 'deployment_manager']
        
        for component in components:
            self.system_status[component] = SystemStatus(
                component=component,
                status='stopped',
                last_run=None,
                next_run=None
            )
    
    def start_scheduler(self):
        """Start the automation scheduler."""
        logger.info("Starting NeuroBM Automation Orchestrator")
        
        # Schedule weekly release cycle
        self._schedule_weekly_cycle()
        
        # Schedule daily research monitoring
        self._schedule_research_monitoring()
        
        # Schedule status checks
        self._schedule_status_checks()
        
        self.is_running = True
        
        try:
            while self.is_running and not self.emergency_stop:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            self.stop_scheduler()
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            self._handle_critical_error(e)
    
    def stop_scheduler(self):
        """Stop the automation scheduler."""
        logger.info("Stopping NeuroBM Automation Orchestrator")
        self.is_running = False
        schedule.clear()
    
    def emergency_stop(self):
        """Emergency stop all automation."""
        logger.critical("EMERGENCY STOP activated")
        self.emergency_stop = True
        self.is_running = False
        schedule.clear()
        
        # Send emergency notification
        self._send_notification(
            "EMERGENCY STOP - NeuroBM Automation",
            "Emergency stop has been activated. All automation is halted."
        )
    
    def _schedule_weekly_cycle(self):
        """Schedule the weekly release cycle."""
        release_day = self.config['schedule']['weekly_release_day']
        release_time = self.config['schedule']['weekly_release_time']
        
        # Sunday: Research monitoring and preparation
        schedule.every().sunday.at("20:00").do(self._run_weekly_preparation)
        
        # Monday: Version preparation and review
        schedule.every().monday.at("09:00").do(self._run_version_preparation)
        
        # Tuesday: Release deployment
        if release_day.lower() == 'tuesday':
            schedule.every().tuesday.at(release_time).do(self._run_weekly_release)
        
        logger.info(f"Scheduled weekly release cycle for {release_day} at {release_time}")
    
    def _schedule_research_monitoring(self):
        """Schedule daily research monitoring."""
        schedule.every().day.at("06:00").do(self._run_research_monitoring)
        logger.info("Scheduled daily research monitoring at 06:00")
    
    def _schedule_status_checks(self):
        """Schedule regular status checks."""
        schedule.every(30).minutes.do(self._check_system_health)
        logger.info("Scheduled system health checks every 30 minutes")
    
    async def _run_weekly_preparation(self):
        """Run weekly preparation (Sunday)."""
        logger.info("Starting weekly preparation cycle")
        
        try:
            # Update research monitor status
            self.system_status['research_monitor'].status = 'running'
            self.system_status['research_monitor'].last_run = datetime.now()
            
            # Run research monitoring
            result = await self._execute_component('research_monitor', [
                'python', 'automation/research_monitor.py', 
                '--scan-weekly', '--generate-digest'
            ])
            
            if result['success']:
                # Evaluate papers for integration
                self.system_status['integration_pipeline'].status = 'running'
                
                integration_result = await self._execute_component('integration_pipeline', [
                    'python', 'automation/integration_pipeline.py', 
                    '--evaluate-papers'
                ])
                
                if integration_result['success']:
                    logger.info("Weekly preparation completed successfully")
                    self._send_notification(
                        "Weekly Preparation Complete",
                        "Research monitoring and integration evaluation completed successfully."
                    )
                else:
                    logger.error("Integration evaluation failed")
                    self._handle_component_error('integration_pipeline', integration_result['error'])
            else:
                logger.error("Research monitoring failed")
                self._handle_component_error('research_monitor', result['error'])
                
        except Exception as e:
            logger.error(f"Weekly preparation failed: {e}")
            self._handle_critical_error(e)
    
    async def _run_version_preparation(self):
        """Run version preparation (Monday)."""
        logger.info("Starting version preparation")
        
        try:
            self.system_status['version_manager'].status = 'running'
            self.system_status['version_manager'].last_run = datetime.now()
            
            # Prepare release version
            result = await self._execute_component('version_manager', [
                'python', 'automation/version_manager.py', 
                '--prepare-release'
            ])
            
            if result['success']:
                logger.info("Version preparation completed successfully")
                self._send_notification(
                    "Version Preparation Complete",
                    "Release version prepared and ready for Tuesday deployment."
                )
            else:
                logger.error("Version preparation failed")
                self._handle_component_error('version_manager', result['error'])
                
        except Exception as e:
            logger.error(f"Version preparation failed: {e}")
            self._handle_critical_error(e)
    
    async def _run_weekly_release(self):
        """Run weekly release (Tuesday)."""
        logger.info("Starting weekly release deployment")
        
        try:
            self.system_status['deployment_manager'].status = 'running'
            self.system_status['deployment_manager'].last_run = datetime.now()
            
            # Execute deployment
            result = await self._execute_component('deployment_manager', [
                'python', 'automation/deployment_manager.py', 
                '--schedule-weekly-release'
            ])
            
            if result['success']:
                # Finalize release
                finalize_result = await self._execute_component('version_manager', [
                    'python', 'automation/version_manager.py', 
                    '--finalize-release'
                ])
                
                if finalize_result['success']:
                    logger.info("Weekly release completed successfully")
                    self._send_notification(
                        "Weekly Release Complete",
                        "New version deployed successfully to production."
                    )
                else:
                    logger.error("Release finalization failed")
                    self._handle_component_error('version_manager', finalize_result['error'])
            else:
                logger.error("Weekly release deployment failed")
                self._handle_component_error('deployment_manager', result['error'])
                
        except Exception as e:
            logger.error(f"Weekly release failed: {e}")
            self._handle_critical_error(e)
    
    async def _run_research_monitoring(self):
        """Run daily research monitoring."""
        logger.info("Running daily research monitoring")
        
        try:
            self.system_status['research_monitor'].status = 'running'
            self.system_status['research_monitor'].last_run = datetime.now()
            
            result = await self._execute_component('research_monitor', [
                'python', 'automation/research_monitor.py', 
                '--scan-weekly'
            ])
            
            if result['success']:
                self.system_status['research_monitor'].status = 'completed'
                logger.info("Daily research monitoring completed")
            else:
                self._handle_component_error('research_monitor', result['error'])
                
        except Exception as e:
            logger.error(f"Daily research monitoring failed: {e}")
            self._handle_component_error('research_monitor', str(e))
    
    async def _execute_component(self, component_name: str, command: List[str]) -> Dict[str, Any]:
        """Execute a component command."""
        try:
            logger.info(f"Executing {component_name}: {' '.join(command)}")
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.system_status[component_name].status = 'completed'
                return {
                    'success': True,
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode()
                }
            else:
                self.system_status[component_name].status = 'error'
                self.system_status[component_name].error_message = stderr.decode()
                return {
                    'success': False,
                    'error': stderr.decode(),
                    'stdout': stdout.decode()
                }
                
        except Exception as e:
            self.system_status[component_name].status = 'error'
            self.system_status[component_name].error_message = str(e)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_system_health(self):
        """Check overall system health."""
        logger.debug("Checking system health")
        
        # Check component status
        error_components = []
        for component, status in self.system_status.items():
            if status.status == 'error':
                error_components.append(component)
        
        if error_components:
            logger.warning(f"Components in error state: {', '.join(error_components)}")
        
        # Check disk space
        disk_usage = self._check_disk_usage()
        if disk_usage > 90:
            logger.warning(f"High disk usage: {disk_usage}%")
        
        # Check log file sizes
        self._check_log_sizes()
    
    def _check_disk_usage(self) -> float:
        """Check disk usage percentage."""
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            return (used / total) * 100
        except Exception:
            return 0.0
    
    def _check_log_sizes(self):
        """Check and rotate log files if needed."""
        log_dir = Path('automation/logs')
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                size_mb = log_file.stat().st_size / (1024 * 1024)
                if size_mb > 50:  # 50MB limit
                    logger.warning(f"Large log file: {log_file} ({size_mb:.1f}MB)")
    
    def _handle_component_error(self, component: str, error_message: str):
        """Handle component error."""
        logger.error(f"Component {component} failed: {error_message}")
        
        self.system_status[component].status = 'error'
        self.system_status[component].error_message = error_message
        
        # Send error notification
        self._send_notification(
            f"Component Error: {component}",
            f"Component {component} encountered an error:\n\n{error_message}"
        )
        
        # Check if critical error
        if self.config['error_handling']['emergency_stop_on_critical']:
            critical_components = ['deployment_manager', 'version_manager']
            if component in critical_components:
                self.emergency_stop()
    
    def _handle_critical_error(self, error: Exception):
        """Handle critical system error."""
        logger.critical(f"Critical system error: {error}")
        
        self._send_notification(
            "CRITICAL ERROR - NeuroBM Automation",
            f"Critical system error occurred:\n\n{error}\n\nSystem may require manual intervention."
        )
        
        if self.config['error_handling']['emergency_stop_on_critical']:
            self.emergency_stop()
    
    def _send_notification(self, subject: str, message: str):
        """Send notification to administrators."""
        try:
            # Log notification
            logger.info(f"Notification: {subject}")
            
            # In a real implementation, this would send email/Slack notifications
            notification_file = Path('automation/logs/notifications.log')
            with open(notification_file, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {subject}: {message}\n\n")
                
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status_dict = {}
        for component, status in self.system_status.items():
            status_dict[component] = {
                'status': status.status,
                'last_run': status.last_run.isoformat() if status.last_run else None,
                'next_run': status.next_run.isoformat() if status.next_run else None,
                'error_message': status.error_message,
                'metrics': status.metrics
            }
        
        return {
            'orchestrator_status': 'running' if self.is_running else 'stopped',
            'emergency_stop': self.emergency_stop,
            'components': status_dict,
            'last_check': datetime.now().isoformat()
        }
    
    async def run_manual_cycle(self):
        """Run a complete cycle manually."""
        logger.info("Starting manual automation cycle")
        
        try:
            # Run research monitoring
            await self._run_research_monitoring()
            
            # Run version preparation
            await self._run_version_preparation()
            
            # Run weekly release
            await self._run_weekly_release()
            
            logger.info("Manual automation cycle completed")
            
        except Exception as e:
            logger.error(f"Manual cycle failed: {e}")
            raise


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuroBM Automation Orchestrator')
    parser.add_argument('--start-scheduler', action='store_true', help='Start the automation scheduler')
    parser.add_argument('--run-weekly-cycle', action='store_true', help='Run complete weekly cycle manually')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--emergency-stop', action='store_true', help='Emergency stop all automation')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    config_path = args.config or "automation/config/orchestrator_config.yaml"
    orchestrator = AutomationOrchestrator(config_path)
    
    if args.start_scheduler:
        print("🚀 Starting NeuroBM Automation Orchestrator")
        print("Press Ctrl+C to stop")
        orchestrator.start_scheduler()
    
    elif args.run_weekly_cycle:
        print("🔄 Running manual weekly cycle")
        await orchestrator.run_manual_cycle()
        print("✅ Manual cycle completed")
    
    elif args.status:
        status = orchestrator.get_system_status()
        print("📊 NeuroBM Automation Status")
        print("=" * 40)
        print(f"Orchestrator: {status['orchestrator_status']}")
        print(f"Emergency Stop: {status['emergency_stop']}")
        print("\nComponents:")
        
        for component, info in status['components'].items():
            print(f"  {component}:")
            print(f"    Status: {info['status']}")
            print(f"    Last Run: {info['last_run'] or 'Never'}")
            if info['error_message']:
                print(f"    Error: {info['error_message']}")
    
    elif args.emergency_stop:
        print("🛑 Activating emergency stop")
        orchestrator.emergency_stop()
        print("Emergency stop activated")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    asyncio.run(main())
