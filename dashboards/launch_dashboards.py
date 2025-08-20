#!/usr/bin/env python3
"""
Dashboard Launcher for NeuroBM.

This script provides a unified interface to launch all NeuroBM dashboards
with proper configuration and management.

Available Dashboards:
- Training Monitor: Real-time training visualization
- Model Explorer: Interactive model analysis
- Results Analyzer: Multi-experiment comparison

Usage:
    python dashboards/launch_dashboards.py --all
    python dashboards/launch_dashboards.py --training --port=8050
    python dashboards/launch_dashboards.py --explorer --checkpoint=model.pth
    python dashboards/launch_dashboards.py --analyzer --results_dir=results/
"""

import argparse
import logging
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import List, Dict, Any
import threading
import signal
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dashboard configurations
DASHBOARDS = {
    'training': {
        'script': 'training_monitor.py',
        'default_port': 8050,
        'description': 'Real-time training monitoring and visualization'
    },
    'explorer': {
        'script': 'model_explorer.py', 
        'default_port': 8051,
        'description': 'Interactive model exploration and analysis'
    },
    'analyzer': {
        'script': 'results_analyzer.py',
        'default_port': 8052,
        'description': 'Multi-experiment results comparison and analysis'
    }
}

# Global process tracking
running_processes = []


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal, terminating dashboards...")
    terminate_all_processes()
    sys.exit(0)


def terminate_all_processes():
    """Terminate all running dashboard processes."""
    for process in running_processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception as e:
            logger.warning(f"Error terminating process: {e}")
    
    running_processes.clear()
    logger.info("All dashboard processes terminated")


def launch_dashboard(dashboard_name: str, **kwargs) -> subprocess.Popen:
    """Launch a specific dashboard."""
    if dashboard_name not in DASHBOARDS:
        raise ValueError(f"Unknown dashboard: {dashboard_name}")
    
    config = DASHBOARDS[dashboard_name]
    script_path = Path(__file__).parent / config['script']
    
    if not script_path.exists():
        raise FileNotFoundError(f"Dashboard script not found: {script_path}")
    
    # Build command
    cmd = [sys.executable, str(script_path)]
    
    # Add common arguments
    port = kwargs.get('port', config['default_port'])
    host = kwargs.get('host', '127.0.0.1')
    
    cmd.extend(['--port', str(port)])
    cmd.extend(['--host', host])
    
    if kwargs.get('debug', False):
        cmd.append('--debug')
    
    # Add dashboard-specific arguments
    if dashboard_name == 'training':
        if 'config' in kwargs:
            cmd.extend(['--config', kwargs['config']])
    
    elif dashboard_name == 'explorer':
        if 'checkpoint' in kwargs:
            cmd.extend(['--checkpoint', kwargs['checkpoint']])
    
    elif dashboard_name == 'analyzer':
        if 'results_dir' in kwargs:
            cmd.extend(['--results_dir', kwargs['results_dir']])
    
    # Launch process
    logger.info(f"Launching {dashboard_name} dashboard on http://{host}:{port}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        running_processes.append(process)
        
        # Give the process a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"✅ {dashboard_name.title()} dashboard started successfully")
            
            # Open browser if requested
            if kwargs.get('open_browser', False):
                url = f"http://{host}:{port}"
                webbrowser.open(url)
                logger.info(f"🌐 Opened browser to {url}")
            
            return process
        else:
            # Process failed to start
            stdout, stderr = process.communicate()
            logger.error(f"❌ Failed to start {dashboard_name} dashboard")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error launching {dashboard_name} dashboard: {e}")
        return None


def monitor_processes():
    """Monitor running processes and restart if needed."""
    while True:
        time.sleep(10)  # Check every 10 seconds
        
        for i, process in enumerate(running_processes[:]):  # Copy list to avoid modification during iteration
            if process.poll() is not None:
                # Process has terminated
                logger.warning(f"Dashboard process {i} has terminated unexpectedly")
                running_processes.remove(process)
        
        if not running_processes:
            logger.info("No dashboard processes running, exiting monitor")
            break


def print_dashboard_info():
    """Print information about available dashboards."""
    print("\n🚀 NeuroBM Dashboard Launcher")
    print("=" * 50)
    print("\nAvailable Dashboards:")
    
    for name, config in DASHBOARDS.items():
        print(f"\n📊 {name.upper()}")
        print(f"   Description: {config['description']}")
        print(f"   Default Port: {config['default_port']}")
        print(f"   Script: {config['script']}")
    
    print("\n" + "=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Launch NeuroBM dashboards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Dashboard selection
    parser.add_argument('--all', action='store_true', help='Launch all dashboards')
    parser.add_argument('--training', action='store_true', help='Launch training monitor')
    parser.add_argument('--explorer', action='store_true', help='Launch model explorer')
    parser.add_argument('--analyzer', action='store_true', help='Launch results analyzer')
    
    # Common options
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run on')
    parser.add_argument('--port', type=int, help='Port to run on (overrides defaults)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--open-browser', action='store_true', help='Open browser automatically')
    parser.add_argument('--no-monitor', action='store_true', help='Disable process monitoring')
    
    # Dashboard-specific options
    parser.add_argument('--config', type=str, help='Config file for training monitor')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint for explorer')
    parser.add_argument('--results-dir', type=str, default='results/', help='Results directory for analyzer')
    
    # Utility options
    parser.add_argument('--list', action='store_true', help='List available dashboards')
    parser.add_argument('--kill-all', action='store_true', help='Kill all running dashboard processes')
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list:
        print_dashboard_info()
        return
    
    if args.kill_all:
        logger.info("Attempting to kill all dashboard processes...")
        # This is a simple implementation - in practice you might want to track PIDs
        try:
            subprocess.run(['pkill', '-f', 'training_monitor.py'], check=False)
            subprocess.run(['pkill', '-f', 'model_explorer.py'], check=False)
            subprocess.run(['pkill', '-f', 'results_analyzer.py'], check=False)
            logger.info("✅ Kill commands sent")
        except Exception as e:
            logger.error(f"❌ Error killing processes: {e}")
        return
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Determine which dashboards to launch
    dashboards_to_launch = []
    
    if args.all:
        dashboards_to_launch = list(DASHBOARDS.keys())
    else:
        if args.training:
            dashboards_to_launch.append('training')
        if args.explorer:
            dashboards_to_launch.append('explorer')
        if args.analyzer:
            dashboards_to_launch.append('analyzer')
    
    if not dashboards_to_launch:
        print("❌ No dashboards specified. Use --help for options.")
        return
    
    # Launch dashboards
    logger.info(f"Launching {len(dashboards_to_launch)} dashboard(s): {', '.join(dashboards_to_launch)}")
    
    for i, dashboard in enumerate(dashboards_to_launch):
        # Calculate port
        if args.port:
            port = args.port + i
        else:
            port = DASHBOARDS[dashboard]['default_port']
        
        # Prepare kwargs
        kwargs = {
            'host': args.host,
            'port': port,
            'debug': args.debug,
            'open_browser': args.open_browser and i == 0  # Only open browser for first dashboard
        }
        
        # Add dashboard-specific arguments
        if dashboard == 'training' and args.config:
            kwargs['config'] = args.config
        elif dashboard == 'explorer' and args.checkpoint:
            kwargs['checkpoint'] = args.checkpoint
        elif dashboard == 'analyzer' and args.results_dir:
            kwargs['results_dir'] = args.results_dir
        
        # Launch dashboard
        process = launch_dashboard(dashboard, **kwargs)
        
        if process is None:
            logger.error(f"Failed to launch {dashboard} dashboard")
            continue
        
        # Small delay between launches
        if i < len(dashboards_to_launch) - 1:
            time.sleep(1)
    
    if not running_processes:
        logger.error("❌ No dashboards started successfully")
        return
    
    # Print summary
    print(f"\n✅ Successfully launched {len(running_processes)} dashboard(s)")
    print("\n🌐 Dashboard URLs:")
    for i, dashboard in enumerate(dashboards_to_launch[:len(running_processes)]):
        port = args.port + i if args.port else DASHBOARDS[dashboard]['default_port']
        print(f"   {dashboard.title()}: http://{args.host}:{port}")
    
    print(f"\n💡 Press Ctrl+C to stop all dashboards")
    
    # Start monitoring if not disabled
    if not args.no_monitor and len(running_processes) > 1:
        monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
        monitor_thread.start()
    
    # Wait for processes
    try:
        # Wait for all processes to complete
        for process in running_processes:
            process.wait()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        terminate_all_processes()


if __name__ == '__main__':
    main()
