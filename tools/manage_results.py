#!/usr/bin/env python3
"""
Results Management Tool for NeuroBM.

This tool helps manage the results, logs, and checkpoints directories,
providing utilities for organization, cleanup, and analysis.

Features:
- List and organize experiment results
- Clean up old files and manage disk space
- Generate summary reports across experiments
- Archive and backup important results
- Validate result integrity and completeness

Usage:
    python tools/manage_results.py --list-experiments
    python tools/manage_results.py --cleanup --older-than=30
    python tools/manage_results.py --archive --experiment=my_study
    python tools/manage_results.py --summary --output=summary_report.html
"""

import argparse
import logging
import sys
import shutil
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultsManager:
    """Tool for managing NeuroBM results, logs, and checkpoints."""
    
    def __init__(self, project_root: Path):
        """Initialize results manager."""
        self.project_root = project_root
        self.results_dir = project_root / 'results'
        self.logs_dir = project_root / 'logs'
        self.checkpoints_dir = project_root / 'checkpoints'
    
    def list_experiments(self, detailed: bool = False) -> List[Dict[str, Any]]:
        """List all experiments with their status and metadata."""
        experiments = []
        experiments_dir = self.results_dir / 'experiments'
        
        if not experiments_dir.exists():
            logger.warning("No experiments directory found")
            return experiments
        
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_info = self._get_experiment_info(exp_dir, detailed)
                experiments.append(exp_info)
        
        return sorted(experiments, key=lambda x: x.get('created_at', ''))
    
    def _get_experiment_info(self, exp_dir: Path, detailed: bool = False) -> Dict[str, Any]:
        """Get information about a specific experiment."""
        info = {
            'name': exp_dir.name,
            'path': str(exp_dir),
            'created_at': None,
            'status': 'unknown',
            'size_mb': 0,
            'has_config': False,
            'has_results': False,
            'has_model': False
        }
        
        # Check for configuration
        config_file = exp_dir / 'config' / 'experiment.yaml'
        if config_file.exists():
            info['has_config'] = True
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    info['created_at'] = config.get('created_at')
                    info['status'] = config.get('status', 'unknown')
                    info['description'] = config.get('description', '')
            except Exception as e:
                logger.warning(f"Could not read config for {exp_dir.name}: {e}")
        
        # Check for results
        metrics_file = exp_dir / 'logs' / 'metrics.json'
        if metrics_file.exists():
            info['has_results'] = True
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    info['status'] = metrics.get('experiment_info', {}).get('status', 'unknown')
            except Exception as e:
                logger.warning(f"Could not read metrics for {exp_dir.name}: {e}")
        
        # Check for models
        models_dir = exp_dir / 'models'
        if models_dir.exists() and any(models_dir.glob('*.pth')):
            info['has_model'] = True
        
        # Calculate directory size
        info['size_mb'] = self._get_directory_size(exp_dir) / (1024 * 1024)
        
        if detailed:
            info.update(self._get_detailed_experiment_info(exp_dir))
        
        return info
    
    def _get_detailed_experiment_info(self, exp_dir: Path) -> Dict[str, Any]:
        """Get detailed information about an experiment."""
        detailed_info = {
            'files_count': 0,
            'plots_count': 0,
            'analysis_files': [],
            'model_files': [],
            'last_modified': None
        }
        
        # Count files
        detailed_info['files_count'] = len(list(exp_dir.rglob('*')))
        
        # Count plots
        plots_dir = exp_dir / 'plots'
        if plots_dir.exists():
            detailed_info['plots_count'] = len(list(plots_dir.glob('*.png')))
        
        # List analysis files
        analysis_dir = exp_dir / 'analysis'
        if analysis_dir.exists():
            detailed_info['analysis_files'] = [f.name for f in analysis_dir.iterdir() if f.is_file()]
        
        # List model files
        models_dir = exp_dir / 'models'
        if models_dir.exists():
            detailed_info['model_files'] = [f.name for f in models_dir.glob('*.pth')]
        
        # Get last modified time
        try:
            detailed_info['last_modified'] = datetime.fromtimestamp(exp_dir.stat().st_mtime).isoformat()
        except:
            pass
        
        return detailed_info
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not calculate size for {directory}: {e}")
        return total_size
    
    def cleanup_old_files(self, older_than_days: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up old files and directories."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        cleanup_summary = {
            'files_removed': 0,
            'directories_removed': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        # Clean up old logs
        logs_cleaned = self._cleanup_directory(
            self.logs_dir, cutoff_date, dry_run, 
            patterns=['*.log', '*.json'], exclude_patterns=['current*']
        )
        cleanup_summary['files_removed'] += logs_cleaned['files_removed']
        cleanup_summary['space_freed_mb'] += logs_cleaned['space_freed_mb']
        
        # Clean up old checkpoints (be more careful)
        checkpoints_cleaned = self._cleanup_directory(
            self.checkpoints_dir, cutoff_date, dry_run,
            patterns=['*.pth'], exclude_patterns=['*best*', '*final*']
        )
        cleanup_summary['files_removed'] += checkpoints_cleaned['files_removed']
        cleanup_summary['space_freed_mb'] += checkpoints_cleaned['space_freed_mb']
        
        # Clean up temporary experiment files
        temp_cleaned = self._cleanup_directory(
            self.results_dir, cutoff_date, dry_run,
            patterns=['*.tmp', '*.temp', '*~'], exclude_patterns=[]
        )
        cleanup_summary['files_removed'] += temp_cleaned['files_removed']
        cleanup_summary['space_freed_mb'] += temp_cleaned['space_freed_mb']
        
        return cleanup_summary
    
    def _cleanup_directory(
        self, 
        directory: Path, 
        cutoff_date: datetime, 
        dry_run: bool,
        patterns: List[str],
        exclude_patterns: List[str]
    ) -> Dict[str, Any]:
        """Clean up files in a specific directory."""
        summary = {'files_removed': 0, 'space_freed_mb': 0}
        
        if not directory.exists():
            return summary
        
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if not file_path.is_file():
                    continue
                
                # Check if file should be excluded
                if any(file_path.match(exclude) for exclude in exclude_patterns):
                    continue
                
                # Check file age
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_size = file_path.stat().st_size
                        
                        if not dry_run:
                            file_path.unlink()
                            logger.info(f"Removed: {file_path}")
                        else:
                            logger.info(f"Would remove: {file_path}")
                        
                        summary['files_removed'] += 1
                        summary['space_freed_mb'] += file_size / (1024 * 1024)
                
                except Exception as e:
                    logger.warning(f"Could not process {file_path}: {e}")
        
        return summary
    
    def archive_experiment(self, experiment_name: str, archive_dir: Optional[Path] = None) -> bool:
        """Archive an experiment to compressed storage."""
        exp_dir = self.results_dir / 'experiments' / experiment_name
        
        if not exp_dir.exists():
            logger.error(f"Experiment not found: {experiment_name}")
            return False
        
        if archive_dir is None:
            archive_dir = self.project_root / 'archives'
        
        archive_dir.mkdir(exist_ok=True)
        
        # Create compressed archive
        archive_path = archive_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            shutil.make_archive(str(archive_path), 'zip', str(exp_dir))
            logger.info(f"Archived experiment to: {archive_path}.zip")
            
            # Optionally remove original (with confirmation)
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive experiment: {e}")
            return False
    
    def generate_summary_report(self, output_file: Optional[Path] = None) -> str:
        """Generate a summary report of all experiments."""
        experiments = self.list_experiments(detailed=True)
        
        # Create summary statistics
        total_experiments = len(experiments)
        completed_experiments = len([e for e in experiments if e['status'] == 'completed'])
        total_size_mb = sum(e['size_mb'] for e in experiments)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuroBM Experiments Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e8f5e8; padding: 20px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>🧠 NeuroBM Experiments Summary</h1>
            
            <div class="summary">
                <h2>📊 Overview</h2>
                <p><strong>Total Experiments:</strong> {total_experiments}</p>
                <p><strong>Completed:</strong> {completed_experiments}</p>
                <p><strong>Total Storage:</strong> {total_size_mb:.1f} MB</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📋 Experiment Details</h2>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Size (MB)</th>
                    <th>Has Model</th>
                    <th>Has Results</th>
                </tr>
        """
        
        for exp in experiments:
            html_content += f"""
                <tr>
                    <td>{exp['name']}</td>
                    <td>{exp['status']}</td>
                    <td>{exp.get('created_at', 'Unknown')}</td>
                    <td>{exp['size_mb']:.1f}</td>
                    <td>{'✅' if exp['has_model'] else '❌'}</td>
                    <td>{'✅' if exp['has_results'] else '❌'}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
            logger.info(f"Summary report saved to: {output_file}")
        
        return html_content
    
    def validate_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Validate an experiment's completeness and integrity."""
        exp_dir = self.results_dir / 'experiments' / experiment_name
        
        validation_result = {
            'experiment': experiment_name,
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        if not exp_dir.exists():
            validation_result['valid'] = False
            validation_result['errors'].append(f"Experiment directory not found: {exp_dir}")
            return validation_result
        
        # Check required directories
        required_dirs = ['config', 'logs']
        for req_dir in required_dirs:
            dir_path = exp_dir / req_dir
            if dir_path.exists():
                validation_result['checks'][f'has_{req_dir}'] = True
            else:
                validation_result['checks'][f'has_{req_dir}'] = False
                validation_result['warnings'].append(f"Missing {req_dir} directory")
        
        # Check configuration file
        config_file = exp_dir / 'config' / 'experiment.yaml'
        if config_file.exists():
            validation_result['checks']['has_config'] = True
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                validation_result['checks']['config_valid'] = True
            except Exception as e:
                validation_result['checks']['config_valid'] = False
                validation_result['errors'].append(f"Invalid config file: {e}")
        else:
            validation_result['checks']['has_config'] = False
            validation_result['warnings'].append("No configuration file found")
        
        return validation_result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Manage NeuroBM results, logs, and checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--list-experiments', action='store_true', help='List all experiments')
    parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old files')
    parser.add_argument('--older-than', type=int, default=30, help='Clean files older than N days')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned without doing it')
    parser.add_argument('--archive', action='store_true', help='Archive an experiment')
    parser.add_argument('--experiment', type=str, help='Experiment name for operations')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')
    parser.add_argument('--output', type=str, help='Output file for reports')
    parser.add_argument('--validate', action='store_true', help='Validate experiment integrity')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create results manager
    manager = ResultsManager(project_root)
    
    if args.list_experiments:
        experiments = manager.list_experiments(detailed=args.detailed)
        
        print("🧪 NeuroBM Experiments")
        print("=" * 50)
        
        if not experiments:
            print("No experiments found.")
        else:
            for exp in experiments:
                print(f"\n📋 {exp['name']}")
                print(f"   Status: {exp['status']}")
                print(f"   Size: {exp['size_mb']:.1f} MB")
                print(f"   Config: {'✅' if exp['has_config'] else '❌'}")
                print(f"   Results: {'✅' if exp['has_results'] else '❌'}")
                print(f"   Model: {'✅' if exp['has_model'] else '❌'}")
                
                if args.detailed and 'files_count' in exp:
                    print(f"   Files: {exp['files_count']}")
                    print(f"   Plots: {exp['plots_count']}")
    
    elif args.cleanup:
        print(f"🧹 Cleaning up files older than {args.older_than} days...")
        if args.dry_run:
            print("🔍 DRY RUN - No files will be actually removed")
        
        summary = manager.cleanup_old_files(args.older_than, args.dry_run)
        
        print(f"\n📊 Cleanup Summary:")
        print(f"   Files removed: {summary['files_removed']}")
        print(f"   Space freed: {summary['space_freed_mb']:.1f} MB")
        
        if summary['errors']:
            print(f"   Errors: {len(summary['errors'])}")
    
    elif args.archive:
        if not args.experiment:
            print("❌ Please specify --experiment for archiving")
            sys.exit(1)
        
        print(f"📦 Archiving experiment: {args.experiment}")
        success = manager.archive_experiment(args.experiment)
        
        if success:
            print("✅ Archive created successfully")
        else:
            print("❌ Archive creation failed")
            sys.exit(1)
    
    elif args.summary:
        print("📊 Generating summary report...")
        
        output_file = Path(args.output) if args.output else None
        report = manager.generate_summary_report(output_file)
        
        if not output_file:
            print("📋 Summary Report:")
            # Print a text version
            experiments = manager.list_experiments()
            print(f"Total experiments: {len(experiments)}")
            completed = len([e for e in experiments if e['status'] == 'completed'])
            print(f"Completed: {completed}")
    
    elif args.validate:
        if not args.experiment:
            print("❌ Please specify --experiment for validation")
            sys.exit(1)
        
        print(f"🔍 Validating experiment: {args.experiment}")
        result = manager.validate_experiment(args.experiment)
        
        if result['valid']:
            print("✅ Experiment validation passed")
        else:
            print("❌ Experiment validation failed")
            for error in result['errors']:
                print(f"   Error: {error}")
        
        for warning in result['warnings']:
            print(f"   Warning: {warning}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
