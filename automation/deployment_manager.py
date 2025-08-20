#!/usr/bin/env python3
"""
Deployment Management System for NeuroBM.

This system handles the automated deployment pipeline, including
weekly release scheduling, quality assurance, and deployment coordination.

Features:
- Automated weekly release schedule (Tuesdays)
- Multi-stage deployment (dev -> staging -> production)
- Quality gates and validation
- Rollback capabilities
- Community notification system
- Performance monitoring

Usage:
    python automation/deployment_manager.py --schedule-weekly-release
    python automation/deployment_manager.py --deploy --stage=staging
    python automation/deployment_manager.py --rollback --stage=production
    python automation/deployment_manager.py --status
"""

import asyncio
import logging
import json
import yaml
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import git
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import schedule
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeploymentInfo:
    """Data class for deployment information."""
    version: str
    stage: str  # development, staging, production
    deployment_time: datetime
    status: str  # pending, in_progress, completed, failed, rolled_back
    quality_gates_passed: Dict[str, bool]
    performance_metrics: Dict[str, float]
    rollback_available: bool
    notification_sent: bool
    
    def __post_init__(self):
        if not self.quality_gates_passed:
            self.quality_gates_passed = {}
        if not self.performance_metrics:
            self.performance_metrics = {}


class DeploymentManager:
    """Manages automated deployments for NeuroBM."""
    
    def __init__(self, config_path: str = "automation/config/deployment_config.yaml"):
        """Initialize the deployment manager."""
        self.config = self._load_config(config_path)
        self.repo = git.Repo('.')
        self.deployment_history = self._load_deployment_history()
        
        # Initialize notification systems
        self._init_notification_systems()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'schedule': {
                    'release_day': 'tuesday',
                    'release_time': '10:00',
                    'timezone': 'UTC',
                    'auto_deploy': True
                },
                'stages': {
                    'development': {
                        'branch': 'dev',
                        'auto_deploy': True,
                        'quality_gates': ['unit_tests', 'integration_tests']
                    },
                    'staging': {
                        'branch': 'staging',
                        'auto_deploy': True,
                        'quality_gates': ['unit_tests', 'integration_tests', 'performance_tests']
                    },
                    'production': {
                        'branch': 'main',
                        'auto_deploy': False,
                        'quality_gates': ['all_tests', 'security_scan', 'performance_benchmark']
                    }
                },
                'quality_gates': {
                    'unit_tests': {
                        'command': 'python -m pytest tests/',
                        'timeout': 300,
                        'required': True
                    },
                    'integration_tests': {
                        'command': 'python test_neurobm_comprehensive.py',
                        'timeout': 180,
                        'required': True
                    },
                    'performance_tests': {
                        'command': 'python automation/performance_tests.py',
                        'timeout': 600,
                        'required': True
                    }
                },
                'notifications': {
                    'email': {
                        'enabled': True,
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'recipients': ['team@neurobm.org']
                    },
                    'slack': {
                        'enabled': False,
                        'webhook_url': ''
                    },
                    'github': {
                        'enabled': True,
                        'create_release': True
                    }
                },
                'rollback': {
                    'auto_rollback_on_failure': True,
                    'max_rollback_time_hours': 24,
                    'require_approval': True
                }
            }
    
    def _load_deployment_history(self) -> List[DeploymentInfo]:
        """Load deployment history."""
        history_file = Path('automation/data/deployment_history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                deployments = []
                for deployment in history_data:
                    deployment['deployment_time'] = datetime.fromisoformat(deployment['deployment_time'])
                    deployments.append(DeploymentInfo(**deployment))
                return deployments
        return []
    
    def _save_deployment_history(self):
        """Save deployment history."""
        history_file = Path('automation/data/deployment_history.json')
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        history_data = []
        for deployment in self.deployment_history:
            deployment_dict = asdict(deployment)
            deployment_dict['deployment_time'] = deployment.deployment_time.isoformat()
            history_data.append(deployment_dict)
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _init_notification_systems(self):
        """Initialize notification systems."""
        # Email configuration
        if self.config['notifications']['email']['enabled']:
            self.email_config = self.config['notifications']['email']
        
        # Slack configuration
        if self.config['notifications']['slack']['enabled']:
            self.slack_webhook = self.config['notifications']['slack']['webhook_url']
    
    def schedule_weekly_releases(self):
        """Set up weekly release schedule."""
        release_day = self.config['schedule']['release_day']
        release_time = self.config['schedule']['release_time']
        
        # Schedule the weekly release
        if release_day.lower() == 'tuesday':
            schedule.every().tuesday.at(release_time).do(self._execute_weekly_release)
        elif release_day.lower() == 'monday':
            schedule.every().monday.at(release_time).do(self._execute_weekly_release)
        # Add other days as needed
        
        # Schedule pre-release activities
        # Sunday: Automated builds and testing
        schedule.every().sunday.at("20:00").do(self._prepare_weekly_release)
        
        # Monday: Review and approval
        schedule.every().monday.at("09:00").do(self._review_weekly_release)
        
        logger.info(f"Scheduled weekly releases for {release_day} at {release_time}")
    
    def _prepare_weekly_release(self):
        """Prepare weekly release (Sunday)."""
        logger.info("Preparing weekly release...")
        
        try:
            # Run research monitoring
            subprocess.run([
                'python', 'automation/research_monitor.py', 
                '--scan-weekly', '--generate-digest'
            ], check=True)
            
            # Evaluate papers for integration
            subprocess.run([
                'python', 'automation/integration_pipeline.py', 
                '--evaluate-papers'
            ], check=True)
            
            # Run comprehensive tests
            test_results = self._run_quality_gates(['unit_tests', 'integration_tests', 'performance_tests'])
            
            # Generate release preparation report
            self._generate_release_preparation_report(test_results)
            
            # Send notification
            self._send_notification(
                "Weekly Release Preparation Complete",
                "Automated testing and preparation completed. Ready for Monday review."
            )
            
        except Exception as e:
            logger.error(f"Weekly release preparation failed: {e}")
            self._send_notification(
                "Weekly Release Preparation Failed",
                f"Automated preparation failed: {e}"
            )
    
    def _review_weekly_release(self):
        """Review weekly release (Monday)."""
        logger.info("Starting weekly release review...")
        
        # This would trigger human review process
        # For now, just send notification
        self._send_notification(
            "Weekly Release Review Required",
            "Please review the prepared release and approve for Tuesday deployment."
        )
    
    def _execute_weekly_release(self):
        """Execute weekly release (Tuesday)."""
        logger.info("Executing weekly release...")
        
        try:
            # Prepare release
            from automation.version_manager import VersionManager
            version_manager = VersionManager()
            release_info = version_manager.prepare_release()
            
            # Deploy to staging first
            staging_success = self.deploy_to_stage('staging', release_info.version)
            
            if staging_success:
                # If staging successful, deploy to production
                production_success = self.deploy_to_stage('production', release_info.version)
                
                if production_success:
                    # Finalize release
                    version_manager.finalize_release(release_info)
                    
                    # Send success notification
                    self._send_release_notification(release_info, 'success')
                else:
                    # Production deployment failed
                    self._send_release_notification(release_info, 'production_failed')
            else:
                # Staging deployment failed
                self._send_release_notification(release_info, 'staging_failed')
                
        except Exception as e:
            logger.error(f"Weekly release execution failed: {e}")
            self._send_notification(
                "Weekly Release Failed",
                f"Release execution failed: {e}"
            )
    
    def deploy_to_stage(self, stage: str, version: str) -> bool:
        """Deploy to a specific stage."""
        logger.info(f"Deploying version {version} to {stage}")
        
        # Create deployment info
        deployment = DeploymentInfo(
            version=version,
            stage=stage,
            deployment_time=datetime.now(),
            status='in_progress',
            quality_gates_passed={},
            performance_metrics={},
            rollback_available=True,
            notification_sent=False
        )
        
        try:
            # Switch to appropriate branch
            stage_config = self.config['stages'][stage]
            target_branch = stage_config['branch']
            
            self.repo.git.checkout(target_branch)
            
            # Merge release branch if needed
            if stage in ['staging', 'production']:
                release_branch = f"release/{version}"
                try:
                    self.repo.git.merge(release_branch)
                except git.exc.GitCommandError as e:
                    logger.error(f"Merge failed: {e}")
                    deployment.status = 'failed'
                    return False
            
            # Run quality gates
            quality_gates = stage_config['quality_gates']
            gate_results = self._run_quality_gates(quality_gates)
            deployment.quality_gates_passed = gate_results
            
            # Check if all required gates passed
            if not all(gate_results.values()):
                logger.error(f"Quality gates failed for {stage}")
                deployment.status = 'failed'
                
                # Auto-rollback if configured
                if self.config['rollback']['auto_rollback_on_failure']:
                    self._auto_rollback(stage)
                
                return False
            
            # Run performance benchmarks
            performance_metrics = self._run_performance_benchmarks()
            deployment.performance_metrics = performance_metrics
            
            # Tag deployment
            if stage == 'production':
                tag_name = f"v{version}"
                self.repo.create_tag(tag_name, message=f"Production release {version}")
            
            # Update deployment status
            deployment.status = 'completed'
            
            # Store deployment info
            self.deployment_history.append(deployment)
            self._save_deployment_history()
            
            # Send deployment notification
            self._send_deployment_notification(deployment)
            
            logger.info(f"Successfully deployed {version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment to {stage} failed: {e}")
            deployment.status = 'failed'
            self.deployment_history.append(deployment)
            self._save_deployment_history()
            return False
    
    def _run_quality_gates(self, gates: List[str]) -> Dict[str, bool]:
        """Run quality gate tests."""
        results = {}
        
        for gate in gates:
            if gate not in self.config['quality_gates']:
                logger.warning(f"Unknown quality gate: {gate}")
                results[gate] = False
                continue
            
            gate_config = self.config['quality_gates'][gate]
            command = gate_config['command']
            timeout = gate_config.get('timeout', 300)
            
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                results[gate] = result.returncode == 0
                
                if result.returncode != 0:
                    logger.error(f"Quality gate {gate} failed: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                logger.error(f"Quality gate {gate} timed out")
                results[gate] = False
            except Exception as e:
                logger.error(f"Quality gate {gate} error: {e}")
                results[gate] = False
        
        return results
    
    def _run_performance_benchmarks(self) -> Dict[str, float]:
        """Run performance benchmarks."""
        metrics = {}
        
        try:
            # Run basic performance test
            result = subprocess.run([
                'python', '-c', 
                '''
import time
import torch
from neurobm.models.rbm import RestrictedBoltzmannMachine

# Simple performance test
start_time = time.time()
rbm = RestrictedBoltzmannMachine(n_visible=100, n_hidden=200)
data = torch.rand(1000, 100)

train_start = time.time()
for _ in range(10):
    rbm.train_batch(data[:32], k=1)
train_time = time.time() - train_start

total_time = time.time() - start_time
print(f"init_time:{start_time}")
print(f"train_time:{train_time}")
print(f"total_time:{total_time}")
                '''
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Parse performance metrics
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            metrics[key] = float(value)
                        except ValueError:
                            pass
            
        except Exception as e:
            logger.warning(f"Performance benchmark failed: {e}")
        
        return metrics
    
    def _auto_rollback(self, stage: str):
        """Automatically rollback failed deployment."""
        logger.info(f"Auto-rolling back {stage} deployment")
        
        # Find last successful deployment for this stage
        last_successful = None
        for deployment in reversed(self.deployment_history):
            if deployment.stage == stage and deployment.status == 'completed':
                last_successful = deployment
                break
        
        if last_successful:
            self.rollback_stage(stage, last_successful.version)
        else:
            logger.error(f"No previous successful deployment found for {stage}")
    
    def rollback_stage(self, stage: str, target_version: str) -> bool:
        """Rollback a stage to a specific version."""
        logger.info(f"Rolling back {stage} to version {target_version}")
        
        try:
            # Switch to stage branch
            stage_config = self.config['stages'][stage]
            target_branch = stage_config['branch']
            
            self.repo.git.checkout(target_branch)
            
            # Reset to target version tag
            tag_name = f"v{target_version}"
            self.repo.git.reset('--hard', tag_name)
            
            # Create rollback deployment record
            rollback_deployment = DeploymentInfo(
                version=target_version,
                stage=stage,
                deployment_time=datetime.now(),
                status='rolled_back',
                quality_gates_passed={},
                performance_metrics={},
                rollback_available=False,
                notification_sent=False
            )
            
            self.deployment_history.append(rollback_deployment)
            self._save_deployment_history()
            
            # Send rollback notification
            self._send_notification(
                f"Rollback Completed: {stage}",
                f"Successfully rolled back {stage} to version {target_version}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _send_notification(self, subject: str, message: str):
        """Send notification via configured channels."""
        # Email notification
        if self.config['notifications']['email']['enabled']:
            self._send_email_notification(subject, message)
        
        # Slack notification
        if self.config['notifications']['slack']['enabled']:
            self._send_slack_notification(subject, message)
    
    def _send_email_notification(self, subject: str, message: str):
        """Send email notification."""
        try:
            email_config = self.config['notifications']['email']
            
            msg = MimeMultipart()
            msg['From'] = email_config.get('from_address', 'neurobm@automated.system')
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[NeuroBM] {subject}"
            
            msg.attach(MimeText(message, 'plain'))
            
            # Note: This would need proper SMTP configuration
            logger.info(f"Email notification: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack_notification(self, subject: str, message: str):
        """Send Slack notification."""
        try:
            webhook_url = self.slack_webhook
            payload = {
                'text': f"*{subject}*\n{message}"
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_deployment_notification(self, deployment: DeploymentInfo):
        """Send deployment-specific notification."""
        if deployment.status == 'completed':
            subject = f"Deployment Successful: {deployment.stage} v{deployment.version}"
            message = f"""
Deployment completed successfully:

Version: {deployment.version}
Stage: {deployment.stage}
Time: {deployment.deployment_time.strftime('%Y-%m-%d %H:%M:%S')}

Quality Gates: {', '.join([f"{k}: {'✅' if v else '❌'}" for k, v in deployment.quality_gates_passed.items()])}

Performance Metrics:
{chr(10).join([f"- {k}: {v:.3f}s" for k, v in deployment.performance_metrics.items()])}
"""
        else:
            subject = f"Deployment Failed: {deployment.stage} v{deployment.version}"
            message = f"""
Deployment failed:

Version: {deployment.version}
Stage: {deployment.stage}
Time: {deployment.deployment_time.strftime('%Y-%m-%d %H:%M:%S')}

Failed Quality Gates: {', '.join([k for k, v in deployment.quality_gates_passed.items() if not v])}
"""
        
        self._send_notification(subject, message)
        deployment.notification_sent = True
    
    def _send_release_notification(self, release_info, status: str):
        """Send release-specific notification."""
        if status == 'success':
            subject = f"Weekly Release Successful: v{release_info.version}"
            message = f"""
Weekly release completed successfully!

Version: {release_info.version}
Release Type: {release_info.release_type}
Changes: {len(release_info.changes)}
Breaking Changes: {len(release_info.breaking_changes)}

The new version is now available in production.
"""
        else:
            subject = f"Weekly Release Failed: v{release_info.version}"
            message = f"""
Weekly release failed during {status.replace('_', ' ')}.

Version: {release_info.version}
Release Type: {release_info.release_type}

Please review the deployment logs and take appropriate action.
"""
        
        self._send_notification(subject, message)
    
    def _generate_release_preparation_report(self, test_results: Dict[str, bool]):
        """Generate release preparation report."""
        report_dir = Path('automation/reports/release_preparation')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"preparation_report_{timestamp}.md"
        
        report = f"""# Weekly Release Preparation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Results

"""
        
        for test, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            report += f"- **{test}**: {status}\n"
        
        report += f"""

## Summary

- Total Tests: {len(test_results)}
- Passed: {sum(test_results.values())}
- Failed: {len(test_results) - sum(test_results.values())}

## Next Steps

{'✅ Ready for Monday review and Tuesday release' if all(test_results.values()) else '⚠️ Issues found - review required before release'}
"""
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Release preparation report generated: {report_file}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        status = {
            'current_versions': {},
            'last_deployments': {},
            'pending_releases': [],
            'system_health': 'unknown'
        }
        
        # Get current version for each stage
        for stage in ['development', 'staging', 'production']:
            last_deployment = None
            for deployment in reversed(self.deployment_history):
                if deployment.stage == stage and deployment.status == 'completed':
                    last_deployment = deployment
                    break
            
            if last_deployment:
                status['current_versions'][stage] = last_deployment.version
                status['last_deployments'][stage] = {
                    'version': last_deployment.version,
                    'time': last_deployment.deployment_time.isoformat(),
                    'status': last_deployment.status
                }
        
        return status


def main():
    """Main function for deployment management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuroBM Deployment Manager')
    parser.add_argument('--schedule-weekly-release', action='store_true', help='Set up weekly release schedule')
    parser.add_argument('--deploy', action='store_true', help='Deploy to stage')
    parser.add_argument('--stage', type=str, choices=['development', 'staging', 'production'], help='Deployment stage')
    parser.add_argument('--version', type=str, help='Version to deploy')
    parser.add_argument('--rollback', action='store_true', help='Rollback stage')
    parser.add_argument('--to-version', type=str, help='Target version for rollback')
    parser.add_argument('--status', action='store_true', help='Show deployment status')
    parser.add_argument('--run-scheduler', action='store_true', help='Run the deployment scheduler')
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.schedule_weekly_release:
        manager.schedule_weekly_releases()
        print("Weekly release schedule configured")
    
    elif args.deploy:
        if not args.stage or not args.version:
            print("Please specify --stage and --version for deployment")
            return
        
        success = manager.deploy_to_stage(args.stage, args.version)
        if success:
            print(f"✅ Successfully deployed v{args.version} to {args.stage}")
        else:
            print(f"❌ Deployment to {args.stage} failed")
    
    elif args.rollback:
        if not args.stage or not args.to_version:
            print("Please specify --stage and --to-version for rollback")
            return
        
        success = manager.rollback_stage(args.stage, args.to_version)
        if success:
            print(f"✅ Successfully rolled back {args.stage} to v{args.to_version}")
        else:
            print(f"❌ Rollback failed")
    
    elif args.status:
        status = manager.get_deployment_status()
        print("🚀 NeuroBM Deployment Status")
        print("=" * 40)
        
        print("\nCurrent Versions:")
        for stage, version in status['current_versions'].items():
            print(f"  {stage}: v{version}")
        
        print("\nLast Deployments:")
        for stage, info in status['last_deployments'].items():
            print(f"  {stage}: v{info['version']} ({info['time']}) - {info['status']}")
    
    elif args.run_scheduler:
        print("Starting deployment scheduler...")
        manager.schedule_weekly_releases()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
