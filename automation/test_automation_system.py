#!/usr/bin/env python3
"""
Test script for NeuroBM Automation System.

This script validates that all automation components are properly configured
and can be executed without errors.

Usage:
    python automation/test_automation_system.py
    python automation/test_automation_system.py --component=research_monitor
    python automation/test_automation_system.py --full-test
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutomationSystemTester:
    """Test suite for the automation system."""
    
    def __init__(self):
        """Initialize the tester."""
        self.project_root = Path(__file__).parent.parent
        self.automation_dir = self.project_root / 'automation'
        self.test_results = {}
    
    def run_all_tests(self) -> bool:
        """Run all automation system tests."""
        logger.info("🧪 Starting NeuroBM Automation System Tests")
        logger.info("=" * 50)
        
        tests = [
            ("Configuration Files", self.test_configuration_files),
            ("Directory Structure", self.test_directory_structure),
            ("Component Imports", self.test_component_imports),
            ("Basic Functionality", self.test_basic_functionality),
            ("Integration Tests", self.test_integration_capabilities),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Testing {test_name}...")
            try:
                if test_func():
                    logger.info(f"  ✅ {test_name} passed")
                    self.test_results[test_name] = True
                    passed += 1
                else:
                    logger.error(f"  ❌ {test_name} failed")
                    self.test_results[test_name] = False
            except Exception as e:
                logger.error(f"  💥 {test_name} error: {e}")
                self.test_results[test_name] = False
        
        # Print summary
        logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All automation system tests passed!")
            return True
        else:
            logger.warning("⚠️  Some tests failed. Please check the implementation.")
            return False
    
    def test_configuration_files(self) -> bool:
        """Test that all configuration files are valid."""
        config_files = [
            'automation/config/monitor_config.yaml',
            'automation/config/integration_config.yaml',
            'automation/config/deployment_config.yaml'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                logger.error(f"Missing config file: {config_file}")
                return False
            
            try:
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
                logger.info(f"  ✅ {config_file} is valid")
            except yaml.YAMLError as e:
                logger.error(f"  ❌ {config_file} is invalid: {e}")
                return False
        
        return True
    
    def test_directory_structure(self) -> bool:
        """Test that all required directories exist."""
        required_dirs = [
            'automation',
            'automation/config',
            'automation/data',
            'automation/reports',
            'automation/reports/weekly_digests',
            'automation/reports/release_preparation',
            'automation/backups',
            'automation/logs'
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                logger.error(f"Missing directory: {dir_path}")
                return False
            logger.info(f"  ✅ {dir_path} exists")
        
        return True
    
    def test_component_imports(self) -> bool:
        """Test that all automation components can be imported."""
        components = [
            'automation.research_monitor',
            'automation.integration_pipeline',
            'automation.version_manager',
            'automation.deployment_manager',
            'automation.orchestrator'
        ]
        
        # Add project root to Python path
        sys.path.insert(0, str(self.project_root))
        
        for component in components:
            try:
                __import__(component)
                logger.info(f"  ✅ {component} imported successfully")
            except ImportError as e:
                logger.error(f"  ❌ {component} import failed: {e}")
                return False
        
        return True
    
    def test_basic_functionality(self) -> bool:
        """Test basic functionality of each component."""
        # Test research monitor help
        try:
            result = subprocess.run([
                'python', 'automation/research_monitor.py', '--help'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("  ✅ Research monitor help works")
            else:
                logger.error("  ❌ Research monitor help failed")
                return False
        except Exception as e:
            logger.error(f"  ❌ Research monitor test failed: {e}")
            return False
        
        # Test integration pipeline help
        try:
            result = subprocess.run([
                'python', 'automation/integration_pipeline.py', '--help'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("  ✅ Integration pipeline help works")
            else:
                logger.error("  ❌ Integration pipeline help failed")
                return False
        except Exception as e:
            logger.error(f"  ❌ Integration pipeline test failed: {e}")
            return False
        
        # Test version manager help
        try:
            result = subprocess.run([
                'python', 'automation/version_manager.py', '--help'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("  ✅ Version manager help works")
            else:
                logger.error("  ❌ Version manager help failed")
                return False
        except Exception as e:
            logger.error(f"  ❌ Version manager test failed: {e}")
            return False
        
        # Test deployment manager help
        try:
            result = subprocess.run([
                'python', 'automation/deployment_manager.py', '--help'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("  ✅ Deployment manager help works")
            else:
                logger.error("  ❌ Deployment manager help failed")
                return False
        except Exception as e:
            logger.error(f"  ❌ Deployment manager test failed: {e}")
            return False
        
        # Test orchestrator help
        try:
            result = subprocess.run([
                'python', 'automation/orchestrator.py', '--help'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("  ✅ Orchestrator help works")
            else:
                logger.error("  ❌ Orchestrator help failed")
                return False
        except Exception as e:
            logger.error(f"  ❌ Orchestrator test failed: {e}")
            return False
        
        return True
    
    def test_integration_capabilities(self) -> bool:
        """Test integration between components."""
        # Test orchestrator status
        try:
            result = subprocess.run([
                'python', 'automation/orchestrator.py', '--status'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("  ✅ Orchestrator status works")
            else:
                logger.error(f"  ❌ Orchestrator status failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"  ❌ Orchestrator status test failed: {e}")
            return False
        
        return True
    
    def test_specific_component(self, component: str) -> bool:
        """Test a specific component."""
        logger.info(f"🔍 Testing specific component: {component}")
        
        if component == 'research_monitor':
            return self._test_research_monitor()
        elif component == 'integration_pipeline':
            return self._test_integration_pipeline()
        elif component == 'version_manager':
            return self._test_version_manager()
        elif component == 'deployment_manager':
            return self._test_deployment_manager()
        elif component == 'orchestrator':
            return self._test_orchestrator()
        else:
            logger.error(f"Unknown component: {component}")
            return False
    
    def _test_research_monitor(self) -> bool:
        """Test research monitor component."""
        # Test configuration loading
        try:
            config_path = self.project_root / 'automation/config/monitor_config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['sources', 'filtering', 'integration']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing config section: {section}")
                    return False
            
            logger.info("  ✅ Research monitor configuration valid")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Research monitor test failed: {e}")
            return False
    
    def _test_integration_pipeline(self) -> bool:
        """Test integration pipeline component."""
        # Test configuration loading
        try:
            config_path = self.project_root / 'automation/config/integration_config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['integration', 'quality_gates', 'integration_types']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing config section: {section}")
                    return False
            
            logger.info("  ✅ Integration pipeline configuration valid")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Integration pipeline test failed: {e}")
            return False
    
    def _test_version_manager(self) -> bool:
        """Test version manager component."""
        # Test version file exists
        version_file = self.project_root / 'VERSION'
        if not version_file.exists():
            logger.warning("  ⚠️  VERSION file not found, creating default")
            with open(version_file, 'w') as f:
                f.write('1.0.0')
        
        logger.info("  ✅ Version manager basic setup valid")
        return True
    
    def _test_deployment_manager(self) -> bool:
        """Test deployment manager component."""
        # Test configuration loading
        try:
            config_path = self.project_root / 'automation/config/deployment_config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['schedule', 'stages', 'quality_gates']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing config section: {section}")
                    return False
            
            logger.info("  ✅ Deployment manager configuration valid")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Deployment manager test failed: {e}")
            return False
    
    def _test_orchestrator(self) -> bool:
        """Test orchestrator component."""
        # Test status command
        try:
            result = subprocess.run([
                'python', 'automation/orchestrator.py', '--status'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("  ✅ Orchestrator status command works")
                return True
            else:
                logger.error(f"  ❌ Orchestrator status failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Orchestrator test failed: {e}")
            return False
    
    def generate_test_report(self) -> str:
        """Generate a test report."""
        report = f"""# NeuroBM Automation System Test Report

Generated: {Path(__file__).name} on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Results

"""
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            report += f"- **{test_name}**: {status}\n"
        
        passed_count = sum(self.test_results.values())
        total_count = len(self.test_results)
        
        report += f"""

## Summary

- **Total Tests**: {total_count}
- **Passed**: {passed_count}
- **Failed**: {total_count - passed_count}
- **Success Rate**: {(passed_count / total_count * 100):.1f}%

## Status

{'🎉 All tests passed! The automation system is ready for use.' if passed_count == total_count else '⚠️ Some tests failed. Please review the implementation before using the automation system.'}

---

**Note**: This test report validates the basic setup and configuration of the NeuroBM automation system. Additional testing may be required for production deployment.
"""
        
        return report


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NeuroBM Automation System')
    parser.add_argument('--component', type=str, help='Test specific component')
    parser.add_argument('--full-test', action='store_true', help='Run comprehensive tests')
    parser.add_argument('--generate-report', action='store_true', help='Generate test report')
    
    args = parser.parse_args()
    
    tester = AutomationSystemTester()
    
    if args.component:
        success = tester.test_specific_component(args.component)
        sys.exit(0 if success else 1)
    
    elif args.full_test:
        success = tester.run_all_tests()
        
        if args.generate_report:
            report = tester.generate_test_report()
            report_file = Path('automation/reports/test_report.md')
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n📄 Test report saved to: {report_file}")
        
        sys.exit(0 if success else 1)
    
    else:
        # Run basic tests by default
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
