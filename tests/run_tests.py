#!/usr/bin/env python3
"""
Test runner for NeuroBM test suite.

This script runs all tests with proper configuration and reporting.
"""

import unittest
import sys
import os
import time
from pathlib import Path
from io import StringIO
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class ColoredTextTestResult(unittest.TextTestResult):
    """Test result class with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.write("✅ ")
            self.stream.writeln(self.getDescription(test))
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write("❌ ")
            self.stream.writeln(self.getDescription(test))
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write("❌ ")
            self.stream.writeln(self.getDescription(test))
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write("⏭️  ")
            self.stream.writeln(f"{self.getDescription(test)} (skipped: {reason})")


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Test runner with colored output."""
    
    resultclass = ColoredTextTestResult
    
    def run(self, test):
        """Run the test suite."""
        print("🧪 NeuroBM Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        result = super().run(test)
        end_time = time.time()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"✅ Passed: {result.success_count}")
        print(f"❌ Failed: {len(result.failures)}")
        print(f"💥 Errors: {len(result.errors)}")
        print(f"⏭️  Skipped: {len(result.skipped)}")
        print(f"🏃 Total: {result.testsRun}")
        
        if result.failures:
            print("\n💥 Failures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\n❌ Errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        success_rate = (result.success_count / result.testsRun * 100) if result.testsRun > 0 else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if result.wasSuccessful():
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed. Please check the output above.")
        
        return result


def discover_tests(test_dir: Path, pattern: str = "test_*.py") -> unittest.TestSuite:
    """Discover and load tests from directory."""
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern=pattern)
    return suite


def run_specific_test(test_name: str) -> unittest.TestSuite:
    """Run a specific test module or test case."""
    loader = unittest.TestLoader()
    
    try:
        # Try to load as module
        suite = loader.loadTestsFromName(test_name)
    except (ImportError, AttributeError):
        # Try to load as file
        test_file = Path(__file__).parent / f"{test_name}.py"
        if test_file.exists():
            spec = unittest.util.spec_from_file_location(test_name, test_file)
            module = unittest.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            suite = loader.loadTestsFromModule(module)
        else:
            raise ValueError(f"Could not find test: {test_name}")
    
    return suite


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run NeuroBM tests')
    parser.add_argument('--test', type=str, help='Run specific test module')
    parser.add_argument('--pattern', type=str, default='test_*.py', 
                       help='Test file pattern')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2],
                       help='Test verbosity level')
    parser.add_argument('--failfast', action='store_true',
                       help='Stop on first failure')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Set up test environment
    test_dir = Path(__file__).parent
    
    # Discover or load specific tests
    if args.test:
        suite = run_specific_test(args.test)
    else:
        suite = discover_tests(test_dir, args.pattern)
    
    # Configure runner
    verbosity = 0 if args.quiet else args.verbosity
    runner = ColoredTextTestRunner(
        verbosity=verbosity,
        failfast=args.failfast,
        buffer=True
    )
    
    # Run tests
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
