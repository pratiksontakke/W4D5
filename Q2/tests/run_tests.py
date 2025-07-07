"""
Test runner script for the Indian Legal Document Search System.
"""

import json
import logging
import sys
import unittest
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_results.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


def run_tests():
    """
    Run all test suites and generate a report.
    """
    # Start timing
    start_time = datetime.now()

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Create test result and runner
    result = unittest.TestResult()
    runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    logging.info("Starting test execution...")
    test_result = runner.run(suite)

    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()

    # Prepare test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "execution_time": execution_time,
        "total_tests": test_result.testsRun,
        "failures": len(test_result.failures),
        "errors": len(test_result.errors),
        "skipped": len(test_result.skipped),
        "success_rate": (
            test_result.testsRun - len(test_result.failures) - len(test_result.errors)
        )
        / test_result.testsRun
        * 100,
    }

    # Log results
    logging.info("Test Execution Summary:")
    logging.info(f"Total Tests: {report['total_tests']}")
    logging.info(f"Failures: {report['failures']}")
    logging.info(f"Errors: {report['errors']}")
    logging.info(f"Skipped: {report['skipped']}")
    logging.info(f"Success Rate: {report['success_rate']:.2f}%")
    logging.info(f"Execution Time: {report['execution_time']:.2f} seconds")

    # Save report to file
    report_file = Path("test_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    return test_result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
