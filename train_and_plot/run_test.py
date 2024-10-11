import sys
import importlib

test_module_name = sys.argv[1]
test_module_path = f"tests.{test_module_name}"
test_module = importlib.import_module(test_module_path)
test_args = sys.argv[2:]

test_module.run_test(*test_args)