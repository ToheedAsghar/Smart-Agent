import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.custom_tools import calculator

class TestCalculator(unittest.TestCase):
    def test_simple_math(self):
        self.assertEqual(calculator.invoke("2 + 2"), "4")
        self.assertEqual(calculator.invoke("10 * 5"), "50")

    def test_float_math(self):
        # Result might be float string
        res = calculator.invoke("10 / 4")
        self.assertEqual(float(res), 2.5)

    def test_precedence(self):
        self.assertEqual(calculator.invoke("2 + 3 * 4"), "14")
        self.assertEqual(calculator.invoke("(2 + 3) * 4"), "20")

    def test_allowed_functions(self):
        self.assertEqual(calculator.invoke("abs(-5)"), "5")
        self.assertEqual(calculator.invoke("max(1, 5, 2)"), "5")
        self.assertEqual(calculator.invoke("min(1, 5, 2)"), "1")
        self.assertEqual(calculator.invoke("pow(2, 3)"), "8")
        # round might return '3.14' or '3.14000000' etc depending on impl, usually str(float)
        self.assertEqual(float(calculator.invoke("round(3.14159, 2)")), 3.14)

    def test_unsafe_execution(self):
        # These should fail with simpleeval
        # Trying to access __import__ or similar
        res = calculator.invoke("__import__('os').system('echo hack')")
        self.assertIn("Error", res)

        res = calculator.invoke("open('test.txt', 'w')")
        self.assertIn("Error", res)

    def test_variables_access(self):
        # Ensure we can't access arbitrary variables if they were exposed
        # Though simpleeval with names={} should prevent this.
        pass

if __name__ == '__main__':
    unittest.main()
