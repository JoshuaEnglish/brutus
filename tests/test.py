import unittest

from stackvm import stack

class StackTest(unittest.TestCase):
    def test_push(self):
        s = stack.Stack()
        s.push(1)
        self.assertEqual(s.top(), 1)

if __name__ == '__main__':
    unittest.main()