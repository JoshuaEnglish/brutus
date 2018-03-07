
import unittest

from brutus import stack


class StackTest(unittest.TestCase):
    def test_push(self):
        s = stack.Stack()
        s.push(1)
        self.assertEqual(s.top(), 1)
        s.push('bogart')
        self.assertEqual(s.top(), 'bogart')
        self.assertListEqual(s.items, [1, 'bogart'])

    def test_push_items(self):
        s = stack.Stack()
        s.push(('here', 'there', 'everywhere'))
        self.assertEqual(s.top(), 'everywhere')

    def test_pop(self):
        s = stack.Stack()
        s.push(1)
        s.push('bogart')
        s.pop()
        self.assertEqual(s.top(), 1)

    def test_creation(self):
        s = stack.Stack()
        self.assertTrue(s.is_empty)
        self.assertListEqual(s.items, [])
        self.assertRaises(IndexError, s.top)
        self.assertRaises(IndexError, s.pop)

    def test_clear(self):
        s = stack.Stack()
        s.push(1)
        s.push('marmalade')
        s.clear()
        self.assertTrue(s.is_empty)


if __name__ == '__main__':
    unittest.main()
