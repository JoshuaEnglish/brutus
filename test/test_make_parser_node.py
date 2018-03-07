import unittest

from brutus.ebnf import make_parser_node, EBNFTokenizer


class StupidErrors(unittest.TestCase):
    def test_mismatched_bracket(self):
        tokens = list(EBNFTokenizer('[A};'))
        self.assertRaises(SyntaxError, make_parser_node, 'test', tokens)

    def test_optional(self):
        tokens = list(EBNFTokenizer('[A] B'))
        node, remaining = make_parser_node('optional', tokens)
        self.assertEqual(node.token.lexeme, 'optional')


if __name__ == '__main__':
    unittest.main()
