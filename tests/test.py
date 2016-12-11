import unittest

from stackvm import stack, Parser
from stackvm.tokenizer import Symbol, QTerminal, QNonTerminal, Token, Tokenizer


class StackTest(unittest.TestCase):
    def test_push(self):
        s = stack.Stack()
        s.push(1)
        self.assertEqual(s.top(), 1)

class SymbolTest(unittest.TestCase):
    def test_qterm(self):
        t = QTerminal('test')
        self.assertTrue(t.is_terminal)
    
    def test_qnonterm(self):
        t = QNonTerminal('test')
        self.assertFalse(t.is_terminal)
    
    def test_symbol_name(self):
        s = Symbol('this is a test')
        self.assertEqual(s.name, 'this is a test')

class TokenTest(unittest.TestCase):
    def test_token_creation(self):
        s = QTerminal('test')
        t = Token(s, 'something')
        self.assertEqual(t.lexeme, 'something')
        self.assertEqual(t.value, 'something')
        self.assertEqual(t.symbol.name, 'test')
    
    def test_token_symbolreassignment(self):
        temp = QTerminal('temp')
        final = QTerminal('final')
        token = Token(temp, 'temp')
        token.symbol = final
        self.assertEqual(token.symbol.name, 'final')
    
    def test_token_is_terminal_passthrough(self):
        term = QTerminal('terminal')
        token = Token(term, 'a token')
        self.assertTrue(token.is_terminal)
        token.symbol = QNonTerminal('nonterminal')
        self.assertFalse(token.is_terminal)

def lexemes(tokens):
    """linear string of token lexemes"""
    return '{}'.format(" ".join(t.lexeme for t in tokens))
    
def symbols(tokens):
    """linear string of symbol names"""
    return "{}".format(" ".join(t.symbol.name for t in tokens))

    
class TokenizerTest(unittest.TestCase):
    tokenizer = Tokenizer('')
    tokenizer.add_lexer('\\s+', None)
    tokenizer.add_lexer(r'"([^"]+)"', 'LITERAL')
    tokenizer.add_lexer(r"[a-zA-Z_]+\:", 'LABEL')
    tokenizer.add_lexer(r"[a-zA-Z_]+'", 'STORAGE')
    tokenizer.add_lexer(r'#.+$', 'COMMENT')
    tokenizer.add_lexer(r'[a-zA-Z_]+', 'NAME')
    tokenizer.add_lexer(r'\.', 'STOP')
    tokenizer.add_lexer(r'-?[\d.]+', 'NUMBER')

    tokenizer.add_lexer(r'\S+', 'SYMBOL')
    
    def test_name(self):
        self.assertEqual(lexemes(self.tokenizer('mylife dorun')), 'mylife dorun')
        self.assertEqual(symbols(self.tokenizer('mylife dorun')), 'NAME NAME')
    
    def test_number(self):
        self.assertEqual(lexemes(self.tokenizer('1')), '1')
        self.assertEqual(symbols(self.tokenizer('1')), 'NUMBER')
    
    def test_linebreak(self):
        self.assertEqual(lexemes(self.tokenizer('a b \n c')), 'a b c')
        
    def test_comment(self):
        self.assertEqual(lexemes(self.tokenizer("#comment\nname")), '#comment name')
        self.assertEqual(symbols(self.tokenizer("#comment\nname")), 'COMMENT NAME')
        
    def test_symbol(self):
        self.assertEqual(symbols(self.tokenizer('%')), 'SYMBOL')
        
class ParserTest(unittest.TestCase):
    def setUp(self):
        text = """varlist := VAR { VAR }; VAR := [a-z]+;"""
        self.parser = Parser(text)
        
    def test_the_first(self):
        self.assertEqual(self.parser.start_rule, 'varlist')
    
    def test_the_second(self):
        ebnfnode = self.parser.rules['varlist']
        self.assertFalse(ebnfnode.alternate)
        self.assertFalse(ebnfnode.optional)
        self.assertFalse(ebnfnode.repeating)
        self.assertFalse(ebnfnode.oneormore)
        self.assertFalse(ebnfnode.token.is_terminal)
        self.assertEqual(ebnfnode.token.lexeme, 'varlist')
        self.assertEqual(ebnfnode.token.symbol.name, 'SEQUENCE')
        self.assertEqual(len(ebnfnode.children), 2)
        second = ebnfnode.children[1]
        self.assertEqual(len(second.children), 1)
        self.assertEqual(second.token.symbol.name, 'REPEATING')
        self.assertTrue(second.repeating)
        self.assertFalse(second.alternate)
        self.assertFalse(second.optional)
        bracket = second.children[0]
        
        self.assertEqual(bracket.token.symbol.name, 'TERM')
if __name__ == '__main__':
    unittest.main()