import unittest
import logging

from brutus import Parser
from brutus.tokenizer import Symbol, QTerminal, QNonTerminal, Token, Tokenizer


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
    tokenizer.add_lexer(r'[-+*/]', 'BINOP')

    tokenizer.add_lexer(r'\S+', 'SYMBOL')

    def test_name(self):
        self.assertEqual(lexemes(self.tokenizer('mylife dorun')),
                         'mylife dorun')
        self.assertEqual(symbols(self.tokenizer('mylife dorun')),
                         'NAME NAME')

    def test_number(self):
        self.assertEqual(lexemes(self.tokenizer('1')), '1')
        self.assertEqual(symbols(self.tokenizer('1')), 'NUMBER')
        self.assertEqual(symbols(self.tokenizer('-1')), 'NUMBER')
        self.assertEqual(lexemes(self.tokenizer('-1')), '-1')

    def test_linebreak(self):
        self.assertEqual(lexemes(self.tokenizer('a b \n c')), 'a b c')

    def test_comment(self):
        self.assertEqual(lexemes(self.tokenizer("#comment\nname")),
                         '#comment name')
        self.assertEqual(symbols(self.tokenizer("#comment\nname")),
                         'COMMENT NAME')

    def test_symbol(self):
        self.assertEqual(symbols(self.tokenizer('%')), 'SYMBOL')

    def test_binop(self):
        self.assertEqual(symbols(self.tokenizer('-')), 'BINOP')
        self.assertEqual(symbols(self.tokenizer('+')), 'BINOP')
        self.assertEqual(symbols(self.tokenizer('/')), 'BINOP')
        self.assertEqual(symbols(self.tokenizer('*')), 'BINOP')


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


# NEED TO TEST FOR ALTERTANE SEQUENCES AND RULES, NOT JUST ALTERNATE TOKENS
class OrSequenceTest(unittest.TestCase):
    text = """thing := A B | C D;
        A := [a];
        B := [b];
        C := [c];
        D := [d];
        """

    parser = Parser(text)

    def test_ab(self):
        self.parser.parse_text('ab')

    def test_cd(self):
        self.parser.parse_text('cd')

    def test_ac(self):
        self.assertRaises(ValueError, self.parser.parse_text, 'ac')

    def test_bd(self):
        self.assertRaises(ValueError, self.parser.parse_text, 'bd')


class OrSequenceTestWithCommonStart(unittest.TestCase):
    text = """thing := A B | A D;
        A := [a];
        B := [b];
        C := [c];
        D := [d];
        """

    parser = Parser(text)

    def test_ab(self):
        self.parser.parse_text('ab')

    def test_ad(self):
        self.parser.parse_text('ad')


class RuleMustGetEverything(unittest.TestCase):
    text = """this := A B; A := [a]; B := [b];"""
    parser = Parser(text)

    def test_okily_dokily(self):
        ok, node, detrituse = self.parser.parse_text('ab')

    def test_nope(self):
        self.assertRaises(ValueError, self.parser.parse_text, 'a')


class OrSequenceTestWithRules(unittest.TestCase):
    text = """thing := this | that;
        this := A B;
        that := A D;
        A := [a];
        B := [b];
        C := [c];
        D := [d];
        """

    parser = Parser(text)

    def tearDown(self):
        logging.root.setLevel(logging.INFO)
        self.parser.report()

    def test_ab(self):
        self.parser.parse_text('ab')

    def test_ad(self):
        logging.root.setLevel(logging.DEBUG)
        self.parser.parse_text('ad')


class OrTest(unittest.TestCase):
    text = """thing := NUMBER | LETTER;
        LETTER := [a-z];
        NUMBER := [0-9];
    """

    parser = Parser(text)

    def test_parser(self):
        self.assertEqual(self.parser.start_rule, 'thing')

        self.assertTrue(self.parser.rules['thing'].alternate)
        self.assertEqual(len(self.parser.rules['thing'].children), 3)
        # The OR token is not added to the node.children list

    def test_number(self):
        ok, node, detritus = self.parser.parse_text('1')
        self.assertTrue(ok)
        self.assertListEqual(detritus, [])
        self.assertEqual(node.token.lexeme, 'thing')

    def test_letter(self):
        ok, node, detritus = self.parser.parse_text('j')
        self.assertTrue(ok)
        self.assertListEqual(detritus, [])
        self.assertEqual(node.token.lexeme, 'thing')


class IffyTest(unittest.TestCase):

    text = """ifstmt := "if" "(" NAME ")" "(" assignment ")" ["else" "(" assignment ")"] ;
            assignment := NAME STORE expr;
            expr := term {("+" | "-") term};
            term := factor {("*" | "/") factor};
            factor := INTEGER | NAME | LITERAL | "(" expr ")";
            NAME := [a-z]+;
            INTEGER := -?[0-9]+;
            STORE := <-;
            OP := [+\-*/];
            PARENS := [()];
            """

    parser = Parser(text)

    def test_ifonly(self):
        ok, node, detritus = self.parser.parse_text('if (a) (res <- 1)')
        self.assertTrue(ok)
        self.assertListEqual(detritus, [])
        self.assertEqual(node.token.lexeme, 'ifstmt')


class CalcTest(unittest.TestCase):
    text = """statements := assignment { assignment } ;
        assignment := VAR STORE expr STOP;
        expr := term { EXPROP term};
        term := factor { TERMOP factor};
        factor := INTEGER | VAR | "(" expr ")";
        VAR := [a-z]+;
        INTEGER := -?[0-9]+;
        STORE := <-;
        EXPROP :=[-+];
        TERMOP :=[*/];
        STOP := [\.];
        PARENS := [()];
        """
    parser = Parser(text)

    def setUp(self):
        logging.root.setLevel(logging.INFO)

    def tearDown(self):
        logging.root.setLevel(logging.INFO)

    def test_simple(self):
        ok, node, detritus = self.parser.parse_text("x <- 2 - 1.")
        self.assertTrue(ok)
        self.assertListEqual(detritus, [])
        self.assertEqual(node.token.lexeme, 'statements')
        self.assertEqual(len(node.children), 1)
        child = node.children[0]
        self.assertEqual(child.token.lexeme, 'assignment')
        self.assertEqual(len(child.children), 4)
        self.assertEqual(child.children[0].token.lexeme, 'x')
        self.assertEqual(child.children[1].token.lexeme, '<-')
        self.assertEqual(child.children[2].token.lexeme, 'expr')
        self.assertEqual(child.children[3].token.lexeme, '.')


class OptionalTest(unittest.TestCase):
    text = """list := A[B]; A := [a-z]+; B := [0-9];"""
    parser = Parser(text)

    def test_creation(self):
        self.assertEqual(len(self.parser.rules), 1)
        start = self.parser.start_rule
        self.assertEqual(len(self.parser.rules[start].children), 2)
        first, second = self.parser.rules[start].children
        self.assertTrue(second.optional)
        self.assertFalse(second.alternate)
        self.assertFalse(second.repeating)
        self.assertFalse(second.oneormore)

    def test_simple(self):
        ok, node, detritus = self.parser.parse_text('a1')
        self.assertTrue(ok)

    def test_ignored(self):
        ok, node, detritus = self.parser.parse_text('a')
        self.assertTrue(ok)

    def test_optional_more_letters(self):
        ok, node, detritus = self.parser.parse_text('az3')
        self.assertTrue(ok)

    def test_optional(self):
        ebnfnode = self.parser.rules['list']
        self.assertEqual(len(ebnfnode.children), 2)

        second = ebnfnode.children[1]
        self.assertEqual(len(second.children), 1)
        self.assertEqual(second.token.symbol.name, 'OPTIONAL')
        self.assertFalse(second.repeating)
        self.assertFalse(second.alternate)
        self.assertTrue(second.optional)


class AtLeastOnce(unittest.TestCase):
    text = ''''statement := A <B>; A := [a]; B := [b];'''
    parser = Parser(text)

    def test_ontology(self):
        start = self.parser.start_rule
        self.assertEqual(len(self.parser.rules), 1)
        self.assertEqual(len(self.parser.rules[start].children), 2)
        first, second = self.parser.rules[start].children
        self.assertTrue(second.oneormore)
        self.assertFalse(second.alternate)
        self.assertFalse(second.optional)
        self.assertFalse(second.repeating)

    def test_atleastonce(self):
        for text in ["ab", "abb", "abbb", "abbb"]:
            ok, node, detritus = self.parser.parse_text(text)
            self.assertTrue(ok)
            self.assertListEqual(detritus, [])

    def test_failure(self):
        self.assertRaises(SyntaxError, self.parser.parse_text, "a")


class ZeroOrMore(unittest.TestCase):
    text = '''statement := A {B}; A:= [a]; B:= [b];'''
    parser = Parser(text)

    def test_zeroormore(self):
        for text in ["a", "ab", "abb", "abbb"]:
            ok, node, detritus = self.parser.parse_text(text)
            self.assertTrue(ok)
            self.assertListEqual(detritus, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
