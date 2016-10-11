"""EBNF Parser
Takes an EBNF definition and creates an Abstract Tree Node
"""

from __future__ import (absolute_import, print_function)

from operator import attrgetter

import tokenizer

getsym = attrgetter('symbol')

class TerminalSymbol(tokenizer.Symbol):
    _next_id = 1
    is_terminal = True


class NonTerminalSymbol(tokenizer.Symbol):
    _next_id = 1000
    is_terminal = False

class Node(object):
    def __init__(self, name, token=None, children=None):
        self.name = name
        self.token = token
        self.children = children or []


class EBNFParser(object):
    """
    Class to read EBNF text and return a dictionary of rules
    plus a dictionary of (name, Symbol) pairs
    """

    def __init__(self, text):
        self.symbol_table = {}

        lines = [line for line in text.split(';') if line.strip()]
        data = [line.partition(":=") for line in lines]
        self.rules = dict((key.strip(), list(EBNFTokenizer(val))) for key, junk, val in data)

        for nonterminal in self.rules:
            rule = self.rules[nonterminal]

    def IS(self, token, symbol):
        if not symbol.is_terminal:
            raise ValueError("IS requires terminal symbol")
        return token if token.symbol == symbol else None

    def OR(self, token, symbols):
        for symbol in symbols:
            if symbol.is_terminal and token.symbol == symbol:
                return token
        return None

    def match_terminal(self, rule, source):
        """

        :param rule: the rule from the ebnf dictionary
        :param source: list of tokens
        :return: ASTNode or None
        """
        # a terminal rule is only going to match a single token

        return None

    def match_rule(self, rule, tokens):
        """
        matches a list of tokens to a rule

        :param rule: name of the rule to match
        :param tokens: list of input tokens
        :return: Node or None
        """

        expected_tokens = list(self.rules[rule]) # want a disposable copy
        idx = 0

        maybe_node = Node(rule)

        while expected_tokens:
            current_token = tokens[idx]
            what_i_want = expected_tokens.pop(0)
            print("What I want:", what_i_want)
            print("Current Token:", current_token)
            if what_i_want.symbol.is_terminal:
                if current_token.symbol.name == what_i_want.value:
                    maybe_node.children.append(Node(what_i_want.value, current_token))
                    print("..Match, moving on")
                    idx += 1
                    if expected_tokens[0].symbol == OR:
                        return maybe_node
                elif expected_tokens[0].symbol == OR:
                    expected_tokens.pop(0)
                    continue
                else:
                    return None

            else:
                maybe_node.children.append( self.match_rule(what_i_want.value, tokens[idx:]))
        return maybe_node





from string import ascii_lowercase, ascii_uppercase, whitespace
# Tokens for the EBNF Tokenizer and Parser
STARTREPEAT = TerminalSymbol('STARTREPEAT')
ENDREPEAT = TerminalSymbol('ENDREPEAT')
STARTGROUP = TerminalSymbol('STARTGROUP')
ENDGROUP = TerminalSymbol('ENDGROUP')
STARTOPTION = TerminalSymbol('STARTOPTION')
ENDOPTION = TerminalSymbol('ENDOPTION')
OR = TerminalSymbol('OR')
SYMBOL = TerminalSymbol('SYMBOL')
LITERAL = TerminalSymbol('LITERAL')
RULE = NonTerminalSymbol('rule')

# Tokens for the actual language we're parsing
PLUS = TerminalSymbol('PLUS')
MINUS = TerminalSymbol('MINUS')
MUL = TerminalSymbol('MUL')
DIV = TerminalSymbol('DIV')

OPARENS = TerminalSymbol('OPARENS')
CPARENS = TerminalSymbol('CPARENS')

NUMBER = TerminalSymbol('NUMBER')

class EBNFTokenizer(tokenizer.BaseLexer):
    def _emit(self, symbol,):
        # print('emit', token)
        tokenizer.BaseLexer._emit(self, symbol)

    def _lex_start(self):
        assert self._start == self._pos

        peek = self._peek

        if peek is None:
            return self._lex_eof

        elif peek in ' \t\n\r':
            return self.skip_whitespace()

        elif peek == '"':
            return self.get_quoted_string()

        elif peek in '{}':
            self._skip()
            self._emit(STARTREPEAT if peek=='{' else ENDREPEAT)
            return self._lex_start

        elif peek in '()':
            self._skip()
            self._emit(STARTGROUP if peek=='(' else ENDGROUP)
            return self._lex_start

        elif peek in '[]':
            self._skip()
            self._emit(STARTOPTION if peek=='[' else ENDOPTION)
            return self._lex_start

        elif peek == '|':
            self._skip()
            self._emit(OR)
            return self._lex_start

        elif peek in ascii_lowercase:
            return self.get_rule()

        elif peek in ascii_uppercase:
            return self.get_symbol()

    def get_symbol(self):
        self._accept_run(ascii_uppercase)
        self._emit(SYMBOL)
        return self._lex_start

    def get_rule(self):
        self._accept_run(ascii_lowercase)
        self._emit(RULE)
        return self._lex_start


    def skip_whitespace(self):
        self._accept_run(' \r\n\t')
        self._ignore()
        return self._lex_start

    def get_quoted_string(self):
        self._skip() # over opening quote
        self._accept_until('"')
        self._emit(LITERAL)

        # raise unterminated if next character not closing quote
        if self._peek != '"':
            raise SyntaxError("unterminated quote")
        self._skip()

        return self._lex_start

    def _lex_eof(self):
        assert self._start == self._pos == self._len
        return None

def print_node(node, indent=0):
    print(" "*indent, node.name, node.token)
    for child in node.children:
        print_node(child, indent+2)

if __name__ == '__main__':
    text = """expr := term {(PLUS | MINUS) term};
            term := factor {(MUL | DIV) factor)};
            factor := NUMBER | OPARENS expr CPARENS;
            """
    p = EBNFParser(text)
    #t = EBNFTokenizer('RECALL | NUMBER')
    #print(list(t))

    #print_node(p.match_rule('factor', [tokenizer.Token(NUMBER, 1)]))
    print_node(p.match_rule('expr', [tokenizer.Token(NUMBER, 2)]))
    #print_node(p.match_rule('factor', [tokenizer.Token(OPARENS, '('),
    #                                   tokenizer.Token(NUMBER,3),
    #                                   tokenizer.Token(CPARENS, ')')]))





