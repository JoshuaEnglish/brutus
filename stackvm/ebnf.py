"""EBNF Parser
Takes an EBNF definition and creates an Abstract Tree Node
"""

from __future__ import (absolute_import, print_function)

from operator import attrgetter
import itertools

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
STARTREPEAT = TerminalSymbol('STARTREPEAT') # {
ENDREPEAT = TerminalSymbol('ENDREPEAT') # }
STARTGROUP = TerminalSymbol('STARTGROUP') # (
ENDGROUP = TerminalSymbol('ENDGROUP')
STARTOPTION = TerminalSymbol('STARTOPTION')
ENDOPTION = TerminalSymbol('ENDOPTION')
OR = TerminalSymbol('OR')
SYMBOL = TerminalSymbol('SYMBOL')
LITERAL = TerminalSymbol('LITERAL')
RULE = NonTerminalSymbol('RULE')

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


def match_terminal(rule, tokens):
    if rule.symbol == LITERAL:
        print("Matching LITERAL", rule.lexeme, tokens[0].lexeme, rule.lexeme==tokens[0].lexeme)
        if rule.lexeme == tokens[0].lexeme:
            return Node(tokens[0].symbol.name, tokens[0]), tokens[1:]
        else:
            return None, tokens
    elif rule.lexeme == tokens[0].symbol.name:
        return Node(tokens[0].symbol.name, tokens[0]), tokens[1:]
    else:
        return None, tokens

# if OR in [t.symbol for t in expected[0]]:
def match_sequence(rulename, rule, tokens):
    expected = collapse_groups(rule)
    is_or_group =any(t.symbol == OR for t in expected if not isinstance(t, list))
    print("match sequence:", rulename, expected, is_or_group )
    node = Node(rulename)

    if is_or_group:
        junk, tokens = match_alternate('junk', split_by_or(rule), tokens)
        node.children.extend(junk.children)
        return node, tokens

    while expected:

        if isinstance(expected[0], list):
            child, tokens = match_sequence('junk', expected[0], tokens)
            node.children.extend(child.children)

        elif expected[0].symbol != LITERAL and expected[0].lexeme != tokens[0].symbol.name:
            return None, tokens

        elif expected[0].symbol.is_terminal:
            #print("matching terminal symbol?")
            child, tokens = match_terminal(expected[0], tokens)
            if child is not None:
                node.children.append(child)

        else:
            pass # deal with non-terminals
        expected.pop(0)
    return node, tokens

def match_rule(rulename, parser, tokens):
    """entry point.
    """
    rule = parser.rules.get(rulename)
    if rule is None:
        raise SyntaxError("No rule for {}".format(rulename))
    return match_sequence(rulename, rule, tokens)


def match_alternate(rulename, alternates, tokens):
    """rulename is the name. Alternates is a list of lists. Tokens a list of tokens"""
    print(rulename, alternates, tokens)
    for alternate in alternates:
        node, tokens = match_sequence(rulename, alternate, tokens)
        if node is not None:
            return node, tokens
    raise SyntaxError

def split_by_or(iterable):
    return [list(g) for k,g in itertools.groupby(iterable, lambda x:x.symbol == OR) if not k]

"""break groups into sublists
NUMBER STARTGROUP PLUS OR MINUS ENDGROUP NUMBER -> [NUMBER, [PLUS, OR, MINUS], NUMBER]
"""
def quick_tree(iterable, res=None):
    res = [] if res is None else res
    stuff = list(iterable)
    while stuff:

        thing = stuff.pop(0)

        if thing.symbol == STARTGROUP:

            stuff, thing = quick_tree(stuff)
            thing = thing

        elif thing.symbol == ENDGROUP:

            return stuff, res
        res.append(thing)
    return stuff, res

def collapse_groups(iterable):
    return quick_tree(iterable)[1]

class EParser(object):
    def __init__(self, expectation):
        self.expected = expectation

class TerminalParser(EParser):
    def __call__(self, tokens):
        sym = self.expected.symbol
        lex = self.expected.lexeme
        tlex = tokens[0].lexeme
        tsym = tokens[0].symbol.name

        if (sym == LITERAL and lex == tlex) or lex == tsym:
            return Node(tsym, tokens[0]), tokens[1:]
        else:
            return None, tokens


class OrParser(EParser):
    # expected is a list of lists of parsers
    def __call__(self, tokens):
        pass

def build_or_parser(ebnftokens):
    return OrParser(split_by_or(ebnftokens))

def RuleParser(EParser):
    """this is the non-terminal version"""
    def __call__(self, tokens):
        pass

def build_parser(ebnftoken):
    if ebnftoken.is_terminal:
        return TerminalParser(ebnftoken)
    else:
        return RuleParser(ebnftoken)
if __name__ == '__main__':
    text = """expr := NUMBER {(PLUS | MINUS) NUMBER};
            sum := NUMBER (PLUS | MINUS) NUMBER;
            term := factor {(MUL | DIV) factor)};
            factor := NUMBER | OPARENS NUMBER CPARENS;
            lfactor := "(" NUMBER ")";
            """
    p = EBNFParser(text)
    #t = EBNFTokenizer('RECALL | NUMBER')
    #print(list(t))

    number = [tokenizer.Token(NUMBER, 2)]
    parens = [tokenizer.Token(OPARENS, '('), tokenizer.Token(NUMBER,3), tokenizer.Token(CPARENS, ')')]
    addends = [tokenizer.Token(NUMBER, 1), tokenizer.Token(PLUS, '+'), tokenizer.Token(NUMBER, 2)]
    subtract = [tokenizer.Token(NUMBER, 3), tokenizer.Token(MINUS, '-'), tokenizer.Token(NUMBER, 2)]
##    node, remaining = match_sequence('factor', p.rules['factor'][2:], parens)
##
##    print_node(node)
##    assert remaining == []
##
##    node, remaining = match_sequence('factor', p.rules['factor'][:1], number )
##    print_node(node)
##    assert remaining == []
##
##    node, remaining = match_alternate('factor', split_by_or(p.rules['factor']), number )
##    print_node(node)
##    assert remaining == []
##
##    node, remaining = match_alternate('factor', split_by_or(p.rules['factor']), parens )
##    print_node(node)
##    assert remaining == []
##
##    try:
##        node, remaining = match_alternate('factor', split_by_or(p.rules['factor']), [tokenizer.Token(LITERAL, 'if')])
##    except SyntaxError:
##        print("Syntax error as expected")

    ### match_rule tests
    node, remaining = match_rule('sum', p, addends )
    print_node(node)
    assert remaining == []

    node, remaining = match_rule('sum', p, subtract )
    print_node(node)
    assert remaining == []

    print('testing factor')
    node, remaining = match_rule('factor', p, parens)
    print_node(node)
    assert remaining==[]

##
##    number_parser = TerminalParser(tokenizer.Token(SYMBOL, "NUMBER"))
##    node, remaining = number_parser(number)
##    print_node(node)
##    assert remaining==[]
##
##    node, remaining = number_parser(addends)
##    print_node(node)
##    print(remaining)

    print('test lfactor')
    node, remaining = match_rule('lfactor', p, parens)
    print_node(node)
    assert remaining == []

    print('test parensparser')
    parens_parser = TerminalParser(tokenizer.Token(LITERAL, "("))
    node, remaining = parens_parser(parens)
    print_node(node)