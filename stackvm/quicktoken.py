# -*- coding: utf-8 -*-
"""
quicktokenizer

Idea to build a tokenizer based on generated rules
Created on Sat Nov  5 18:08:31 2016

@author: Josh
"""
import re
import logging
from collections import namedtuple, OrderedDict, Counter
from itertools import groupby, islice, cycle

DOTS = " :  "
def indent(x):
    """indent(x)
    Return a string of length x with dot characters spaced into columns.
    This is used in printing tree structures
    """
    return ''.join(islice(cycle(DOTS), x))


class Symbol(object):
    """Symbol(name)
    Base class for Terminal and NonTerminal Symbols
    """
    is_terminal = None
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __str__(self):
        return "[%s %s]" % (self.__class__.__name__, self.name)

    __repr__ = __str__

class QTerminal(Symbol):
    is_terminal = True

class QNonTerminal(Symbol):
    is_terminal = False

class EBNFTerminalSymbol(Symbol):
    is_terminal = True

class EBNFNonTerminalSymbol(Symbol):
    is_terminal = False

class Token(object):
    __slots__ = ('_symbol', '_lexeme')

    def __init__(self, symbol, lexeme):
        self._symbol = symbol  # always a terminal symbol
        self._lexeme = lexeme

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, newsymbol):
        if not isinstance(newsymbol, Symbol):
            raise ValueError("cannot update EBNFToken symbol to non Symbol")
        self._symbol = newsymbol

    @property
    def lexeme(self):
        return self._lexeme

    value = lexeme

    @property
    def is_terminal(self):
        return self.symbol.is_terminal


    def __str__(self):
        return "<%s %s: %s >" % (self.symbol.name, self.__class__.__name__, self.lexeme)

    __repr__ = __str__

Lexer = namedtuple('Lexer', 'pattern symbol')

class Tokenizer(object):
    def __init__(self, text, token_class=None):
        self.text = text
        self._t_class = token_class or Token
        self.lexers = []
        self._compiled_lexers = {}
        self.symbols = {}
        self._token_count = 0

    def __call__(self, text):
        self.text = text
        return self

    def __iter__(self):

        return iter(self.get_next_token, None)

    def add_lexer(self, pattern, symbol_name):
        if pattern is self._compiled_lexers:
            raise ValueError("lexer for %s already exists" % pattern)

        if symbol_name is None:
            self.lexers.append(Lexer(pattern, None))
        else:
            self.lexers.append(Lexer(pattern,
                                     self.get_symbol(symbol_name)))


    def get_next_token(self):
        if not self.text:
            return None
        if self._token_count == 10:
            pass
        self._token_count += 1
        # print("Text:", self.text)
        for lexer in self.lexers:

            matcher = self.get_matcher(lexer.pattern)
            match = matcher.match(self.text)
            # print("trying %s, got %s" % (lexer.pattern, match))

            if match:
                lexeme = match.group()
                # print("found %s, emiting %s" % (lexeme, lexer.symbol))
                self.text = self.text[match.end():]
                if lexer.symbol is not None:
                    return self._t_class(lexer.symbol, lexeme)

    def get_symbol(self, name):
        return self.symbols.setdefault(name, QTerminal(name))

    def get_matcher(self, pattern):
        return self._compiled_lexers.setdefault(pattern, re.compile(pattern))

EBNFTokenizer = Tokenizer('')
EBNFTokenizer.add_lexer('\s+', None)
EBNFTokenizer.add_lexer('[a-z]+', 'RULE')
EBNFTokenizer.add_lexer('[A-Z]+', 'TERM')

EBNFTokenizer.add_lexer(r'"[^"]+"', 'LITERAL')

EBNFTokenizer.add_lexer("[{]", 'STARTREPEAT')
EBNFTokenizer.add_lexer("[}]", 'ENDREPEAT')
EBNFTokenizer.add_lexer("[(]", 'STARTGROUP')
EBNFTokenizer.add_lexer("[)]", 'ENDGROUP')
EBNFTokenizer.add_lexer("[[]", 'STARTOPTIONAL')
EBNFTokenizer.add_lexer("[]]", 'ENDOPTIONAL')
EBNFTokenizer.add_lexer("[|]", 'OR')
EBNFTokenizer.add_lexer(":=", 'DEFINE')
EBNFTokenizer.add_lexer(";", 'ENDDEFINE')

STARTREPEAT = EBNFTokenizer.symbols['STARTREPEAT']
ENDREPEAT = EBNFTokenizer.symbols['ENDREPEAT']
STARTGROUP = EBNFTokenizer.symbols['STARTGROUP']
ENDGROUP = EBNFTokenizer.symbols['ENDGROUP']
STARTOPTION = EBNFTokenizer.symbols['STARTOPTIONAL']
ENDOPTION = EBNFTokenizer.symbols['ENDOPTIONAL']
OR = EBNFTokenizer.symbols['OR']
LITERAL = EBNFTokenizer.symbols['LITERAL']
RULE = EBNFTokenizer.symbols['RULE']
TERM = EBNFTokenizer.symbols['TERM']


REPEATING = EBNFNonTerminalSymbol('REPEATING')
SEQUENCE = EBNFNonTerminalSymbol('SEQUENCE')
OPTIONAL = EBNFNonTerminalSymbol('OPTIONAL')
OPT = EBNFTerminalSymbol('OPT')
REP = EBNFTerminalSymbol('REP')
ATL = EBNFTerminalSymbol('ATL')
ATLEASTONCE = EBNFNonTerminalSymbol('ATLEASTONCE')
ALTERNATING = EBNFNonTerminalSymbol('ALTERNATING')


class ParserNode(object):
    """ParserNode(token)
    ParserNode stores a token and optional children of other ParserNode
    objects to represent a single EBNF rule in a tree.
    """
    def __init__(self, token):
        self.token = token
        self.children = []
        self.alternate = False
        self.optional = False
        self.repeating = False
        self.oneormore = False

    def add(self, thing):
        assert isinstance(thing, ParserNode)
        self.children.append(thing)

    def guess_symbol_class(self):
        """returns NonTerminalSymbol or EBNFNonTerminalSymbol"""
        is_term = all(kid.token.symbol in [OR, LITERAL] for kid in self.children)
        return EBNFTerminalSymbol if is_term else EBNFNonTerminalSymbol

    def __str__(self):
        if len(self.children) > 1:
            flags = ["alternate" if self.alternate else "",
                     "optional" if self.optional else "",
                     "repeating" if self.repeating else ""]
            if any(flags):
                flagstr = ", ".join(f for f in flags if f)
            else:
                flagstr = "sequence"
        else:
            flagstr = ""
        return "<ParserNode {} {}>".format(self.token, flagstr)

    __repr__ = __str__

groupcount = Counter()
SEQ_MAP = {STARTGROUP: SEQUENCE, STARTREPEAT: REPEATING, STARTOPTION: OPTIONAL}
SUFFIX_MAP = {REP: REPEATING, OPT: OPTIONAL, ATL: ATLEASTONCE}

groupings = {STARTGROUP: ENDGROUP, STARTREPEAT: ENDREPEAT, STARTOPTION: ENDOPTION}


def make_parser_node(name, tokens, endtoken=None):
    """make_parser_node(name, tokens [,endtoken])
    Recursive method for taking a series of tokens representing one EBNF
    rule and returning a ParserNode tree.
    """
    if not tokens:
        return None, []
    logging.debug("make_parser_node for '%s' with %d tokens", name, len(tokens))
    this = ParserNode(EBNFToken(SEQUENCE, name))
    logging.debug("this is %d", id(this))

    while tokens:
        first = tokens[0]

        if first.symbol in [STARTGROUP, STARTREPEAT, STARTOPTION]:
            logging.debug("found %s in %d", first.symbol.name, id(this))
            key = name.split('-')[0]
            groupcount[key] += 1
            child, tokens = make_parser_node("%s-%d" % (key, groupcount[key]),
                                             tokens[1:],
                                             groupings[first.symbol])
            if child:
                child.token.symbol = SEQ_MAP[first.symbol]
                this.add(child)

        elif first.symbol in [ENDGROUP, ENDREPEAT, ENDOPTION]:
            logging.debug("Found %s with %d tokens left in %d",
                          first.symbol.name, len(tokens), id(this))
            if endtoken != first.symbol:
                raise SyntaxError("Expected %s to close group, got %s" %
                                  (endtoken.name, first.symbol.name))

            used_brackets = first.symbol in [ENDREPEAT, ENDOPTION]

            if len(tokens) > 1:
                if used_brackets and tokens[1].symbol in [REP, OPT, ATL]:
                    msg = "Illegal mix of brackets and suffixes in %s" % name
                    logging.error(msg)
                    raise SyntaxError(msg)

                if tokens[1].symbol in [REP, OPT, ATL]:
                    this.token.symbol = SUFFIX_MAP[tokens[1].symbol]
                    logging.debug("Changed %d to symbol %s",
                                  id(this), this.token.symbol)
                    tokens.pop(0)

            return this, tokens[1:]

        elif first.symbol == OR:
            logging.debug("Found or in %d with symbol %s",
                          id(this), this.token.symbol)
            this.alternate = True
            this.token.symbol = ALTERNATING
            logging.debug("Changed %d to symbol %s",
                          id(this), this.token.symbol)
            this.add(ParserNode(first))
            tokens.pop(0)

        else:
            this.add(ParserNode(first))
            tokens.pop(0)

    logging.debug("Returning %d with %d tokens", id(this), len(tokens))
    return this, tokens

class ParseTreeNode(object):
    """This node represents the parsed code
    """
    def __init__(self, token):
        self.token = token
        self.children = []

    def __str__(self):
        return "<ParseTreeNode:{} {} >".format(self.token,
                                               self.token)

def print_parsetree(treenode, ind=0):
    print("{0}< {1} >".format(indent(ind), treenode.token.lexeme))
    for child in treenode.children:
        print_parsetree(child, ind+2)

def lexemes(tokens):
    return '"{}"'.format(" ".join(t.lexeme for t in tokens))

def token_lexemes(tokens):
    return '"{}"'.format(" ".join(t.token.lexeme for t in tokens))





class EBNFToken(Token):
    pass

def split_by_or(iterable):
    """splits a list of EBNFTokens into separate lists defined by OR symbols"""
    return [list(g) for k, g in groupby(iterable, lambda x: x.token.symbol == OR)
            if not k]

class EBNFParser(object):
    """
    Class to read EBNF text and return a dictionary of rules
    plus a dictionary of (name, Symbol) pairs
    """

    def __init__(self, text):
        self.symbol_table = {}
        self.start_rule = None
        self.collapse_tree = True
        self.token_class = Token
        logging.debug("EBNFParser.__init__() start")
        lines = [line for line in text.split(';') if line.strip()]
        data = [line.split(":=") for line in lines]
        self.rules = OrderedDict()
        self.tokenizer = Tokenizer('', self.token_class)
        self.tokenizer.add_lexer('\s+', None)
        self.tokenizer.add_lexer(r'"[^"]+"', 'LITERAL')
        for key, val in data:
            key = key.strip()
            if self.start_rule is None:
                if key.islower():
                    self.start_rule = key
            if key in self.symbol_table:
                raise SyntaxError('rule for %s already exists' % key)
            ebnf_tokens = list(EBNFTokenizer(val))

            for token in ebnf_tokens:
                if token.symbol == EBNFTokenizer.symbols['TERM']:
                    if token.lexeme not in self.tokenizer.symbols:
                        self.tokenizer.symbols[token.lexeme] = None
            if key.islower():
                parser_node, remaining = make_parser_node(key, ebnf_tokens)
                if remaining:
                    logging.exception("rule %s did not process correctly", key)
                    raise SyntaxError("rule %s did not process correctly" % key)
                self.rules[key] = parser_node
                self.symbol_table[key] = EBNFNonTerminalSymbol(key)
            elif key.isupper():
                if self.tokenizer.symbols.get(key) is not None:
                    raise ValueError("Redefining terminal symbol %s in EBNF" % key)
                self.tokenizer.symbols[key] = QTerminal(key)
                self.tokenizer.add_lexer(val.strip(), key)
                self.symbol_table[key] = EBNFTerminalSymbol(key)
        #self.make_symbol_table()
        logging.debug("EBNFParser.__init__() end")

        self._calls = Counter()


    def parse_text(self, text):
        self.tokenizer.text = text
        tokens = list(self.tokenizer)
        # print("Parsing tokens:", tokens)
        return self.parse(tokens)

    def parse(self, tokens):
        """Parse tokens into an ParseTreeNode tree"""
        if not self.start_rule:
            raise ValueError("no start rule established")
        return self.match_rule(self.start_rule, tokens)

    def match_rule(self, rule, tokens, i=0):
        """Given a rule name and a list of tokens, return an
        ParseTreeNode and remaining tokens
        """
        parser_node = self.rules.get(rule)
        # print(indent(i), "trying rule", parser_node.token.lexeme, token_lexemes(parser_node.children), lexemes(tokens))

        if parser_node is None:
            logging.exception("No rule for %s", rule)
            raise SyntaxError("No rule for {}".format(rule))

        logging.debug("matching rule %s for %d tokens with %s",
                      rule, len(tokens), parser_node)

        if not isinstance(parser_node.token, EBNFToken):
            raise ValueError("parser node missing required EBNFToken")


        node, tokens = self.match(parser_node, tokens, i)
        if i == 0 and tokens:
            logging.error("Not all input consumed")
            raise ValueError("not all input consumed %s" % tokens)
        return node, tokens

    def match(self, parser_node, tokens, i):
        # print(indent(i),"match '%s' against '%s' with %d remaining" % (parser_node.token.lexeme, tokens[0].lexeme, len(tokens)))
        if parser_node.token.is_terminal:
            res, tokens = self.match_terminal(parser_node, tokens, i)
        else:
            res, tokens = self.match_nonterminal(parser_node, tokens, i)
        # print(indent(i), "match returning (%s, %s)" % (res, lexemes(tokens)))
        return res, tokens

    def match_terminal(self, parser_node, tokens, i):
        #self._calls['match_terminal'] += 1
        #self._calls[parser_node.token.symbol.name] += 1

        logging.debug("matching terminal with %s for %d tokens",
                      parser_node, len(tokens))
        if not tokens:
            return None, tokens

        node = ParseTreeNode(parser_node.token.lexeme)

        if parser_node.token.symbol == RULE:
            child, tokens = self.match_rule(parser_node.token.lexeme, tokens, i+1)
            if child is not None:
                # collapse_tree shortens long descendencies with only one child at the end
                if self.collapse_tree and len(child.children) == 1:
                    node.children.append(child.children[0])
                else:
                    node.children.append(child)
            #else:
                #return None, tokens
                #raise SyntaxError("did not match rule %s" % parser_node.token.lexeme)
        elif parser_node.token.symbol == LITERAL:

            if parser_node.token.lexeme == tokens[0].lexeme:
                logging.debug("matching literal .. matched")
                # print(indent(i), "ate literal", tokens[0].lexeme)
                node.children.append(ParseTreeNode(tokens[0]))
                return node, tokens[1:]

            else:
                logging.debug("matching literal .. nope")
                return None, tokens
        elif parser_node.token.lexeme == tokens[0].symbol.name:
            logging.debug("matching symbol %s", parser_node.token.lexeme)
            # print(indent(i), "ate terminal", tokens[0].lexeme)
            node.children.append(ParseTreeNode(tokens[0]))
            return node, tokens[1:]
        else:
            logging.debug("did not match terminal %s", parser_node)
            return None, tokens
        return node, tokens

    def match_nonterminal(self, parser_node, tokens, i):
        #self._calls[parser_node.token.symbol.name] += 1
        # print(indent(i),"match_nonterminal '%s' with %d children against %d tokens" % (parser_node.token.lexeme, len(parser_node.children), len(tokens)))
        symbol = self.symbol_table[parser_node.token.lexeme.split('-')[0]]
        node = ParseTreeNode(Token(symbol, parser_node.token.lexeme))
        if not tokens:
            return None, tokens

        child = None

        if parser_node.alternate:
            child, tokens = self.match_alternate(parser_node, tokens, i+1)

        elif parser_node.token.symbol == REPEATING:
            logging.debug("Handle repeating elements (0 or more)")
            # print(indent(i),"Matching Repeating Element", parser_node.token.lexeme)
            child, tokens = self.match_repeating(parser_node, tokens, i+1)

        elif parser_node.token.symbol == SEQUENCE: # match a sequence
            # print(indent(i),"Matching sequence:", token_lexemes(parser_node.children))
            found, tokens = self.match_sequence(parser_node.token.lexeme,
                                                parser_node.children,
                                                tokens, i+1)
            # print(indent(i), "matched sequence, extending with", found)
            node.children.extend(found)

#        elif parser_node.token.symbol == RULE:
#            child, tokens = self.match_rule(parser_node.token.lexeme, tokens, i+1)

        else:
            raise SyntaxError("ran out of options in match_nonterminal")

        if child is not None:
            # print(indent(i), "matched nonterminal, extending with", token_lexemes(child.children))
            node.children.extend(child.children)

        if node.children:
            return node, tokens
        else:
            return None, tokens



    def match_repeating(self, parser_node, tokens, i):
        #self._calls['match_repeating'] += 1
        token = parser_node.token
        expected = parser_node.children
        node = ParseTreeNode(token)
        logging.debug("match repeating")
        keep_trying = True
        while keep_trying:
            # print(indent(i),"match repeating for", token, expected, tokens)
            addends, tokens = self.match_sequence(token.lexeme, expected,
                                                  tokens, i+1)
            # print(indent(i),"from match_sequence:", addends, tokens)
            if addends:
                node.children.extend(addends)
                #print_parsetree(node, i)

            else:
                keep_trying = False
                logging.debug("returning node with %d children",
                              len(node.children))

        if node.children:
            return node, tokens
        else:
            return None, tokens

    def match_alternate(self, parser_node, tokens, i):
        """ParserNode is a node with OR in its children."""
        #self._calls['match_alternate'] += 1
        logging.debug("match_alternate for %s", parser_node)
        if not tokens:
            return None, tokens

        alternates = split_by_or(parser_node.children)

        node = ParseTreeNode(parser_node.token)

        for alternate in alternates:
            preview = tokens[0] if tokens else "No tokens"
            logging.debug("trying alternate: %s against %s",
                          token_lexemes(alternate), preview)
            # print(indent(i),"trying alternate", token_lexemes(alternate))
            found, tokens = self.match_sequence(parser_node.token.lexeme,
                                                alternate, tokens, i+1)
            if found:
                logging.debug("..got it!")
                # print(indent(i), "matched alternate", token_lexemes(alternate))
                node.children.extend(found)
                #print_parsetree(node, i+1)
                return node, tokens
            else:
                # print(indent(i), "did not match alternate", token_lexemes(alternate))
                logging.debug("..nope")
        # print(indent(i),"all alternates failed")
        return None, tokens


    def match_sequence(self, name, originals, tokens, i):
        """returns a list of ParseTreeNodes and a list of remaining tokens"""
        #self._calls['match_sequence'] += 1
        expected = list(originals) # should create a copy

        found = []

        while expected:
            this = expected[0]
            if not tokens:
                # print(indent(i),"sequence '%s' out of tokens, bailing" % name)
                return found, tokens
            # print(indent(i), "expecting in '%s': '%s', got '%s'" % (name, this.token.lexeme, tokens[0].lexeme), end='')
            # print(" (%d expected items)" % (len(expected)+1)) # add one because we popped the value from expected
            if tokens:
                logging.debug("expecting: %s got %s", this, tokens[0])
            else:
                logging.debug("expecing: %s with no tokens", this)
            node, tokens = self.match(this, tokens, i+1)
            if node is not None:
                # print(indent(i), "found", token_lexemes(node.children))
                found.extend(node.children)
            else:
                # print(indent(i), "sequence failed on", this.token.lexeme)
                return  found, tokens
            expected.pop(0)
            # print(indent(i), 'end of sequence loop, expecting: %s against %s' % (token_lexemes(expected), lexemes(tokens)), originals)

         #print(indent(i),"Out of expectations. Node has", len(found), "child%s" % ("" if len(found)==1 else "ren"))
        return found, tokens

def print_node(node, ind=0):
    print(indent(ind), str(node))
    for child in node.children:
        print_node(child, ind+2)

def print_xml(node, ind=0):
    if node.children:
        print("{0}<{1}>".format(indent(ind), node.token.lexeme))
        for child in node.children:
            print_xml(child, ind+2)
        print("{0}</{1}>".format(indent(ind), node.token.lexeme))
    else:
        print("{0}<{1}> {2} </{1}>".format(indent(ind), node.token.symbol.name.lower(), node.token.lexeme))


if __name__ == '__main__':

    from coder import Coder
    text = """statements := assignment { assignment } ;
            assignment := VAR STORE expr STOP;
            expr := term {(PLUS | MINUS) term};
            term := factor {(MUL | DIV) factor};
            factor := INTEGER | VAR | OP expr CP;
            VAR := [a-z]+;
            INTEGER := [0-9]+;
            STORE := <-;
            PLUS := [+];
            MINUS := [\-];
            MUL := [*];
            DIV := [/];
            STOP := [\.];
            OP := [(];
            CP := [)];
            """

    p = EBNFParser(text)
    program = "a <- 2*7+3*2 . \nb<-a/2."
    print(list(p.tokenizer(program)))
    node, detritus = p.parse_text(program)
    print_xml(node)
    print(detritus)

    class MathCoder(Coder):
        encode_integer = Coder.encode_terminal
        encode_op = Coder.handle_terminal
        
        encode_mul = encode_div = encode_plus = encode_minus = Coder.handle_terminal
        encode_parens = Coder.do_nothing
        encode_op = encode_cp = Coder.do_nothing
        encode_var = Coder.handle_terminal
        encode_factor = Coder.handle_children
        encode_statements = Coder.handle_children
        encode_expr = Coder.handle_binary_node
        encode_term = Coder.handle_binary_node

        def encode_assignment(self, node):
            #print("assignment", node, token_lexemes(node.children))
            variable, op, stuff, stop = node.children
            self.handle_node(stuff)
            self.code.append("{}'".format(variable.token.lexeme))

    coder = MathCoder()
    code = coder.encode(node)
    print("Code:", code)

    from machine import BaseMachine
    mather = BaseMachine('math')
    mather.feed(code)
    mather.run()
    print(mather.registers)