# -*- coding: utf-8 -*-
"""
Sample multi-line 4-function calculator.

Results from previous lines can be used in later lines.
"""
import logging

from brutus import Parser, Coder, BaseMachine
from brutus.utils import print_xml

text = """statement := assignment | expr ;
        assignment := VAR STORE expr STOP [COMMENT];
        expr := term { EXPROP term };
        term := factor { TERMOP factor };
        factor := INTEGER | VAR | "(" expr ")";
        VAR := [a-z]+;
        INTEGER := -?[0-9]+;
        STORE := <-;
        EXPROP := [-+];
        TERMOP := [*/];
        STOP := [\.];
        PARENS := [()];
        COMMENT := #.*$;
        """


class MathCoder(Coder):
    encode_integer = Coder.encode_terminal

    encode_exprop = Coder.handle_terminal
    encode_termop = Coder.handle_terminal
    encode_parens = Coder.do_nothing
    encode_var = Coder.handle_terminal
    encode_factor = Coder.handle_children
    encode_statement = Coder.handle_children
    encode_expr = Coder.handle_binary_node
    encode_term = Coder.handle_binary_node
    encode_comment = Coder.do_nothing

    def encode_assignment(self, node):
        variable, op, stuff, stop = node.children[:4]
        self.handle_node(stuff)
        self.code.append("{}'".format(variable.token.lexeme))


p = Parser(text)

mather = BaseMachine('math')


def do(text):
    print(list(p.tokenizer(text)))
    logging.root.setLevel(logging.DEBUG)
    ok, node, detritus = p.parse_text(text)
    if detritus:
        print("OOPS", detritus)
        return None
    print_xml(node)
    p.report()
    coder = MathCoder()
    code = coder.encode(node)
    print(code)
    mather.reset()
    mather.feed(code)
    mather.run()
    print(mather.stack.items)
    print(mather.registers)


do('3 + 4')
do('a <- 5.')
