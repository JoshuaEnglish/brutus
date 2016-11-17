# -*- coding: utf-8 -*-
"""
Sample multi-line 4-function calculator.

Results from previous lines can be used in later lines.
"""
from stackvm import Parser, Coder, BaseMachine
from stackvm.utils import print_xml

text = """statement := assignment | expr ;
        assignment := VAR STORE expr STOP;
        expr := term {("+" | "-") term};
        term := factor {("*" | "/") factor};
        factor := INTEGER | VAR | "(" expr ")";
        VAR := [a-z]+;
        INTEGER := -?[0-9]+;
        STORE := <-;
        BINOP := [+\-*/];
        STOP := [\.];
        PARENS := [()];

        """


class MathCoder(Coder):
    encode_integer = Coder.encode_terminal

    encode_binop = Coder.handle_terminal
    encode_parens = Coder.do_nothing
    encode_var = Coder.handle_terminal
    encode_factor = Coder.handle_children
    encode_statement = Coder.handle_children
    encode_expr = Coder.handle_binary_node
    encode_term = Coder.handle_binary_node

    def encode_assignment(self, node):
        #print("assignment", node, token_lexemes(node.children))
        variable, op, stuff, stop = node.children
        self.handle_node(stuff)
        self.code.append("{}'".format(variable.token.lexeme))

p = Parser(text)

mather = BaseMachine('math')


def do_something(text):
    print(list(p.tokenizer(text)))
    node, detritus = p.parse_text(text)
    if detritus:
        print("OOPS", detritus)
        return None
    print_xml(node)
    coder = MathCoder()
    code =coder.encode(node)
    print(code)
    mather.reset()
    mather.feed(code)
    mather.run()
    print(mather.stack.items)
    print(mather.registers)