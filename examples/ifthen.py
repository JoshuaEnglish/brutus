# -*- coding: utf-8 -*-
"""
Sample if - then statement
"""
import logging

from brutus import Parser, Coder, BaseMachine
from brutus.utils import print_xml, print_node

text = """ifstmt := "if" "(" NAME ")" "(" assignment ")" {"else" "(" assignment ")"} ;
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

p = Parser(text)
print_node(p.rules['ifstmt'])
print_node(p.rules['expr'])

ifelse = """if (a) (res <- true)
else (res <- "false" ) """

ifonly = """ if (a) (res <- 1 - 2) """
program = ifonly

print("Tokenized Program:")
print(list(p.tokenizer(program)))
print()

ok, node, detritus = p.parse_text(program)
print("Parsed Concrete Syntax Tree:")
print_xml(node)
print("Detritus:", detritus)
# p.report()


class IfThenCoder(Coder):
    encode_integer = Coder.encode_terminal
    encode_op = Coder.handle_terminal
    encode_parens = Coder.do_nothing

    encode_name = Coder.handle_terminal
    encode_literal = Coder.handle_terminal
    encode_factor = Coder.handle_children
    encode_statements = Coder.handle_children
    encode_expr = Coder.handle_binary_node
    encode_term = Coder.handle_binary_node

    def encode_assignment(self, node):
        variable, op, stuff = node.children
        self.handle_node(stuff)
        self.code.append("{}'".format(variable.token.lexeme))

    def encode_ifstmt(self, node):
        if 'else' not in (t.token.lexeme for t in node.children):
            # we've got a regular if statement
            # cond not endif if if_true endif:

            endif = self.make_label('if')

            self.handle_node(node.children[2])  # name
            self.code.append('not')
            self.code.append(endif)

            self.code.append('if')

            self.handle_node(node.children[5])  # if true clause
            self.code.append('{}:'.format(endif))
        else:
            iftrue = self.make_label('if')
            iffalse = self.make_label('if')
            endif = self.make_label('if')
            self.handle_node(node.children[2])  # name
            self.code.append(iftrue)
            self.code.append(iffalse)

            self.code.append('ife')

            self.code.append('{}:'.format(iftrue))
            self.handle_node(node.children[5])
            self.code.append(endif)
            self.code.append('jump')
            self.code.append('{}:'.format(iffalse))
            self.handle_node(node.children[9])
            self.code.append('{}:'.format(endif))


coder = IfThenCoder()
code = coder.encode(node)
print("\nCode:", code)

chooser = BaseMachine('ifthen')
chooser.feed(code)
print("\nProgram:", chooser.program)
log = logging.getLogger('STACKVM')
# log.setLevel(logging.DEBUG)
# log.addHandler(logging.StreamHandler())

chooser.run(a=1)
print("\nRegisters:", chooser.registers)
chooser.run(a=0)
print("\nRegisters:", chooser.registers)
print("\nStack:", chooser.stack)
