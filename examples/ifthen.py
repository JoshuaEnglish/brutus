# -*- coding: utf-8 -*-
"""
Sample if - then statement
"""
from stackvm import Parser, Coder, BaseMachine
from stackvm.utils import print_xml
from stackvm.ebnf import token_lexemes

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

ifelse = """if (a) (res <- "true")
else (res <- "false" ) """

ifonly = """ if (a) (res <- "true") """
program = ifonly

print(list(p.tokenizer(program)))
node, detritus = p.parse_text(program)
print_xml(node)
print(detritus)

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
        print("assignment", node, token_lexemes(node.children))
        variable, op, stuff = node.children
        self.handle_node(stuff)
        self.code.append("{}'".format(variable.token.lexeme))

    def encode_ifstmt(self, node):
        print("ifstmt", node, token_lexemes(node.children))
        # target comp if
        # target else comp ife
        if len(node.children) < 8:
            # we've got a regular if statement
            # cond not endif if if_true endif:

            endif = self.make_label('if')

            self.handle_node(node.children[2]) # name
            self.code.append('not')
            self.code.append(endif)

            self.code.append('if')

            self.handle_node(node.children[5]) # if true clause
            self.code.append('{}:'.format(endif))
        else:
            # we've got an ife statemnt
            # cond iftrue iffalse ife iftrue: if_true endif jump iffalse: iffalse endif:
            iftrue = self.make_label('if')
            iffalse =self.make_label('if')
            endif = self.make_label('if')
            self.handle_node(node.children[2]) # name
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



#
coder = IfThenCoder()
code = coder.encode(node)
print("Code:", code)

chooser = BaseMachine('ifthen')
chooser.feed(code)
print(chooser.program)
chooser.run(a=0)
print(chooser.registers)