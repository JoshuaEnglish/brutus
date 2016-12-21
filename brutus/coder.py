# -*- coding: utf-8 -*-
"""
Coder

Translates :class:`CSTNodes` to code for the stack machine
"""
from collections import Counter


class Coder(object):
    """Code generation tool. For each token in the concrete syntax tree,
    declare a method ``encode_nnn`` where `nnn` is the name of the token's
    symbol.
    
    Several ``handle_xxx`` methods exist as shortcuts.
    
    Use these to define your own class::
        
        class MathCoder(Coder):
            encode_integer = Coder.encode_terminal
            encode_binop = Coder.handle_terminal
            encode_parens = Coder.do_nothing
            
        mycoder = MathCoder()
        code = mycode.encode(parsed_cst_node)
    
    """

    def __init__(self):
        self.code = []
        self._labels = Counter()
        """Keeps track of individual sequential labels"""

    def make_label(self, name):
        """Returns a new label in a sequence.
        
        Save the results of this call to use later.
        
        Sample:
            self.make_label('subroutine') -> 'subroutine_0'
            self.make_label('subroutine') -> 'subroutine_1'
            
        Please see examples\ifthen.py for an example of this method's use.

        """
        this = self._labels[name]
        label = "{}_{}".format(name, this)
        self._labels[name] += 1
        return label

    def encode(self, cstnode):
        """Return a space deliniated string of items of code.
        Automatically appends 'END' if needed.
        """
        self.handle_node(cstnode)
        if self.code[-1] not in ['.', 'END', 'end']:
            self.code.append('END')
        return " ".join(self.code)

    def handle_terminal(self, node):
        """Adds the token's lexeme to the code output"""
        self.code.append(node.token.lexeme)

    def encode_terminal(self, node):
        """Adds the token's lexeme to the code output"""
        self.code.append(node.token.lexeme)

    def handle_binary_node(self, node):
        """Converts a repeating infix operation to postfix in the code output"""
        self.handle_node(node.children[0])
        idx = 1
        while idx < len(node.children):
            self.handle_node(node.children[idx+1])
            self.handle_node(node.children[idx])
            idx += 2


    def handle_node(self, node):
        """Determines which encode_xxx function to call based on the CSTNode's
        token or symbol name.
        """
        func_name = "encode_{}".format(node.token.lexeme)
        symbol_name = "encode_{}".format(node.token.symbol.name.lower())
        if hasattr(self, func_name):
            getattr(self, func_name)(node)
        elif hasattr(self, symbol_name):
            getattr(self, symbol_name)(node)
        else:
            raise SyntaxError("Coder cannot handle %s" % node)

    def handle_children(self, node):
        """Encodes children, ignoring the parent node"""
        for child in node.children:
            self.handle_node(child)

    def do_nothing(self, node):
        """Ignores the node in code generation."""
        pass