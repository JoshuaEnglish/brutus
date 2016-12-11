# -*- coding: utf-8 -*-
"""
Utility functions for Brutus.

The print_xxx functions call for a single ``node`` parameter. 
This parameter is assumed to have an iterable 11children``.
"""

from itertools import islice, cycle

DOTS = " .  "
def indent(length):
    """indent(length)
    Return a string of length ``length`` with dot characters spaced into
    columns.
    This is used in printing tree structures
    """
    return ''.join(islice(cycle(DOTS), length))

def print_node(node, ind=0):
    """generic node printer"""
    print(indent(ind), str(node))
    for child in node.children:
        print_node(child, ind+2)

def print_xml(node, ind=0):
    """produces XML output for a node with a token and children"""
    if node.children:
        print("{0}<{1}>".format(indent(ind), node.token.lexeme))
        for child in node.children:
            print_xml(child, ind+2)
        print("{0}</{1}>".format(indent(ind), node.token.lexeme))
    else:
        print("{0}<{1}> {2} </{1}>".format(indent(ind), 
              node.token.symbol.name.lower(), node.token.lexeme))

    