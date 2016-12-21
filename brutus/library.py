# -*- coding: utf-8 -*-
"""
Libraries for Brutus's :class:`Machine`.
"""

import collections
import operator as OP


class VMLibrary:
    """This class allows users to import from predefined libraries.
    Importing a library may generate errors if you use a keyword that is in
    the library before importing the library.
    """

    def __init__(self, name, version):
        self.name = str(name)
        self.version = float(version)
        self.reserved_words = []
        self.exports = []

    def __repr__(self):
        return "<VMLibrary '%s' version %0.1f>" % (self.name, self.version)

    def add_export(self, matches, func, caller=None):
        """add_export(list, string, [string])
        The first item is a list of string objects which can be used to call the method,
        which is determined by the first string. The optional second string is passed
        to the method when called.

        This method follows the same rules as the VM.add_rule method.
        """

        if not isinstance(matches, (list, tuple)):
            raise ValueError("Matches must be list or tuple")

        if not all(isinstance(x, str) for x in matches):
            raise ValueError("Non-String in matches list")

        for match in matches:
            if match.upper() in self.reserved_words:
                raise ValueError("Reserved Word %s reused" % (match))
            if match.split()[0] != match:
                raise ValueError("Match %s cannot contain spaces" % (match))

        matches = [m.upper() for m in matches]

        ### Make sure func is valid
        if not isinstance(func, str):
            raise ValueError("Cannot convert function name to string")
        func = func.strip()
        if not func:
            raise ValueError("Function must contain alphanumeric characters")
        if func.split()[0] != func:
            raise ValueError("Function cannot contain white space")

        ### Make sure that caller is valid (if there)
        if caller:
            if not isinstance(caller, (str, collections.Callable)):
                raise ValueError("Caller must be a string or callable")
            if isinstance(caller, str):
                caller = caller.strip()
                if caller.split()[0] != caller:
                    raise ValueError("Caller cannot contain whitespace")

        ### Add the matches to
        self.reserved_words.extend(matches)
        self.exports.append((matches, func, caller))

    def methods(self):
        """Returns a list of method names that are available from this library"""
        res = []
        for match, func, caller in self.exports:
            if func not in res: 
                res.append(func)
        return res



class FiveFunctionLibrary(VMLibrary):
    """FiveFunctionLibrary
    Import this library using the ImportLibrary method of any VM
    instance.

    This defines five basic integer functions (Keywords in parenthesis):
        addition (plus, +)
        subtraction (minus, -)
        multiply (times,*)
        division (div, /) (Note: This is integer division which drops the remainder)
        remainder (mod, %)
    """
    def __init__(self):
        VMLibrary.__init__(self, 'FiveFunctionLibrary', 0.1)

        self.add_export(['plus', '+'], 'do_5f_op', OP.add)
        self.add_export(['minus', '-'], 'do_5f_op', OP.sub)
        self.add_export(['times', '*'], 'do_5f_op', OP.mul)
        self.add_export(['div', '/'], 'do_5f_op', OP.truediv)
        self.add_export(['mod', '%'], 'do_5f_op', OP.mod)

    def do_5f_op(self, op, parent):
        """perform one of the five basic binary operations;
        addition, subtraction, multiplication, integer division, remainder
        """
        right = parent.stack.pop()
        right = int(parent.registers[right]
                    if right in parent.registers
                    else right)

        left = parent.stack.pop()
        left = int(parent.registers[left]
                   if left in parent.registers
                   else left)

        res = op(left, right)
        parent.stack.push(res)



class ComparisonsLibrary(VMLibrary):
    """ComparisonsLibrary
    Import this library using the ImportLibrary method of any VM
    instance.

    This defines six basic comparison functions (Keywords in parenthesis):
        less than (lessthan, lt, <)
        greater than (greaterthan, gt, >)
        less than or equal (lte, <=, =<)
        greater than or equal (gte, =>, >=)
        not equal to (neq, !=, <>)
        equal to (eq, ==, =) (note: can use either form, since storage is handled differently)
    """
    def __init__(self):
        VMLibrary.__init__(self, "ComparisonsLibrary", 0.1)

        self.add_export(['lessthan', 'lt', '<'], 'do_cmp_op', '<')
        self.add_export(['greaterthan', 'gt', '>'], 'do_cmp_op', '>')
        self.add_export(['lte', '<=', '=<'], 'do_cmp_op', '<=')
        self.add_export(['gte', '>=', '=>'], 'do_cmp_op', '>=')
        self.add_export(['neq', '!=', '<>'], 'do_cmp_op', '!=')
        self.add_export(['eq', '==', '='], 'do_cmp_op', '==')

    def do_cmp_op(self, op, parent):
        try:
            a, b = parent.stack.pop(), parent.stack.pop()
        except IndexError:
            raise IndexError("Cannot pop two items for %s operator" % op)
        try:
            a = int(a)
        except:
            raise ValueError("%s not valid input for %s" % (a, op))
        try:
            b = int(b)
        except:
            raise ValueError("%s not valid input for %s" % (b, op))

        opstring = '%s %s %s' % (b, op, a)

        parent.stack.push(str(int(eval(opstring))))


class ControlOperationsLibrary(VMLibrary):
    """ControlOperationsLibrary
    Import this library using the ImportLibrary method of any VM
    instance.

    This defines three basic binary functions (Keywords in parenthesis):
        if ... then ... (if)
        if ... then ... else (ife)
        jump (jump, goto)
    """
    def __init__(self):
        VMLibrary.__init__(self, "ControlOperationsLibrary", 0.1)

        self.add_export(['if',], 'do_if_control')
        self.add_export(['ife',], 'do_ife_control')
        self.add_export(['jump', 'goto'], 'do_jump_control')

    def do_if_control(self, caller, parent):
        "cond if_true_line if"
        line, value = parent.stack.pop(), parent.stack.pop()
        if isinstance(line, str) and not line.isdigit():
            raise ValueError("Possibly missing label: %s" % line)
        if value:
            parent.go_to_instruction(int(line))

    def do_ife_control(self, caller, parent):
        "cond if_true_line if_false_line ife"
        line1, line2, value = parent.stack.pop(), parent.stack.pop(), parent.stack.pop()
        if isinstance(line1, str) and not line1.isdigit():
            raise ValueError("Possibly missing label: %s" % line1)
        if isinstance(line2, str) and not line2.isdigit():
            raise ValueError("Possibly missing label: %s" % line2)
        if value:
            parent.go_to_instruction(int(line2))
        else:
            parent.go_to_instruction(int(line1))

    def do_jump_control(self, caller, parent):
        "target jump"
        line = parent.stack.pop()
        if isinstance(line, str) and not line.isdigit():
            raise ValueError("Possible missing label: %s" % line)
        parent.go_to_instruction(int(line))
