# -*- coding: utf-8 -*-
"""Stack

Simple Last-In-First-Out stack.
"""


class Stack:
    """Stack(items)
    Simple LIFO stack. Supports pushing elements and lists, and poping from
    the top of the stack.
    """

    def __init__(self, items=None):
        self.items = items or []

    def push(self, item):
        """stack.push(item)
        Add the item to the stack. If the item is a list type or tuple type,
        it will add all of the elements to that list. This is done through
        list extension, so no item in the stack will be a list itself.
        """

        if isinstance(item, (list, tuple)):
            self.items.extend(item)
        else:
            self.items.append(item)

    def pop(self):
        """stack.pop()
        Removes the top of the stack and returns that value.
        Raises IndexError if the stack is empty
        """
        return self.items.pop()

    @property
    def is_empty(self):
        """stack.isEmpty()
        Returns a boolean value
        """
        return self.items == []

    def __str__(self):
        return '[Stack: %s]' % ', '.join([str(item) for item in self.items])

    def __getitem__(self, item):
        return self.items[item]

    def __contains__(self, item):
        return self.items.__contains__(item)

    def clear(self):
        """clear()
        Removes all values from the stack
        """
        self.items = []

    def __len__(self):
        return len(self.items)

    def top(self):
        """top()
        Returns the top value of the stack without removing from the stack
        """
        return self.items[-1]
