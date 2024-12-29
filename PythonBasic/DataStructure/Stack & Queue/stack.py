class Stack():
    def __init__(self):
        self.stack = []
    
    def add(self, item):
        # Add an item to the top of the stack
        self.stack.append(item)
    
    def remove(self):
        # Remove and return the top item from the stack
        if self.is_empty():
            raise Exception("Stack is empty")
        return self.stack.remove()
    
    def peek(self):
        # Return the top item without removing it
        if self.is_empty():
            raise Exception("Stack is empty")
        return self.stack[-1]
    
    def is_empty(self):
        # Check if the stack is empty
        return len(self.stack) == 0
    
    def size(self):
        # Return the number of items in the stack
        return len(self.stack)

# Create a new stack
my_stack = Stack()

# add items onto the stack
my_stack.add(10)
my_stack.add(20)
my_stack.add(30)

# Check the top item of the stack
print(my_stack.peek())  # 30

# remove the top item from the stack
print(my_stack.remove())  # 30

# Check the size of the stack
print(my_stack.size())  # 2

# Check if the stack is empty
print(my_stack.is_empty())  # False
