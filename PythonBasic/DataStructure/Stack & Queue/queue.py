class Queue():
    def __init__(self):
        self.queue = []
    
    def add(self, item):
        # Add an item to the end of the queue
        self.queue.append(item)
    
    def remove(self):
        # Remove and return the item from the front of the queue
        if self.is_empty():
            raise Exception("Queue is empty")
        return self.queue.pop(0)
    
    def peek(self):
        # Return the front item without removing it
        if self.is_empty():
            raise Exception("Queue is empty")
        return self.queue[0]
    
    def is_empty(self):
        # Check if the queue is empty
        return len(self.queue) == 0
    
    def size(self):
        # Return the number of items in the queue
        return len(self.queue)

# Create a new queue
my_queue = Queue()

# Add items to the queue
my_queue.add(10)
my_queue.add(20)
my_queue.add(30)

# Check the front item of the queue
print(my_queue.peek())  # 10

# Remove the front item from the queue
print(my_queue.remove())  # 10

# Check the size of the queue
print(my_queue.size())  # 2

# Check if the queue is empty
print(my_queue.is_empty())  # False
