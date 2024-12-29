'''
Input: List A includes n items
       x: query item
Output: Indice i of A has x query
        If not found: i = -1
'''
def LinearSearch(items, x):
    for index, item in enumerate(items):
        if item == x:
            return index
    return -1
if __name__=='__main__':
    A = [4, 2, 3, 1, 5]
    x = 4
    print(LinearSearch(A, x))
    
    
        