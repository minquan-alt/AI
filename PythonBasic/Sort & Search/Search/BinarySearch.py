'''
Input: List A includes n ordered items
       x: query item
Output: Indice i of A has x query
        If not found: i = -1
'''
def BinarySearch(items, left_index, right_index, x):
    '''
    right_indices: 0 - start of ordered list
    left_indices: last index of array ~ length of items - 1
    x: query item
    '''
    if left_index > right_index:
        return -1
    mid_index = int((right_index + left_index) / 2)
    if x < items[mid_index]:
        return BinarySearch(items, left_index, mid_index - 1, x)
    elif x > items[mid_index]:
        return BinarySearch(items, mid_index + 1, right_index, x)
    return mid_index
if __name__=='__main__':
    A = [1, 2, 3, 4, 5]
    x = 3
    print(BinarySearch(A, 0, len(A) - 1, x))
    
    
    