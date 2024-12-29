def InterpolationSearch(items, left, right, x):
    '''
    right_indices: 0 - start of ordered list
    left_indices: last index of array ~ length of items - 1
    x: query item
    '''
    if left >= right:
        return -1
    pos = left + int(((right - left) * (x - items[left])) / (items[right] - items[left]))
    if x < items[pos]:
        return InterpolationSearch(items, left, pos - 1, x)
    elif x > items[pos]:
        return InterpolationSearch(items, pos + 1, right, x)
    return pos
if __name__=='__main__':
    A = [1, 2, 3, 4, 4.5, 4.6, 4.7, 5]
    x = 5
    print(InterpolationSearch(A, 0, len(A) - 1, x))