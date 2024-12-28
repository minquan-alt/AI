def InsertionSort(items):
    len_items = len(items)
    if len_items < 1:
        return items
    i = 1
    for i in range(1, len_items):
        current_item = items[i]
        j = i - 1
        while j >= 0 and items[j] > current_item:
            items[j + 1] = items[j]
            j -= 1
        items[j + 1] = current_item
    return items

A = ['B', 'A', 'D', 'G', 'F', 'C', 'E', 'A', 'A', 'F']
print(InsertionSort(A))