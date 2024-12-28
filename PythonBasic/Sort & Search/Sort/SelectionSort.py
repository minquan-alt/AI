def SelectionSort(a):
    len_a = len(a)
    for i in range(len_a - 1):
        current_minimum_idx = i
        for j in range(i + 1, len_a):
            if a[j] < a[current_minimum_idx]:
                current_minimum_idx = j
        a[i], a[current_minimum_idx] = a[current_minimum_idx], a[i]
    return a

A = ['B', 'A', 'D', 'G', 'F', 'C', 'E']
print(SelectionSort(A))