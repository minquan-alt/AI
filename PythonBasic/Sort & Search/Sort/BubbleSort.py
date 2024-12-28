def BubbleSort(a):
    len_a = len(a)
    for i in range(len_a):
        swapped = False
        for j in range(len_a - i - 1):
            if a[j] > a[j+1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a

A = ['B', 'A', 'D', 'G', 'F', 'C', 'E', 'A', 'A', 'F']
print(BubbleSort(A))