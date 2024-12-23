import timeit

# Cách cũ của tôi
def swap_case_original(s):
    result = ''
    for x in s:
        if x.isupper():
            result += x.lower()
        else:
            result += x.upper()
    return result
def swap_case_list(s):
    result = []
    for x in s:
        if x.isupper():
            result.append(x.lower())
        else:
            result.append(x.upper())
    return ''.join(result)
# Cách tối ưu
def swap_case_optimized(s):
    result = [x.lower() if x.isupper() else x.upper() for x in s]
    return ''.join(result)

# Chuỗi test
s = "a" * 10000 + "A" * 10000

# Đo thời gian
print("Original:", timeit.timeit(lambda: swap_case_original(s), number=100))
print("Optimized:", timeit.timeit(lambda: swap_case_list(s), number=100))
print("Twice Optimized:", timeit.timeit(lambda: swap_case_optimized(s), number=100))
