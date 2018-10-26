for x in range(0, 3):
    for y in range(3):
        print('#', end='')
    print()


for x in range(1, 10, 2):
    for y in range(1, 10, 2):
        print(x * y, '\t', end='')
    print()


word_array = ['A', 'BB', 'CCC']
for w in word_array:
    print(w, len(w))


# While loop with else condition
w = 0
while w < 2 ** 2:
    print(w, " within loop.")
    w += 1
else:
    print(w, " within else cond.")
