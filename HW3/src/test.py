def printM(mat, direction, low, high, left, right):
    if direction == [0,1]:
        for ii in range(left, right+1):
            print(mat[high][ii])
        high += 1
        if high >= low and left >= right: return 
        printM(mat, [-1,0], low, high, left, right)
    if direction == [-1,0]:
        for ii in range(high, low+1):
            print(mat[ii][right])
        right -= 1
        if high >= low and left >= right: return 
        printM(mat, [0,-1], low, high, left, right)
    if direction == [0,-1]:
        for ii in range(right, left-1, -1):
            print(mat[low][ii])
        low -= 1
        if high >= low and left >= right: return 
        printM(mat, [1,0], low, high, left, right)
    if direction == [1,0]:
        for ii in range(low, high-1, -1):
            print(mat[ii][left])
        left += 1
        if high >= low and left >= right: return 
        printM(mat, [0,1], low, high, left, right)

mat = [ [1, 2, 3, 4],
        [5, 6, 7, 8],
        [13,14,15,16]]
printM(mat,[0,1],2,0,0,3)
print('')