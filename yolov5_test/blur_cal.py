
def generate_new_array(arr, offset=2): # 1左右两边也为1
    new_arr = [0] * len(arr)

    for i in range(len(arr)):
        if arr[i] == 1:
            for j in range(-offset, offset+1):
                if i + j >= 0 and i + j < len(arr):
                    new_arr[i+j] = 1

    return new_arr


def generate_new_array0(arr, num_elements=10, threshold=0.5): # 为1后5个元素，1的比例
    new_arr = [0] * len(arr)
    i = 0

    while i < len(arr):
        if arr[i] == 0:
            new_arr[i] = 0
            i += 1
        elif arr[i] == 1:
            ones_count = sum(arr[i+1:i+num_elements+1]) if i+num_elements < len(arr) else sum(arr[i+1:])
            if ones_count >= threshold * min(len(arr)-i-1, num_elements):
                new_arr[i:i+num_elements+1] = [1] * min(len(arr)-i, num_elements+1)
            else:
                new_arr[i:i+num_elements+1] = arr[i:i + min(len(arr)-i, num_elements+1)]
            i += num_elements+1

    return new_arr

def generate_new_array1(arr): # 清理1的连续性低于4的区间
    new_arr = arr.copy()
    start = -1
    end = -1

    for i in range(len(arr)):
        if arr[i] == 1:
            if start == -1:
                start = i
            end = i
        else:
            if start != -1:
                if end - start + 1 < 4:
                    new_arr[start:end+1] = [0] * (end - start + 1)
                start = -1
                end = -1

    if start != -1 and end != -1 and end - start + 1 < 4:
        new_arr[start:end+1] = [0] * (end - start + 1)

    return new_arr

def generate_new_array2(arr): # 清理0的连续性低于4的区间
    new_arr = arr.copy()
    start = -1
    end = -1

    for i in range(len(arr)):
        if arr[i] == 0:
            if start == -1:
                start = i
            end = i
        else:
            if start != -1:
                if end - start + 1 < 4:
                    new_arr[start:end+1] = [1] * (end - start + 1)
                start = -1
                end = -1

    if start != -1 and end != -1 and end - start + 1 < 4:
        new_arr[start:end+1] = [1] * (end - start + 1)

    return new_arr

def blur_cal(arr):
    arr = generate_new_array(arr)
    arr = generate_new_array0(arr)
    arr = generate_new_array1(arr)
    arr = generate_new_array2(arr)
    print('result_arr:', len(arr), arr)
    return arr