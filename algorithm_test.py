import numpy as np
import time
from datetime import datetime
# import plotly.express as px
from typing import Any, Callable, List, Dict
from multiprocessing import Pool, TimeoutError
# import ipyparallel as ipp
import multiprocessing
import json
import pickle

_arr_len = 96
# 96 массивов с 10 элементами от 1 до 10
arr10 = [np.random.randint(low=1, high=10, size=10) for i in range(_arr_len)]
arr10s = [ np.sort(arr10[i], kind="mergesort") for i in range(_arr_len)] #отсортированне массивы для проверки

# 96 массивов с 25 элементами от 1 до 25
arr25 = [np.random.randint(low=1, high=25, size=25) for i in range(_arr_len)]
arr25s = [ np.sort(arr25[i], kind="mergesort") for i in range(_arr_len)]#отсортированне массивы для проверки

# 96 массивов с 500 элементами от 1 до 500
arr5h = [np.random.randint(low=1, high=500, size=500) for i in range(_arr_len)]
arr5hs = [ np.sort(arr5h[i], kind="mergesort") for i in range(_arr_len)]#отсортированне массивы для проверки

# 96 массивов с 1000 элементами от 1 до 1000
arr1th = [np.random.randint(low=1, high=1000, size=1000) for i in range(_arr_len)]
arr1ths = [ np.sort(arr1th[i], kind="mergesort") for i in range(_arr_len)]#отсортированне массивы для проверки

# 96 массивов с 5000 элементами от 1 до 5000
arr5th = [np.random.randint(low=1, high=5000, size=5000) for i in range(_arr_len)]
arr5ths = [ np.sort(arr5th[i], kind="mergesort") for i in range(_arr_len)]#отсортированне массивы для проверки

swap_count = 0

def swap(arr: np.ndarray, a: int, b: int):
    swap_count_change()
    arr[a], arr[b] = arr[b], arr[a]

def swap_count_change():
    global swap_count
    swap_count += 1

#сортировка простыми вставками
def insertion_sort(arr: np.ndarray):
    n = arr.shape[0]
    for i in range(1, n):
        key = arr[i]
        j = i - 1 
        while (j >= 0 and arr[j] > key):
            arr[j + 1] = arr[j]
            swap_count_change()
            j = j - 1
        arr[j + 1] = key
        swap_count_change()

#сортировка простым обменом
def exchange_sort(arr: np.ndarray):
    n = arr.shape[0]
    for i in range(0, n-1):
        for j in range(i+1, n):
            if (arr[i] > arr[j]):
                swap(arr, j, i)

#сортировка простым выбором 
def selection_sort(arr: np.ndarray):
    n = arr.shape[0]
    for i in range(n-1):
        _min = i
        for j in range(i+1, n):
            if (arr[j] < arr[_min]):
                _min = j
        if (_min != i):
            swap(arr, i, _min)

#сортировка пирамидальная
def heapify(arr: np.ndarray, n: int, i: int):
    large = i
    l = 2 * i + 1
    r = l + 1
    if (l < n and arr[i] < arr[l]):
        large = l
    if (r < n and arr[large] < arr[r]):
        large = r
    if large != i:
        swap(arr, i, large)
        heapify(arr, n, large)
    
def heap_sort(arr: np.ndarray):
    n = arr.shape[0]
    
    #построение дерева
    for i in range(n//2, -1, -1):
        heapify(arr, n, i)

    # Один за другим извлекаем элементы
    for i in range(n-1, 0, -1):
        swap(arr, i, 0)
        heapify(arr, i, 0)

#быстрая сортировка
def partition(arr: np.ndarray, start: int, end: int) -> int:
    pivot = arr[(start+end) // 2]
    i=start
    j=end

    while i <=j:
        while arr[i] < pivot:
            i+=1
        while arr[j] > pivot:
            j-=1
        if i <= j:
            swap(arr, i, j)
            i+=1
            j-=1
    return i

def quick_sort_f(arr: np.ndarray, start: int, end: int):
    if start < end:
        temp = partition(arr, start, end)
        quick_sort_f(arr, start, temp - 1)
        quick_sort_f(arr, temp, end)

def quick_sort(arr: np.ndarray):
    quick_sort_f(arr, 0, arr.shape[0] - 1)

#проверка времени выполнения блока данных
def check_function_time(arr: List[np.ndarray], algorithm: Callable) -> float:
    arr_time = []
    arr_swap = []
    for i in arr:
        global swap_count
        swap_count = 0
        star_time = time.time()
        algorithm(i)
        end_time = time.time()
        arr_swap.append(swap_count)
        arr_time.append(end_time - star_time)
    return [arr_time, arr_swap]

#запуск процессов
def process_start( arr: List[np.ndarray], algorithm: Callable) -> Dict[str, List[float]]:
    time_arr = []
    swap_arr = []
    with Pool(processes=8) as pool:
        res =  pool.starmap(check_function_time, [(arr[i*12:(i*12)+12], algorithm) for i in range(8)])
    for i in res:
        time_arr.extend(i[0])
        swap_arr.extend(i[1])
    return [time_arr, swap_arr]

if __name__ == '__main__':
    time_arr = {}
    sort_algorithm = [insertion_sort, exchange_sort, selection_sort, heap_sort, quick_sort]
    start = time.time()
    # sort_algorithm = [quick_sort,heap_sort]
    data = [arr10, arr25, arr5h, arr1th, arr5th]
    for algorithm in sort_algorithm:
        algorithm_name = algorithm.__name__
        time_arr[algorithm_name] = {}
        start_alg = time.time()
        for arr in data:
            time_and_swap_arr = process_start(arr, algorithm)
            time_arr[algorithm_name][len(arr[0])] = {"swap": time_and_swap_arr[1], "time": time_and_swap_arr[0]}
        print(algorithm_name, time.time() - start_alg)
    end = time.time()
    print(end - start)
    with open("time_and_swap_arr.json", "w") as outfile:
        json.dump(time_arr, outfile)
    input_arr = {"arr10": [i.tolist() for i in arr10], 
                    "arr25": [i.tolist() for i in arr25], 
                    "arr5h": [i.tolist() for i in arr5h], 
                    "arr1th": [i.tolist() for i in arr1th], 
                    "arr5th": [i.tolist() for i in arr5th]}
    with open("input_file.json", "w") as input_file:
        json.dump(input_arr, input_file)
#         time_arr[algorithm_name].append(process_start(algorithm, arr.copy()))с