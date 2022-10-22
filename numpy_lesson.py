import numpy as np

data = [1, 2, 3, 4, 5]

arr = np.array(data) # создание из list

print(arr)
print(arr.shape)
print(arr.dtype)
print(arr.ndim)
print()

"""
Типы данных NumPy:
np.int64
np.float32
np.complex
np.bool
np.object
np.string
np.unicode
"""

arr2 = np.array([1, 2, 3, 4, 5])
print(arr2)
print(arr2.shape)
print(arr2.dtype)
print(arr2.ndim)
print()

arr3 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print(arr3)
print(arr3.shape)
print(arr3.dtype)
print(arr3.ndim)
print(len(arr3))
print(arr3.size)
print()

arr3 = arr.astype(np.int64)
print(arr3)
print(arr3.dtype)
print()

arr4 = np.arange(0, 20, 1.5)
print(arr4)
print()

arr5 = np.linspace(0, 2, 5) #5 numbers from 0 to 2
print(arr5)
print()

arr5 = np.linspace(0, 2, 50)
print(arr5)
print()

random_arr = np.random.random((5,))
print(random_arr)
print()

random_arr2 = np.random.random_sample((5,))
print(random_arr2)
print()

# Диапазон случайных чисел от a до b (b > a): (b - a) * np.random() + a
random_arr3 = (10 - 9) * np.random.random_sample((5,)) - 9
print(random_arr3)
print()

#Operations
arr = np.array([1, 2, 3, 4, 5])

print(np.sqrt(arr))

print(np.sin(arr))

print(np.cos(arr))

print(np.log(arr))

print(np.exp(arr))
print()

a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])
print(a)
print(b)
c = a + b
print(c)
print(np.add(a, b))
print(a * b)
print(np.multiply(a, b))
print(a - b)
print(np.subtract(a, b))
print(a / b)
print(np.divide(a, b))
print()

arr = np.array([1, 2, 3, 4, 5])
print(arr * 2) # с обычным листом так не работает
print(arr ** 2) # с обычным листом так не работает
print()


#ФУНКЦИИ АГРЕГАТОРЫ
arr = np.random.randint(-5, 10, 10)
print(arr)
print(arr.max())
print(arr.min())
print(arr.mean()) # среднее значение
print(arr.sum())
print(arr.std()) #стандартное отклонение
print(np.median(arr)) #медиана
print(arr < 2) #проверка каждого эл-та
print()

#МАНИПУЛЯЦИИ С МАССИВАМИ
arr = np.array([100, 1, 2, 3, 4, 5])
print(np.insert(arr, 2, 10))
print(np.delete(arr, 2))
print(np.sort(arr))
arr2 = np.array([-1, -2, -3, -4])
print(np.concatenate((arr, arr2)))
print(np.concatenate((arr2, arr)))
print(np.array_split(arr, 4))#hsplit, vsplit
print()

#ИНДЕКСЫ И ОДНОМЕРНЫЕ МАССИВЫ
arr = np.array([1, 2, 3, 4, 1, 5])
arr[0] = 0
print(arr)
print(arr[2])
print(arr[0:2])
print(arr[::-1])
print(arr[arr < 4])
print(arr[(arr < 2) & (arr > 0)])
arr[1:4] = 0
print(arr)
print()

#МАТРИЦЫ И МНОГОМЕРНЫЕ МАССИВЫ
matrix = np.array([(1, 2, 3), (4, 5, 6)], dtype=np.float64)
print(matrix)

matrix = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
print(matrix)
print(matrix.shape)
print(matrix.ndim)
print(matrix.size)

print(matrix.reshape(1, 9))
matrix2 = np.random.random((2, 2))
print(matrix2)
new_matrix = np.arange(16).reshape(2, 8)
print(new_matrix)
print()

#СОЗДАНИЕ СПЕЦИАЛЬНЫХ МАТРИЦ
print(np.zeros((2, 3)))
print(np.ones((3, 3)))
print(np.eye(5))
print(np.full((3, 3), 9))
print(np.empty((3,2)))
print()

#ОПЕРАЦИИ НАД МАТРИЦАМИ
matrix1 = np.array([(1, 2), (3, 4)])
print(matrix1)
print()
matrix2 = np.array([(5, 6), (7, 8)])
print(matrix2)
print(matrix1 + matrix2) # np.add
print(matrix1 - matrix2)
print(matrix1 * matrix2)
print(matrix1 / matrix2)
print(matrix1.dot(matrix2)) #скалярное произведение
print()


# AXISES
matrix1 = np.array([(1, 2), (3, 4)])
print(np.delete(matrix1, 1, axis=0)) #удаляем 1 строку
print(np.delete(matrix1, 1, axis=1))
print(np.sin(matrix1))
print(np.log(matrix1))
print(matrix1.sum())
print(matrix1.max())
print(matrix1.max(axis=0)) #применяем к каждому столбцу
print(matrix1.max(axis=1)) #применяем к каждой строке
print(np.mean(matrix1, axis=0)) #среднее значение по строкам
print(np.mean(matrix1, axis=1)) #среднее значение по столбцам
print(matrix1 ** 2)
print()

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr1)
print()
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
print(arr2)
print()
arr = np.concatenate((arr1, arr2), axis=1) # добавляем как столбцы
print(arr)
arr = np.concatenate((arr1, arr2), axis=0) # добавляем как строки
print(arr)
print()

#ИНДЕКСАЦИЯ
matrix = np.array([(1, 2, 31), (4, 5, 6), (7, 8, 9)])
print(matrix)
print(matrix[1, 2]) # 2 элемент 1 строки
print(matrix[2]) # 2 строка
print(matrix[:, 2]) # 2 столбец
print(matrix[1:3, 0:2]) # 0 и 1 столбец, 1 и 2 строки
print(matrix[matrix > 2])
print()

#СПЕЦИАЛЬНЫЕ ОПЕРАЦИИ
print(matrix.T) # транспонирование
print(matrix.flatten()) #в массив
print(np.linalg.inv(matrix)) # обратная матрица
print(np.trace(matrix)) # след матрицы
print(np.linalg.det(matrix)) # определитель матрицы
print(np.linalg.matrix_rank(matrix)) # ранг матрицы
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(eigenvalues)
print(eigenvectors)
print()

#ДОП. ВОЗМОЖНОСТИ
print(np.info(np.eye)) # справочная информация
# text= np.loadtxt('file.txt') # загрузка из текстового файла
# csv_text = np.genfromtxt('file.csv', delimiter=',') # загрузка из csv файла
# np.savetxt('file.txt', arr, delimiter=' ') # запись в текстовый файл
# np.savetxt('file.csv', arr, delimiter=',') # запись в csv файл
















