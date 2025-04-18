import numpy as np
import itertools
import math
import random
from typing import List

def _binom(n, k):
    """Биномиальный коэффициент."""
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def _construct_vector(m, i):
    """Создает вектор для x_i длиной 2^m."""
    half_size = 2 ** (m - i - 1)
    return np.tile([1] * half_size + [0] * half_size, 2 ** i)

def _vector_mult(*vecs):
    """Попарное умножение элементов для любого количества векторов длины n."""
    return np.prod(np.array(vecs), axis=0)

def _vector_add(*vecs):
    """Попарное сложение элементов для любого количества векторов длины n."""
    return np.sum(np.array(vecs), axis=0) % 2

def _vector_neg(x):
    """Взятие отрицания вектора над Z_2, т.е. замена 1 на 0 и наоборот."""
    return 1 - x

def _vector_reduce(x, modulo):
    """Сокращение каждого элемента вектора по модулю."""
    return x % modulo

def _dot_product(x, y):
    """Вычисление скалярного произведения двух векторов."""
    return np.dot(x, y) % 2

def _generate_all_rows(m, S):
    """Генерация всех строк для мономов в S."""
    if not S:
        return [np.ones(2 ** m, dtype=int)]

    i, Srest = S[0], S[1:]

    Srest_rows = _generate_all_rows(m, Srest)

    xi_row = _construct_vector(m, i)
    not_xi_row = _vector_neg(xi_row)
    return [xi_row * row for row in Srest_rows] + [not_xi_row * row for row in Srest_rows]


def _generate_rm_generator(r, m):
    """
    Генерирует генераторную матрицу для кода Рида–Маллера RM(r, m).
    Каждая строка соответствует значению монома (степень монома не более r)
    на всех 2^m точках пространства.
    """
    x_rows = [np.array(_construct_vector(m, i), dtype=int) for i in range(m)]

    matrix_by_row = [
        np.prod([x_rows[i] for i in S], axis=0) if len(S) > 0 else np.ones(2 ** m, dtype=int)
        for s in range(r + 1)
        for S in itertools.combinations(range(m), s)
    ]
    M = np.array(matrix_by_row)
    return M

def construct_check_matrix(r, m):
    """
    Для кода Рида–Маллера RM(r, m) строит проверочную матрицу H.
    По свойству двойственности:
      H = генераторная матрица кода RM(m - r - 1, m).
    """
    r_dual = m - r - 1
    if r_dual < 0:
        return []
    H = _generate_rm_generator(r_dual, m)
    return H

class ReedMuller:
    """Класс, представляющий код Рида-Маллера RM(r,m)."""

    def __init__(self, r, m):
        """Создание кодера/декодера Рида-Маллера для RM(r,m)."""
        self.r, self.m = (r, m)
        self._construct_matrix()
        self.k = len(self.M[0])
        self.n = 2 ** m

    def strength(self):
        """Количество ошибок, которые можно исправить."""
        return 2 ** (self.m - self.r - 1) - 1

    def message_length(self):
        """Длина сообщения, которое нужно закодировать."""
        return self.k

    def block_length(self):
        """Длина закодированного сообщения."""
        return self.n

    def _construct_matrix(self):
        x_rows = [np.array(_construct_vector(self.m, i), dtype=int) for i in range(self.m)]


        self.matrix_by_row = [
            np.prod([x_rows[i] for i in S], axis=0) if len(S) > 0 else np.ones(2 ** self.m, dtype=int)
            for s in range(self.r + 1)
            for S in itertools.combinations(range(self.m), s)
        ]

        self.voting_rows = [
            _generate_all_rows(self.m, [i for i in range(self.m) if i not in S])
            for s in range(self.r + 1)
            for S in itertools.combinations(range(self.m), s)
        ]

        # print(self.voting_rows)

        self.row_indices_by_degree = [0]
        for degree in range(1, self.r + 1):
            self.row_indices_by_degree.append(self.row_indices_by_degree[degree - 1] + _binom(self.m, degree))

        self.M = np.array(self.matrix_by_row).T

    def encode(self, word):
        if word is None:
            return []
        encoded = [_dot_product(word, col) % 2 for col in self.M]
        # print(f"Кодирование: входное сообщение {word}, закодированное слово {encoded}")
        return encoded

    def decode(self, eword: List[int]) -> np.ndarray:
        word = np.full(self.k, -1, dtype=int)

        for degree in range(self.r, -1, -1):
            upper_r = self.row_indices_by_degree[degree]
            lower_r = self.row_indices_by_degree[degree - 1] + 1 if degree > 0 else 0

            for pos in range(lower_r, upper_r + 1):
                votes = [_dot_product(eword, vrow) for vrow in self.voting_rows[pos]]
                num_zeros = votes.count(0)
                num_ones = votes.count(1)

                if num_zeros == num_ones:
                    continue

                word[pos] = 0 if num_zeros > num_ones else 1

            s = np.array([_dot_product(word[lower_r:upper_r + 1], column[lower_r:upper_r + 1]) for column in self.M])
            eword = _vector_add(eword, s)

        if np.any(word == -1):
            return None

        # print("Декодированное слово:", word)
        return word

    def verify_codeword(self, c: List[int]) -> bool:
        """
        Проверяет, удовлетворяет ли кодовое слово c проверочным уравнениям,
        т.е. H * c^T = 0 (по модулю 2).
        """
        if len(c) == 0:
            return False
        H = construct_check_matrix(self.r, self.m)
        # print("\nПроверочная матрица:\n", H)
        if len(H) == 0 :
            return True
        else:
            c = np.array(c, dtype=int)
            syndrome = np.mod(np.dot(H, c.T), 2)
            return np.all(syndrome == 0)

    def __repr__(self):
        return f'<Код Рида-Маллера RM({self.r},{self.m}), сила={self.strength()}>'

def _generate_all_vectors(n):
    v = np.zeros(n, dtype=int)
    while True:
        yield v

        v[-1] += 1
        pos = n - 1
        while pos >= 0 and v[pos] == 2:
            v[pos] = 0
            pos -= 1
            if pos >= 0:
                v[pos] += 1

        if np.array_equal(v, np.zeros(n, dtype=int)):
            break

def _characteristic_vector(n, S):
    return np.array([0 if i not in S else 1 for i in range(n)], dtype=int)

def add_noise(codeword: List[int], error_probability) -> List[int]:
    """Добавляет шум к кодовому слову: вероятность ошибки для каждого бита."""
    noisy = []
    for bit in codeword:
        if random.random() < error_probability:
            noisy.append(1 - bit)
        else:
            noisy.append(bit)
    return noisy
