import numpy as np
import itertools
import math
import random
from typing import List

def _binom(n, k):
    """Биномиальный коэффициент (n-k)!/k!."""
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

class ReedMuller:
    """Класс, представляющий код Рида-Маллера RM(r,m)."""

    def __init__(self, r, m):
        """Создание кодера/декодера Рида-Маллера для RM(r,m)."""
        self.r, self.m = (r, m)
        self._construct_matrix()
        self.k = len(self.M[0])
        self.n = 2 ** m

    def strength(self):
        """Возвращает силу кода, т.е. количество ошибок, которые можно исправить."""
        return 2 ** (self.m - self.r - 1) - 1

    def message_length(self):
        """Длина сообщения, которое нужно закодировать."""
        return self.k

    def block_length(self):
        """Длина кодированного сообщения."""
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

        self.row_indices_by_degree = [0]
        for degree in range(1, self.r + 1):
            self.row_indices_by_degree.append(self.row_indices_by_degree[degree - 1] + _binom(self.m, degree))

        self.M = np.array(self.matrix_by_row).T

    def encode(self, word):
        assert len(word) == self.k
        return [_dot_product(word, col) % 2 for col in self.M]

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

        return word


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

def add_noise(codeword, error_probability=0.1):
    return [bit ^ random.choice([0, 1]) if random.random() < error_probability else bit for bit in codeword]

def simulate_coding_with_noise():
    r, m = 2, 4
    rm = ReedMuller(r, m)

    message_length = rm.message_length()
    all_messages = _generate_all_vectors(message_length)

    success = True
    for word in all_messages:
        print("\nИсходное сообщение:")
        print(word)

        codeword = rm.encode(word)
        print("\nКодированное слово:")
        print(codeword)

        noisy_codeword = add_noise(codeword, error_probability=0.1)
        print("\nКодовое слово с шумами:")
        print(noisy_codeword)

        decoded_word = rm.decode(noisy_codeword)
        print("\nДекодированное слово:")
        if decoded_word is None:
            print("Не удалось корректно декодировать слово.")
        else:
            print(decoded_word)
        if not np.array_equal(decoded_word, word):
            print(f'ERROR: encode({word}) => {codeword}, noisy({codeword} + noise => {noisy_codeword}) => {decoded_word}')
            success = False

    if success:
        print(f'RM({r},{m}): успех.')

if __name__ == '__main__':
    simulate_coding_with_noise()
