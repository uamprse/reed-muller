from typing import List
import numpy as np

def int16_to_bits(sample: int) -> List[int]:
    value = int(sample) & 0xFFFF
    bits_str = format(value, '016b')
    return [int(b) for b in bits_str]

def bits_to_int16(bits: List[int]) -> np.int16:
    bits_str = ''.join(str(b) for b in bits)
    value = int(bits_str, 2)
    if value >= 2 ** 15:
        value -= 2 ** 16
    return np.int16(value)

def split_bits_into_blocks(bitstream: List[int], block_size: int) -> List[List[int]]:
    """
    Разбивает битовый поток на блоки длины block_size.
    Если последний блок неполный, дополняет его нулями.
    """
    pad_length = (block_size - (len(bitstream) % block_size)) % block_size
    bitstream_extended = bitstream + [0] * pad_length
    return [bitstream_extended[i:i+block_size] for i in range(0, len(bitstream_extended), block_size)]

def combine_blocks_into_bitstream(blocks: List[List[int]]) -> List[int]:
    """Объединяет список блоков битов в один битовый поток."""
    return [bit for block in blocks for bit in block]

def bits_to_int16_list(bitstream: List[int]) -> List[np.int16]:
    """
    Разбивает битовый поток на группы по 16 бит, дополняя последний блок нулями при необходимости,
    и преобразует каждую группу в число int16.
    """
    pad_length = (16 - (len(bitstream) % 16)) % 16
    bitstream_extended = bitstream + [0] * pad_length
    samples = []
    for i in range(0, len(bitstream_extended), 16):
        sample_bits = bitstream_extended[i:i+16]
        samples.append(bits_to_int16(sample_bits))
    return samples
