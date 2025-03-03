import os
import numpy as np
from reedmuller import ReedMuller, add_noise
from pydub import AudioSegment
from utils.config import STREAM_URL, CHUNK_SIZE, TARGET_DURATION_MS, FRAME_RATE, CHANNELS, SAMPLE_WIDTH, NOISE_PROBABILITY
from utils.audio import capture_audio_segment
from utils.bit_utils import int16_to_bits, bits_to_int16_list, split_bits_into_blocks, combine_blocks_into_bitstream

def process_blocks(blocks, rm: ReedMuller, noise_probability: float):
    noisy_blocks = []
    recovered_blocks = []
    error_count = 0

    for block in blocks:
        # наложение шума на обычное аудио для noisy.wav
        noisy_block = add_noise(block, error_probability=(noise_probability * (rm.n/ rm.k)))
        noisy_blocks.append(noisy_block)

        # наложение шума на кодированное аудио для ecovered.wav
        encoded = rm.encode(block)
        noisy_encoded = add_noise(encoded, error_probability=noise_probability)
        decoded = rm.decode(noisy_encoded)
        if decoded is None:
            error_count += 1
            decoded = noisy_block  # если декодирование не удалось, используем исходный испорченный блок
        recovered_blocks.append(decoded)

    return noisy_blocks, recovered_blocks, error_count

def main():
    video_dir = "video"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    print("Захват аудио...")
    original_audio = capture_audio_segment(STREAM_URL, TARGET_DURATION_MS, CHUNK_SIZE)
    if original_audio is None:
        print("Не удалось захватить аудио.")
        return

    original_audio = original_audio.set_channels(CHANNELS).set_frame_rate(FRAME_RATE).set_sample_width(SAMPLE_WIDTH)
    original_path = os.path.join(video_dir, "original.wav")
    original_audio.export(original_path, format="wav")
    print(f"Оригинальное аудио сохранено в '{original_path}'.")

    samples = np.array(original_audio.get_array_of_samples(), dtype=np.int16)
    print(f"Количество сэмплов: {len(samples)}")

    bitstream = []
    for sample in samples:
        bitstream.extend(int16_to_bits(sample))
    print(f"Общее количество бит: {len(bitstream)}")

    RM_r = 1
    RM_m = 4
    rm = ReedMuller(r=RM_r, m=RM_m)
    k = rm.message_length()
    print(f"Используем код: {rm}")

    blocks = split_bits_into_blocks(bitstream, k)
    print(f"Количество блоков по {k} бит: {len(blocks)}")

    noisy_blocks, recovered_blocks, error_count = process_blocks(blocks, rm, NOISE_PROBABILITY)
    print(f"Не удалось декодировать {error_count} блоков из {len(blocks)}.")

    recovered_bitstream = combine_blocks_into_bitstream(recovered_blocks)
    noisy_bitstream = combine_blocks_into_bitstream(noisy_blocks)

    recovered_samples = bits_to_int16_list(recovered_bitstream)
    noisy_samples = bits_to_int16_list(noisy_bitstream)

    recovered_audio = AudioSegment(
        data=np.array(recovered_samples, dtype=np.int16).tobytes(),
        sample_width=SAMPLE_WIDTH,
        frame_rate=FRAME_RATE,
        channels=CHANNELS
    )
    noisy_audio = AudioSegment(
        data=np.array(noisy_samples, dtype=np.int16).tobytes(),
        sample_width=SAMPLE_WIDTH,
        frame_rate=FRAME_RATE,
        channels=CHANNELS
    )

    noisy_path = os.path.join(video_dir, "noisy.wav")
    recovered_path = os.path.join(video_dir, "recovered.wav")
    noisy_audio.export(noisy_path, format="wav")
    recovered_audio.export(recovered_path, format="wav")
    print(f"Промежуточное аудио (с шумами) сохранено в '{noisy_path}'.")
    print(f"Восстановленное аудио сохранено в '{recovered_path}'.")

if __name__ == "__main__":
    main()
