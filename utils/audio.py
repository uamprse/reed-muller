import io
import requests
from pydub import AudioSegment

def capture_audio_segment(url: str, target_duration_ms: int, chunk_size: int) -> AudioSegment:
    """
    Считывает поток, накапливая данные, пока длина аудио
    не станет не менее target_duration_ms миллисекунд. Возвращает сегмент длиной target_duration_ms.
    """
    stream_bytes = bytes()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for block in r.iter_content(chunk_size=chunk_size):
            stream_bytes += block
            try:
                audio_seg = AudioSegment.from_mp3(io.BytesIO(stream_bytes))
                if len(audio_seg) >= target_duration_ms:
                    return audio_seg[:target_duration_ms]
            except Exception:
                continue
    return None
