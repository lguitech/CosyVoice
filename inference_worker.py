import asyncio
import io
import torch
import random
import numpy as np

import wave
from request_queue import request_queue

def save_pcm_chunks_as_wav(filename, pcm_chunks, sample_rate):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(pcm_chunks))

    print(f"save_pcm_chunks_as_wav sample_rate: {sample_rate}")

def tensor_to_pcm_bytes(tensor):
    # float32 [-1.0, 1.0] → int16 [-32768, 32767]
    audio = tensor.squeeze(0).cpu().numpy()
    audio = (audio * 32767.0).clip(-32768, 32767).astype("int16")
    return audio.tobytes()

def inference_worker(cosyvoice, request_queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        task = request_queue.get()
        text = task["text"]
        speaker_id = task["spk"]
        websocket = task["ws"]

        async def run_tts():
            try:
                # 流式推理，逐块发送裸 PCM 数据
                pcm_chunks = []
                async for chunk in stream_tts(cosyvoice, text, speaker_id):
                    pcm_chunks.append(chunk)
                    await websocket.send_bytes(chunk)

                # 测试：推理完成后保存本地 WAV 文件，检查音质
                save_pcm_chunks_as_wav("test_local_output.wav", pcm_chunks, cosyvoice.sample_rate)

                await websocket.send_text("[END]")
            except Exception as e:
                await websocket.send_text(f"[ERROR] {str(e)}")

        loop.run_until_complete(run_tts())

async def stream_tts(cosyvoice, text, speaker_id):

    # 设置种子，固定输出
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    prompt_text = "这是一位温柔、耐心、亲切的女客服的声音，说话语速适中，语调柔和，令人安心。"

    for i, j in enumerate(cosyvoice.inference_zero_shot(
        text, prompt_text, '',
        zero_shot_spk_id=speaker_id,
        stream=True)):

        pcm_bytes = tensor_to_pcm_bytes(j["tts_speech"])
        yield pcm_bytes
