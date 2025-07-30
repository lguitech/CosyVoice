import asyncio
import websockets
import wave

SAMPLE_RATE = 24000
OUTPUT_FILE = "output.wav"

async def receive_audio():
    uri = "ws://localhost:8765/tts"
    pcm_chunks = []

    async with websockets.connect(uri) as websocket:
        await websocket.send("你好，这是一段测试语音。")
        print("已发送文本，正在接收语音...")

        while True:
            msg = await websocket.recv()

            if isinstance(msg, str):
                if msg.startswith("[END]"):
                    print("语音接收完成")
                    break
                elif msg.startswith("[ERROR]"):
                    print("接收错误:", msg)
                    break
            else:
                pcm_chunks.append(msg)

    # 写入 wav 文件（int16 格式）
    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 正确：int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(pcm_chunks))

    print(f"成功保存语音为 {OUTPUT_FILE}")

if __name__ == '__main__':
    asyncio.run(receive_audio())
