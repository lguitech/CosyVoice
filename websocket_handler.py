### websocket_handler.py
# ----------------------
# 这个模块处理 WebSocket 连接，接收文本并推入队列

from fastapi import WebSocket
from request_queue import request_queue

# 处理 WebSocket 连接
async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    speaker_id = "my_zero_shot_spk"  # 指定推理说话人 ID

    try:
        while True:
            # 接收用户上传的文本（支持多段输入）
            text = await websocket.receive_text()

            # 将请求推入到队列，待后端执行
            request_queue.put({
                "text": text,
                "spk": speaker_id,
                "ws": websocket
            })
    except Exception as e:
        await websocket.close()
        print(f"[WebSocket] 连接关闭: {e}")
