### cosyvoice_service.py
# ----------------------
# 这是服务的入口文件，进行 CosyVoice 模型初始化，启动 FastAPI + WebSocket 服务

import sys
import torch
import threading
from fastapi import FastAPI, WebSocket
import uvicorn

# 加载 CosyVoice2 相关模块
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 自定义的模块
from inference_worker import inference_worker
from websocket_handler import handle_websocket
from request_queue import request_queue

# 加载 matcha 模块的路径
sys.path.append('third_party/Matcha-TTS')

# 初始化 CosyVoice2 模型，只初始化一次
print("🚀 CUDA 可用:", torch.cuda.is_available())
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=True,
    load_vllm=False,
    fp16=True
)

# 添加 zero-shot 说话人，便于后续重复使用
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk')
cosyvoice.save_spkinfo()

# 启动接受请求的背景线程
threading.Thread(target=inference_worker, args=(cosyvoice, request_queue), daemon=True).start()

# 创建 FastAPI app
app = FastAPI()

# WebSocket 接口
@app.websocket("/tts")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)

# 程序主入口
if __name__ == '__main__':
    uvicorn.run("cosyvoice_service:app", host="0.0.0.0", port=8765)
