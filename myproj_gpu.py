import warnings
warnings.filterwarnings("ignore")
import sys
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 添加 matcha 模块路径
sys.path.append('third_party/Matcha-TTS')

# 检查 CUDA 可用性
print("🚀 CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🧠 GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ 警告：当前未检测到可用的 CUDA 设备，将使用 CPU！")

# 初始化 CosyVoice2（使用 fp16 表示启用 GPU 推理）
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=True,
    load_vllm=False,
    fp16=True  # 启用 GPU（半精度），若无 GPU 会自动 fallback 到 CPU
)

# 打印当前模型所在设备（内部会自动处理 CUDA）
print("📦 当前运行设备: CUDA" if torch.cuda.is_available() else "📦 当前运行设备: CPU")


# 加载 16kHz 提示语音（zero-shot prompt）
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 添加 zero-shot 说话人
assert cosyvoice.add_zero_shot_spk(
    '希望你以后能够做的比我还好呦。',
    prompt_speech_16k,
    'my_zero_shot_spk'
)
print("现在开始推理")
# 推理：将文本转换为语音并保存为 WAV 文件
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '''7月27日，少林寺管理处发布情况通报：少林寺住持释永信涉嫌刑事犯罪，挪用侵占项目资金寺院资产；''',
    '',
    '',
    zero_shot_spk_id='my_zero_shot_spk',
    stream=True)):

    output_path = f'zero_shot_{i}.wav'
    torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
    print(f"✅ 保存语音到: {output_path}")

# 保存说话人信息（方便后续重复使用）
cosyvoice.save_spkinfo()

print("📁 已保存说话人信息到本地配置")

