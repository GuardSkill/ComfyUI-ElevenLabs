# 文件名：ElevenLabsNodes.py
# 路径：ComfyUI/custom_nodes/ElevenLabsNodes.py
import io
import time
import random
import torch
import librosa
import requests
from elevenlabs.client import ElevenLabs


class ElevenLabsVoiceQuery:
    """
    查询 ElevenLabs 共享语音库
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "category": (["all", "high_quality", "professional"], {"default": "all"}),
                "gender": (["all", "female", "male", "neutral"], {"default": "all"}),
                "age": (["all", "middle-aged", "middle_aged", "old", "young"], {"default": "all"}),
                "language": (["all", "ar", "bg", "cs", "da", "de", "el", "en", "es", "fi",
                             "fil", "fr", "hi", "hr", "hu", "id", "it", "ja", "ko", "ms",
                             "nl", "no", "pl", "pt", "ro", "ru", "sk", "sv", "ta", "tr",
                             "uk", "vi", "zh"], {"default": "all"}),
                "locale": (["all", "ar-EG", "ar-KW", "ar-LB", "ar-MA", "ar-SA", "bg-BG", "ceb-PH",
                           "cmn-CN", "cmn-TW", "cs-CZ", "da-DK", "de-AT", "de-DE", "el-GR",
                           "en-AU", "en-CA", "en-FI", "en-GB", "en-IE", "en-IN", "en-JM",
                           "en-KR", "en-MY", "en-NG", "en-NZ", "en-PH", "en-RU", "en-SG",
                           "en-US", "en-ZA", "es-AR", "es-CL", "es-CO", "es-ES", "es-MX",
                           "es-PE", "es-US", "es-VE", "fi-FI", "fil-PH", "fr-BE", "fr-CA",
                           "fr-CH", "fr-FR", "fr-TN", "fr-US", "hi-IN", "hr-HR", "hu-HU",
                           "id-ID", "ilo-PH", "it-IT", "ja-JP", "jv-ID", "ko-KR", "li-NL",
                           "ms-MY", "nl-BE", "nl-NL", "no-NO", "pl-PL", "pt-BR", "pt-PT",
                           "ro-RO", "ru-RU", "sk-SK", "sv-SE", "ta-IN", "tr-TR", "uk-UA",
                           "vi-VN"], {"default": "all"}),
                "use_cases": (["all", "advertisement", "characters_animation", "conversational",
                              "entertainment_tv", "informative_educational", "narrative_story",
                              "social_media"], {"default": "all"}),
                "descriptive": (["all", "anxious", "calm", "casual", "confident", "excited",
                                "formal", "gentle", "professional", "relaxed", "serious",
                                "soft", "upbeat", "warm"], {"default": "all"}),
                "page_size": ("INT", {"default": 100, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "query_voices"
    CATEGORY = "audio/ElevenLabs"

    def query_voices(self, api_key, seed, category="all", gender="all", age="all",
                    language="all", locale="all", use_cases="all", descriptive="all", page_size=100):

        if not api_key.strip():
            raise RuntimeError("ElevenLabs API-Key 不能为空")

        # 构建查询参数
        params = {
            "page_size": page_size,
            "sort": "cloned_by_count"
        }

        # 只添加非 "all" 的参数
        if category and category != "all":
            params["category"] = category
        if gender and gender != "all":
            params["gender"] = gender
        if age and age != "all":
            params["age"] = age
        if language and language != "all":
            params["language"] = language
        if locale and locale != "all":
            params["locale"] = locale
        if use_cases and use_cases != "all":
            params["use_case"] = use_cases
        if descriptive and descriptive != "all":
            params["descriptive"] = descriptive

        # 调用 ElevenLabs API（带重试机制）
        url = "https://api.elevenlabs.io/v1/shared-voices"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }

        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                # 获取语音列表
                voices = data.get("voices", [])

                if not voices:
                    raise RuntimeError("未找到符合条件的语音")

                # 使用随机种子选择语音
                random.seed(seed)
                selected_voice = random.choice(voices)

                voice_id = selected_voice.get("voice_id", "")
                voice_name = selected_voice.get("name", "Unknown")

                if not voice_id:
                    raise RuntimeError("获取语音 ID 失败")

                print(f"找到 {len(voices)} 个语音，使用随机种子 {seed} 选择: {voice_name} (ID: {voice_id})")
                return (voice_id,)

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < 2:  # 不是最后一次尝试
                    print(f"查询语音库失败 (尝试 {attempt + 1}/3): {str(e)}，重试中...")
                    time.sleep(1)  # 等待1秒后重试
                continue

        # 所有重试都失败
        raise RuntimeError(f"查询语音库失败（已重试3次）: {str(last_error)}")


class ElevenLabsTTS:
    """
    使用 voice_id 生成语音
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "The first move is what sets everything in motion."}),
                "voice_id": ("STRING", {"default": ""}),
                "model_id": (["eleven_multilingual_v2",
                              "eleven_monolingual_v1",
                              "eleven_turbo_v2",
                              "eleven_turbo_v2_5"], {"default": "eleven_multilingual_v2"}),
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/ElevenLabs"

    def generate(self, text, voice_id, model_id, api_key):

        if not api_key.strip():
            raise RuntimeError("ElevenLabs API-Key 不能为空")

        if not voice_id.strip():
            raise RuntimeError("voice_id 不能为空，请提供 voice_id 或连接语音查询节点")

        # 使用 ElevenLabs 客户端生成语音（带重试机制）
        client = ElevenLabs(api_key=api_key)

        last_error = None
        for attempt in range(3):
            try:
                audio_chunks = []
                for chunk in client.text_to_speech.convert(
                        text=text,
                        voice_id=voice_id,
                        model_id=model_id,
                        output_format="mp3_44100_128"):
                    audio_chunks.append(chunk)

                audio_bytes = b"".join(audio_chunks)

                # 转换为 ComfyUI AUDIO 格式
                wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                tensor = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)

                return ({"waveform": tensor, "sample_rate": sr},)

            except Exception as e:
                last_error = e
                if attempt < 2:  # 不是最后一次尝试
                    print(f"生成语音失败 (尝试 {attempt + 1}/3): {str(e)}，重试中...")
                    time.sleep(1)  # 等待1秒后重试
                continue

        # 所有重试都失败
        raise RuntimeError(f"生成语音失败（已重试3次）: {str(last_error)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ElevenLabsVoiceQuery": ElevenLabsVoiceQuery,
    "ElevenLabsTTS": ElevenLabsTTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenLabsVoiceQuery": "ElevenLabs Voice Query (查询语音库)",
    "ElevenLabsTTS": "ElevenLabs TTS (文本转语音)"
}