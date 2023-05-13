import os
import sys
import wave
import enum
# import dataclasses

import numpy as np
from simpleaudio import play_buffer

from .rvc import (
    torch, device, is_half, hubert_model, Converter,
    SynthesizerTrnMs256NSFSid, SynthesizerTrnMs256NSFSidNono,
)


# @dataclasses.dataclass
# class Style:
#     name: str
#     id: int


# @dataclasses.dataclass
# class Meta:
#     name: str
#     styles: list[Style]
#     speaker_uuid: str
#     version: str


# @dataclasses.dataclass
# class SupportedDevices:
#     cpu: bool
#     cuda: bool
#     dml: bool


# class AccelerationMode(str, enum.Enum):
#     AUTO = "AUTO"
#     CPU = "CPU"
#     GPU = "GPU"


# @dataclasses.dataclass
# class Mora:
#     text: str
#     consonant: str
#     consonant_length: float
#     vowel: str
#     vowel_length: float
#     pitch: float


# @dataclasses.dataclass
# class AccentPhrase:
#     moras: list[Mora]
#     accent: int
#     pause_mora: Mora
#     is_interrogative: bool


# @dataclasses.dataclass
# class AudioQuery:
#     accent_phrases: list[AccentPhrase]
#     speed_scale: float
#     pitch_scale: float
#     intonation_scale: float
#     volume_scale: float
#     pre_phoneme_length: float
#     post_phoneme_length: float
#     output_sampling_rate: int
#     output_stereo: bool
#     kana: str


sys.path.append(os.path.dirname(__file__)+"/../")


from .vvox import VoicevoxCore


HERE = os.path.dirname(__file__)
CORE = VoicevoxCore(open_jtalk_dict_dir=os.path.join(HERE, "open_jtalk_dict"))
[CORE.load_model(id) for id in [0,2,6]]


class Tone(enum.IntEnum):
    happy = 0
    calm = 2
    angry = 6


class Wave():
    @classmethod
    def from_ndarray(cls, ndarray: np.ndarray, sr: int):
        ndarray = ndarray.squeeze()
        if len(ndarray.shape) == 1: pass
        elif len(ndarray.shape) == 2: ndarray = np.average(ndarray, 1)
        else: Exception("len(ndarray.squeeze().shape) must be in [1, 2].")
        return Wave(ndarray=ndarray, sr=sr)

    @classmethod
    def from_file(cls, wav_file: str):
        f: wave.Wave_read = wave.open(wav_file, "rb")
        dtype = np.int16 if f.getsampwidth() == 2 else np.int32
        channels, framerate = f.getnchannels(), f.getframerate()
        ndarray = np.frombuffer(f.readframes(f.getnframes()), dtype)
        f.close()
        if channels != 1: ndarray = np.average(ndarray.reshape(-1, channels), 1)
        return Wave(ndarray=ndarray, sr=framerate)

    def __init__(self, ndarray: np.ndarray, sr: int):
        self.ndarray = ndarray
        self.sr = sr

    def __call__(self, wait_done: bool=True):
        try:
            play_obj = play_buffer(self.ndarray.astype(np.int16, copy=False), 1, 2, self.sr)
            if wait_done: play_obj.wait_done()
        except Exception as e: print(e.args)

    @property
    def samples(self): return len(self.ndarray)

    def change_sr(self, sr: int):
        n = (self.samples * sr) // self.sr
        x = np.arange(n, dtype=np.float64) * (self.sr/sr)
        return Wave(np.interp(x, np.arange(self.samples), self.ndarray).astype(self.ndarray.dtype), sr)

    def save(self, wav_file: str):
        f: wave.Wave_write = wave.open(wav_file, "wb")
        f.setparams((1, 2, self.sr, self.samples, "NONE", "not compressed"))
        f.writeframes(self.ndarray.astype(np.int16, copy=False).tobytes())
        f.close()


class Model():
    def __init__(self, model_file: str) -> None:
        self._model = torch.load(model_file, map_location="cpu")
        keys = [
            "spec_channels", "segment_size", "inter_channels", "hidden_channels",
            "filter_channels", "n_heads", "n_layers", "kernel_size", "p_dropout",
            "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
            "upsample_rates", "upsample_initial_channel",
            "upsample_kernel_sizes", "spk_embed_dim", "gin_channels", "sr"]
        for i, key in enumerate(keys): self._model["params"][key] = self._model["config"][i]
        self._model["params"]["spk_embed_dim"] = self._model["weight"]["emb_g.weight"].shape[0]
        self.sr = self._model["params"]["sr"]
        if_f0 = self._model.get("f0", 1) == 1
        if if_f0: self._net_g = SynthesizerTrnMs256NSFSid(**self._model["params"], is_half=is_half)
        else: self._net_g = SynthesizerTrnMs256NSFSidNono(**self._model["params"])
        del self._net_g.enc_q
        self._net_g.load_state_dict(self._model["weight"], strict=False)
        self._net_g.eval().to(device)
        self._net_g = self._net_g.half() if is_half else self._net_g.float()
        self._n_spk = self._model["params"]["spk_embed_dim"]
        self._converter = Converter(self.sr, if_f0)

    def convert(self, wave: Wave, *, raise_pitch: int=0, f0_method: str="pm"):
        wave = wave.change_sr(16000)
        times = [0, 0, 0]
        a = self._converter(hubert_model, self._net_g, 0, wave.ndarray, times, raise_pitch, f0_method)
        return Wave.from_ndarray(a, self.sr)

    def __call__(self, text: str, tone: Tone=Tone.happy, *,
                 speed: float=1.0, pitch: float=0.0, intonation: float=1.0, volume: float=1.0):
        aq = CORE.audio_query(text, tone)
        aq.speed_scale = speed
        aq.pitch_scale = pitch
        aq.intonation_scale = intonation
        aq.volume_scale = volume
        sr = aq.output_sampling_rate
        return self.convert(Wave.from_ndarray(np.frombuffer(CORE.synthesis(aq, tone)[44:], np.int16), sr))
