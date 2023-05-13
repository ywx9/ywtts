from enum import Enum
import dataclasses


@dataclasses.dataclass
class Style:
    name: str
    id: int


@dataclasses.dataclass
class Meta:
    name: str
    styles: list[Style]
    speaker_uuid: str
    version: str


@dataclasses.dataclass
class SupportedDevices:
    cpu: bool
    cuda: bool
    dml: bool


class AccelerationMode(str, Enum):
    AUTO = "AUTO"
    CPU = "CPU"
    GPU = "GPU"


@dataclasses.dataclass
class Mora:
    text: str
    consonant: str
    consonant_length: float
    vowel: str
    vowel_length: float
    pitch: float


@dataclasses.dataclass
class AccentPhrase:
    moras: list[Mora]
    accent: int
    pause_mora: Mora
    is_interrogative: bool


@dataclasses.dataclass
class AudioQuery:
    accent_phrases: list[AccentPhrase]
    speed_scale: float
    pitch_scale: float
    intonation_scale: float
    volume_scale: float
    pre_phoneme_length: float
    post_phoneme_length: float
    output_sampling_rate: int
    output_stereo: bool
    kana: str
