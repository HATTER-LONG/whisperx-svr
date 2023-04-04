import torch
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, RedirectResponse
import whisperx
import os
import whisper
from whisper import tokenizer
from whisperx.utils import write_txt, write_ass, write_vtt, write_tsv, write_srt
import ffmpeg
from typing import BinaryIO, Union
import numpy as np
from io import StringIO
from threading import Lock

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = os.getenv("ASR_MODEL", "medium.en")
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("using CUDA")
    model = whisper.load_model(model_name, device="cuda")
else:
    model = whisper.load_model(model_name, compute_type="int8")
model_lock = Lock()


SAMPLE_RATE = 16000
LANGUAGE_CODES = sorted(list(tokenizer.LANGUAGES.keys()))

app = FastAPI()


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
def transcribe(
    audio_file: UploadFile = File(...),
    task: Union[str, None] = Query(
        default="transcribe", enum=["transcribe", "translate"]
    ),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    output: Union[str, None] = Query(
        default="txt", enum=["txt", "vtt", "srt", "tsv", "ass"]
    ),
):
    result = run_asr(audio_file.file, task, language, initial_prompt)

    filename = str(audio_file.filename).split(".")[0]
    myFile = StringIO()
    if output == "srt":
        write_srt(result, file=myFile)
    elif output == "vtt":
        write_vtt(result, file=myFile)
    elif output == "tsv":
        write_tsv(result, file=myFile)
    elif output == "txt":
        write_txt(result, file=myFile)
    elif output == "ass":
        write_ass(result, file=myFile)
    else:
        return "Please select an output method!"
    myFile.seek(0)
    return StreamingResponse(
        myFile,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}.{output}"'},
    )


# @app.post("/detect-language", tags=["Endpoints"])
# def language_detection(
#     audio_file: UploadFile = File(...),
# ):
#     # load audio and pad/trim it to fit 30 seconds
#     audio = load_audio(audio_file.file)
#     audio = whisper.pad_or_trim(audio)

#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

#     # detect the spoken language
#     with model_lock:
#         _, probs = model.detect_language(mel)
#     detected_lang_code = max(probs, key=probs.get)

#     result = {
#         "detected_language": tokenizer.LANGUAGES[detected_lang_code],
#         "language_code": detected_lang_code,
#     }

#     return result


def run_asr(
    file: BinaryIO,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
):
    audio = load_audio(file)
    if task is None:
        task = "transcribe"
    with model_lock:
        result = model.transcribe(
            audio, task=task, language=language, initial_prompt=initial_prompt
        )
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE
    )["segments"]
    return result_aligned


def load_audio(file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file
    object Parameters
    ----------
    file: BinaryIO
        The audio file like object
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(
                cmd="ffmpeg",
                capture_stdout=True,
                capture_stderr=True,
                input=file.read(),
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
