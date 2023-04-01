# whisperx-svr

Reference [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) using [whisperx](https://github.com/m-bain/whisperX) with fastapi generate api.

## Build docker

```
$ git clone https://github.com/HATTER-LONG/whisperx-svr.git
$ cd whisperx-svr
$ sudo docker build -t layton/cuda11.7_fast-whisperx:v1 .
$ sudo docker run -d --gpus all  --name whisperx-app -e ASR_MODEL=base -p 8888:8888 layton/cuda11.7_fast-whisperx:v1

```

- Access `127.0.0.1:8888` to used by webview.
- Use `-e ASR_MODEL=base` to change model.
  - Available ASR_MODELs are tiny, base, small, medium, large, large-v1 and large-v2.
  - Default is large model.
