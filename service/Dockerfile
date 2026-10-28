# Build from the project root (final-team/):
#   docker build -f service/Dockerfile -t antispam .
# Python 3.12, not 3.11: the model was trained with numpy 2.5 / scipy 1.18, which have no 3.11 wheels.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY service/requirements.txt service/requirements.txt
RUN pip install --root-user-action=ignore -r service/requirements.txt

# Only code and model artifacts get here, see Dockerfile.dockerignore (no data/, no training logs).
COPY --chown=user src/ src/
COPY --chown=user models/ models/
COPY --chown=user service/ service/

USER user
ENV HOME=/home/user \
    MODEL_BACKEND=classic \
    MODEL_PATH=models/final_classic.joblib

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"
CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
