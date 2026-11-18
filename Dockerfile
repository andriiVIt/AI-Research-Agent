FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
COPY tests ./tests
RUN python -m pip install --no-cache-dir --no-deps -e .
ENV PYDANTIC_AI_NO_BANNER=1
# Pass keys at run time, never bake them into the image:
#   docker run --rm --env-file .env research-scout "Find a paper about ..."
ENTRYPOINT ["research-scout"]
CMD ["--demo", "--log", "/tmp/demo_scout.jsonl"]
