# Clean-machine test of the README setup: builds the environment exactly as the
# Setup section describes, then runs docker-check.sh (no model calls).
# Build from the repo root, so the image also holds results/ for scoring:
#   docker build -t metavul .
#   docker run --rm --security-opt seccomp=unconfined metavul                                 # offline checks
#   docker run --rm -it --security-opt seccomp=unconfined -e OPENROUTER_API_KEY metavul bash  # shell for real runs
# seccomp=unconfined lets Codex's bwrap sandbox create namespaces (Docker's default
# profile blocks it); see the README's Docker section.
FROM node:24-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git procps \
 && rm -rf /var/lib/apt/lists/*
# uv, uv tools (litellm) and Claude Code all install into ~/.local/bin.
ENV PATH="/root/.local/bin:${PATH}"

# Setup step 1: uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup step 2: Python environment
WORKDIR /work/code
COPY code/pyproject.toml code/.python-version ./
RUN uv sync

# Setup step 3: LiteLLM proxy
RUN uv tool install 'litellm[proxy]==1.100.0'

# Setup step 4: agent CLIs
RUN npm install -g @openai/codex@0.153.4
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.283

# Finished runs, next to code/ as in the repo, so the scoring scripts find them.
# .dockerignore leaves out the raw logs and workspaces.
COPY results/ /work/results/

# Code last, so editing it does not reinstall the tools above.
COPY code/ ./

CMD ["bash", "docker-check.sh"]
