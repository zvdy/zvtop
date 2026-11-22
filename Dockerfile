# syntax=docker/dockerfile:1

# zvtop - terminal GPU monitor for NVIDIA GPUs.
#
# Real GPUs: the host needs the NVIDIA Container Toolkit; run with
#   docker run --rm -it --gpus all ghcr.io/zvdy/zvtop
# (the toolkit injects libnvidia-ml.so, which go-nvml dlopen()s at runtime).
# No hardware needed to try the UI:
#   docker run --rm -it ghcr.io/zvdy/zvtop -mock 2

FROM golang:1.27.2-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
ENV CGO_ENABLED=1 CGO_CFLAGS=-Wno-deprecated-declarations
RUN go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/zvtop ./cmd/zvtop

FROM debian:bookworm-slim
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="zvtop" \
      org.opencontainers.image.description="Terminal UI monitor for NVIDIA GPUs (btop-style), built on NVML" \
      org.opencontainers.image.source="https://github.com/zvdy/zvtop" \
      org.opencontainers.image.url="https://github.com/zvdy/zvtop" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility \
    TERM=xterm-256color
COPY --from=build /out/zvtop /usr/local/bin/zvtop
ENTRYPOINT ["zvtop"]
