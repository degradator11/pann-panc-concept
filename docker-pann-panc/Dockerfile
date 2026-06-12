# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder

WORKDIR /build/pann-panc
COPY pann-panc/Cargo.toml pann-panc/Cargo.lock ./
COPY pann-panc/src ./src
COPY pann-panc/tests ./tests
COPY pann-panc/data ./data

RUN cargo build --release --bin research-bench

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/pann-panc/target/release/research-bench /usr/local/bin/research-bench
COPY pann-panc/data ./data
COPY pann-panc/configs ./configs

RUN mkdir -p /data /models /reports

VOLUME ["/data", "/models", "/reports"]
ENTRYPOINT ["research-bench"]
CMD ["pann-iris", "--format", "json"]
