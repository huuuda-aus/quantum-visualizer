WASM_TARGET := wasm32-unknown-unknown

.PHONY: gen

gen:
	cargo run -p generator --release

.PHONY: build-wasm

build-wasm:
	RUSTFLAGS="-C target-feature=+simd128" cargo +nightly build -p dashboard_wasm --target $(WASM_TARGET) --release
