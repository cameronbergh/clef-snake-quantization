"""Pinned artifacts used in the published experiment (no weights included)."""
OFFICIAL_REPO = "Cloudflare/clef-flash"
OFFICIAL_REV = "17f0b0ad64efb65d273590632833508766b2aae6"
QUANT_REPO = "bartowski/Cloudflare_clef-flash-GGUF"
QUANT_REV = "d7f376ea88c05e7bb1014dd5351a93df9dd8029e"
NATIVE_REV = "c25030496079fdad724609d94f68b859f25774ce"
HEAD_SHA = "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0"
HEAD_SOURCE_SHA = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
MODELS = {
    "bf16": {"port": 8765},
    "q6-k-l": {"port": 8767, "quantization": "Q6_K_L", "file": "Cloudflare_clef-flash-Q6_K_L.gguf", "size": 8106583360, "sha256": "0219eec52c7a82b1b5e8a679408ff5f28cae8d0327a0d84c028ef535e38cc049"},
    "q4-k-m": {"port": 8768, "quantization": "Q4_K_M", "file": "Cloudflare_clef-flash-Q4_K_M.gguf", "size": 5841052992, "sha256": "45f803cbcb6144784653bc31cde957e0d184d963a5198d423dc589a79e178d45"},
    "iq2-m": {"port": 8769, "quantization": "IQ2_M", "file": "Cloudflare_clef-flash-IQ2_M.gguf", "size": 3536905536, "sha256": "c3ce8a0f1551ce9d267be47622e9d88fa9f08d458b6d4cec5796788d85dd41b6"},
    "q2-k": {"port": 8770, "quantization": "Q2_K", "file": "Cloudflare_clef-flash-Q2_K.gguf", "size": 3644384576, "sha256": "df3bf7f4f68651d900f6a55d1cfe68e153a39240a33ae165cfa9dfa048bb52e4"},
}
