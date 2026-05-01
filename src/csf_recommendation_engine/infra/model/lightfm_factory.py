from lightfm import LightFM


def build_lightfm_model() -> LightFM:
    return LightFM(
        no_components=64,
        loss="warp",
        learning_rate=0.05,
        item_alpha=1e-6,
        user_alpha=1e-6,
    )
