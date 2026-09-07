# DINO-WM causal predictor

`model.py` retains the official causal ViT core at the revision in `SOURCE.json`. It uses constructor-injected frame/patch dimensions and a device-following attention mask.

The platform adapter separately adds normalized state/action conditioning, a zero-initialized residual output head, and a small RGB patch decoder. These adaptations and actual source bytes are recorded with each model; demo outputs do not claim original benchmark performance.
