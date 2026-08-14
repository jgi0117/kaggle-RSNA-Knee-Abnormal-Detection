from pathlib import Path

from .constants import LABEL_COLS, TARGET_TO_GUIDANCE


class GuidanceStore:
    def __init__(self, guidance_dir: str):
        self.guidance_dir = Path(guidance_dir)
        self._guidance = {}

        missing = []

        for target in LABEL_COLS:
            path = self.guidance_dir / TARGET_TO_GUIDANCE[target]

            if not path.exists():
                missing.append(str(path))
                continue

            self._guidance[target] = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

        if missing:
            raise FileNotFoundError(
                "Missing guidance files:\n" + "\n".join(missing)
            )

    def get(self, target: str) -> str:
        if target not in self._guidance:
            raise KeyError(f"Unknown target: {target}")

        return self._guidance[target]
