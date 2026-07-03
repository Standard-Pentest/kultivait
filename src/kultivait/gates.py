"""Phase-gate context hygiene: distill a transcript into a handoff brief.

The full transcript is always composted (archived to disk), never destroyed —
distillation is lossy, and the compost pile is the escape hatch when the
brief turns out to have dropped something load-bearing.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DISTILL_PROMPT = """\
You are a phase-gate distiller. A work session is moving from its "{from_phase}" \
phase to its "{to_phase}" phase. Distill the transcript below into a handoff brief \
containing ONLY what the {to_phase} phase needs. Use exactly these sections:

FINDINGS: (facts established, with file paths where relevant)
DECISIONS: (choices made and why, one line each)
CONSTRAINTS: (anything that must not be violated)
OPEN QUESTIONS: (unresolved items the next phase must address)

Be ruthless about omitting dead ends and process narration. Never omit a \
constraint, a file path, or a decision.

TRANSCRIPT:
{transcript}
"""


@dataclass(frozen=True)
class HandoffBrief:
    brief: str
    tokens_before: int
    tokens_after: int
    compost_id: str


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Gate:
    def __init__(
        self,
        generate: Callable[[str], str],
        compost_dir: Path,
        template: str = DISTILL_PROMPT,
    ):
        self._generate = generate
        self._compost_dir = Path(compost_dir)
        self._template = template

    def distill(self, transcript: str, *, from_phase: str, to_phase: str) -> HandoffBrief:
        compost_id = f"{from_phase}-{to_phase}-{uuid.uuid4().hex[:8]}"
        self._compost_dir.mkdir(parents=True, exist_ok=True)
        (self._compost_dir / f"{compost_id}.txt").write_text(transcript)

        prompt = self._template.format(
            from_phase=from_phase, to_phase=to_phase, transcript=transcript
        )
        brief = self._generate(prompt)
        return HandoffBrief(
            brief=brief,
            tokens_before=_rough_tokens(transcript),
            tokens_after=_rough_tokens(brief),
            compost_id=compost_id,
        )
