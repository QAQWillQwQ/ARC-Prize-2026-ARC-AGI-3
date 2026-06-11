from __future__ import annotations

import json
import time
from typing import Any, Optional

from arcengine import FrameData, FrameDataRaw, GameAction, GameState


class Agent:
    """Small local subset of the official ARC Agent interface.

    The scored notebook agent only needs this lightweight surface for offline
    collection: game identity, environment access, frame conversion, and frame
    history. Keeping the shim narrow avoids importing optional template agents
    and their unrelated LLM dependencies during OpenLab data collection.
    """

    MAX_ACTIONS: int = 80

    def __init__(
        self,
        card_id: str = "",
        game_id: str = "",
        agent_name: str = "forge_v2_collect",
        ROOT_URL: str = "",
        record: bool = False,
        arc_env: Any = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.ROOT_URL = ROOT_URL
        self.card_id = card_id
        self.game_id = game_id
        self.guid = ""
        self.agent_name = agent_name
        self.tags = tags or []
        self.frames: list[FrameData] = [FrameData(levels_completed=0)]
        self.headers: dict[str, str] = {}
        self.arc_env = arc_env
        self.action_counter = 0
        self.timer = time.time()
        self._cleanup = True
        self._record = bool(record)

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def levels_completed(self) -> int:
        return int(self.frames[-1].levels_completed)

    @property
    def seconds(self) -> float:
        return round(time.time() - self.timer, 2)

    @property
    def fps(self) -> float:
        if self.action_counter <= 0:
            return 0.0
        return round(self.action_counter / max(self.seconds, 0.1), 2)

    @property
    def is_playback(self) -> bool:
        return False

    def _convert_raw_frame_data(self, raw: FrameDataRaw | None) -> FrameData:
        if raw is None:
            raise ValueError("Received None frame data from environment")
        return FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(json.loads(frame.model_dump_json()))

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        data = action.action_data.model_dump()
        raw = self.arc_env.step(action, data=data, reasoning={})
        if raw is None:
            return None
        return self._convert_raw_frame_data(raw)

    def cleanup(self, *_args: Any, **_kwargs: Any) -> None:
        self._cleanup = False
