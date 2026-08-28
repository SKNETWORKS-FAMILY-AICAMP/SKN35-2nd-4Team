"""선수 실사진 연결 — MLB 공식 CDN(img.mlbstatic.com) 헤드샷을 직접 링크한다.

파일을 내려받아 저장하지 않는다 — <img src>가 MLB 자체 CDN을 직접 가리키므로
브라우저가 그때그때 받아온다(핫링크). Lahman player_id(=bbrefID 대부분) ->
Chadwick Register(key_bbref -> key_mlbam) 크로스워크로 MLB 공식 person id를
찾고, 그 id로 공식 헤드샷 URL을 만든다. 매칭 안 되는 선수는 None을 반환하고
호출부가 실루엣 아바타로 대체한다 — 가짜 사진을 만들지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PEOPLE_PATH = ROOT / "data" / "processed" / "People.csv"
CACHE_DIR = ROOT / ".cache"
CHADWICK_SHARDS = "0123456789abcdef"

HEADSHOT_URL = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/"
    "w_180,q_auto:best/v1/people/{mlbam_id}/headshot/67/current"
)


@st.cache_data(show_spinner=False)
def load_mlbam_lookup() -> dict[str, int]:
    """player_id(Lahman) -> mlbam_id(MLB 공식 person id). 캐시 파일이 없으면 빈 dict."""
    if not PEOPLE_PATH.exists():
        return {}

    frames = []
    for shard in CHADWICK_SHARDS:
        path = CACHE_DIR / f"chadwick_people-{shard}.csv"
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, usecols=["key_mlbam", "key_bbref"], low_memory=False))
    if not frames:
        return {}

    crosswalk = pd.concat(frames, ignore_index=True).dropna()
    crosswalk["key_mlbam"] = crosswalk["key_mlbam"].astype(int)

    people = pd.read_csv(PEOPLE_PATH, usecols=["playerID", "bbrefID"]).dropna()
    merged = people.merge(crosswalk, left_on="bbrefID", right_on="key_bbref", how="inner")
    return dict(zip(merged["playerID"], merged["key_mlbam"]))


def headshot_url(player_id: str, lookup: dict[str, int]) -> str | None:
    mlbam_id = lookup.get(player_id)
    if mlbam_id is None:
        return None
    return HEADSHOT_URL.format(mlbam_id=mlbam_id)
