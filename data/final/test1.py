import re
from pathlib import Path
import pandas as pd

# 실행 파일(test1.py)의 위치를 기준으로 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "player_injury_stints.csv"  # CSV가 다른 폴더에 있다면 'data/player_injury_stints.csv' 등으로 수정
OUTPUT_PATH = BASE_DIR / "player_injury_stints_with_return.csv"


def calculate_return_date(row):
    note = str(row["injury_note_sample"])
    first_il_date = pd.to_datetime(row["first_il_date"])

    # 1. 부상 일수 추출 (예: 10-day, 15 day)
    duration_match = re.search(r"(\d+)[\s-]*day", note, re.IGNORECASE)
    if not duration_match:
        return None

    duration_days = int(duration_match.group(1))

    # 2. 소급 적용일(retroactive to) 존재 여부 확인
    retro_match = re.search(
        r"retroactive to ([A-Z][a-z]+ \d{1,2}, \d{4})", note, re.IGNORECASE
    )
    if retro_match:
        try:
            start_date = pd.to_datetime(retro_match.group(1))
        except Exception:
            start_date = first_il_date
    else:
        start_date = first_il_date

    # 3. 복귀 날짜 계산 (시작일 + 부상 기간)
    return (start_date + pd.Timedelta(days=duration_days)).strftime("%Y-%m-%d")


# 데이터 로드 및 계산
df = pd.read_csv(CSV_PATH)
df["return_date"] = df.apply(calculate_return_date, axis=1)

# 결과 저장
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"작업 완료: {OUTPUT_PATH}")