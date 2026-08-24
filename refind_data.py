# ==================================================
#                   데이터 정제
# 원본 Lahman에서 프로젝트에 필요한 2000년대 이후 데이터만 추출
# https://sabr.org/lahman-database/
# ==================================================
import sys
import glob
import os
import pandas as pd

START_YEAR = 2000

# 파일별로 연도 정보를 담고 있는 컬럼명 (Lahman DB 기준으로 미리 조사한 값)
# 이 목록에 없는 파일은 자동 탐지를 시도한다.
YEAR_COLUMN_MAP = {
    "AllstarFull.csv": "yearID",
    "Appearances.csv": "yearID",
    "AwardsManagers.csv": "yearID",
    "AwardsPlayers.csv": "yearID",
    "AwardsShareManagers.csv": "yearID",
    "AwardsSharePlayers.csv": "yearID",
    "Batting.csv": "yearID",
    "BattingPost.csv": "yearID",
    "CollegePlaying.csv": "yearID",
    "Fielding.csv": "yearID",
    "FieldingOF.csv": "yearID",
    "FieldingOFsplit.csv": "yearID",
    "FieldingPost.csv": "yearID",
    "HallOfFame.csv": "yearid",       # 소문자 주의
    "HomeGames.csv": "yearkey",
    "Managers.csv": "yearID",
    "ManagersHalf.csv": "yearID",
    "Pitching.csv": "yearID",
    "PitchingPost.csv": "yearID",
    "Salaries.csv": "yearID",
    "SeriesPost.csv": "yearID",
    "Teams.csv": "yearID",
    "TeamsHalf.csv": "yearID",
}

# 연도(시즌) 개념이 없는, 즉 필터링하지 않고 그대로 두는 참조용 테이블
NO_YEAR_FILES = {
    "People.csv",           # 선수 기본정보 (생년/데뷔일 등) - 시즌 데이터 아님
    "Parks.csv",            # 구장 정보
    "Schools.csv",          # 학교 정보
    "TeamsFranchises.csv",  # 프랜차이즈(구단) 기본정보
}

CANDIDATE_YEAR_COLS = ["yearid", "year_id", "season", "yearkey", "year"]


def find_year_column(df: pd.DataFrame):
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in CANDIDATE_YEAR_COLS:
        if cand in lower_cols:
            return lower_cols[cand]
    return None


def process_file(path: str, out_dir: str):
    fname = os.path.basename(path)
    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)

    if fname in NO_YEAR_FILES:
        df.to_csv(os.path.join(out_dir, fname), index=False)
        print(f"[유지-필터없음] {fname:<25} 행 {n_before:>7,} -> {len(df):>7,} (연도 정보 없는 참조 테이블)")
        return

    col = YEAR_COLUMN_MAP.get(fname) or find_year_column(df)

    if col is None or col not in df.columns:
        # 연도 컬럼을 못 찾은 경우 -> 그대로 복사 (걸러내지 못함을 명시)
        df.to_csv(os.path.join(out_dir, fname), index=False)
        print(f"[유지-탐지실패] {fname:<25} 행 {n_before:>7,} -> {len(df):>7,} (연도 컬럼을 찾지 못해 원본 유지)")
        return

    years = pd.to_numeric(df[col], errors="coerce")
    filtered = df[years >= START_YEAR].copy()
    filtered.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"[필터링]        {fname:<25} 행 {n_before:>7,} -> {len(filtered):>7,} (컬럼: {col})")


def main():
    if len(sys.argv) == 3:
        input_dir, output_dir = sys.argv[1], sys.argv[2]
    else:
        input_dir = "./rawdata/"
        output_dir = "./mlb_2000s"

    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    if not csv_files:
        print(f"'{input_dir}' 에서 csv 파일을 찾지 못했습니다.")
        sys.exit(1)

    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}")
    print(f"기준 연도: {START_YEAR}년 이후 (포함)\n")

    for path in csv_files:
        process_file(path, output_dir)

    print("\n완료되었습니다.")


if __name__ == "__main__":
    main()