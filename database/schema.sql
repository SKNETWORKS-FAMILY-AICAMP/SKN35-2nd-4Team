-- ============================================================
-- SKN35 2nd 4Team - baseball schema (PostgreSQL / Supabase)
-- 원본 Lahman CSV 컬럼명 + B(strength.py)가 실제로 기대하는 이름 기준으로 재구성
--
-- [피드백 반영 원칙]
-- 1) 원천 테이블은 year, 가공 테이블은 contract.py 기준 season을 유지한다.
-- 2) y_departed, y_returned, y_home_win은 pandas NaN/float64 적재 호환을 위해 float nullable을 유지한다.
--    DB를 smallint로 바꾸려면 적재 전에 pandas nullable Int64/boolean dtype과 SQLAlchemy dtype을 명시해야 한다.
-- 3) 라벨 계층은 L1=y_departed, L2=y_path, 조건부 L2b=y_fa_release, L3=y_returned를 유지한다.
-- 4) 은퇴·부상 원인은 Lahman CSV의 확정 라벨이 아니므로 player_season에 저장하지 않는다.
--    관련 설명은 reason.py 또는 화면에서 반드시 '추정'으로 계산·표시한다.
-- 5) Day2 CSV/Supabase 범위를 유지하며 거래·부상·외부 ID 테이블은 이번 스키마에 추가하지 않는다.
-- ============================================================

-- ── 참조 테이블 ──────────────────────────────────────────────

CREATE TABLE "players" (
	"player_id"	varchar(20)	NOT NULL,
	"birth_year"	int		NULL,
	"name_first"	varchar(50)	NULL,
	"name_last"	varchar(50)	NULL,
	"bats"		varchar(1)	NULL,
	"throws"	varchar(1)	NULL,
	"debut"		date		NULL,
	"final_game"	date		NULL
);
COMMENT ON COLUMN "players"."player_id" IS '선수 식별자 (Lahman playerID)';
COMMENT ON COLUMN "players"."birth_year" IS '출생연도 - age 계산에 필요';
COMMENT ON COLUMN "players"."name_first" IS '이름';
COMMENT ON COLUMN "players"."name_last" IS '성';
COMMENT ON COLUMN "players"."bats" IS '타석 L/R/B';
COMMENT ON COLUMN "players"."throws" IS '투구 손 L/R';
COMMENT ON COLUMN "players"."debut" IS '데뷔일';
COMMENT ON COLUMN "players"."final_game" IS '마지막 경기일';

CREATE TABLE "franchises" (
	"franch_id"	varchar(20)	NOT NULL,
	"franch_name"	varchar(100)	NULL
);
COMMENT ON COLUMN "franchises"."franch_id" IS '프랜차이즈 식별자';
COMMENT ON COLUMN "franchises"."franch_name" IS '프랜차이즈 이름';

CREATE TABLE "teams" (
	"year"		int		NOT NULL,
	"team_id"	varchar(20)	NOT NULL,
	"lg_id"		varchar(10)	NULL,
	"franch_id"	varchar(20)	NOT NULL,
	"div_id"	varchar(10)	NULL,
	"rank"		int		NULL,
	"g"		int		NULL,
	"w"		int		NULL,
	"l"		int		NULL,
	"win_rate"	float		NULL,
	"name"		varchar(100)	NULL,
	"park"		varchar(100)	NULL
);
COMMENT ON COLUMN "teams"."year" IS '시즌';
COMMENT ON COLUMN "teams"."team_id" IS '시즌별 팀 코드';
COMMENT ON COLUMN "teams"."lg_id" IS 'AL/NL';
COMMENT ON COLUMN "teams"."franch_id" IS '프랜차이즈 식별자';
COMMENT ON COLUMN "teams"."div_id" IS '지구';
COMMENT ON COLUMN "teams"."rank" IS '시즌 최종 순위';
COMMENT ON COLUMN "teams"."g" IS '팀 경기수 - g_ratio 분모';
COMMENT ON COLUMN "teams"."w" IS '승';
COMMENT ON COLUMN "teams"."l" IS '패';
COMMENT ON COLUMN "teams"."win_rate" IS 'W/(W+L) - load.py에서 계산해 적재';
COMMENT ON COLUMN "teams"."name" IS '팀 이름';
COMMENT ON COLUMN "teams"."park" IS '홈구장';

-- ── 원천 테이블 (Lahman 원본 컬럼명 그대로) ──────────────────

CREATE TABLE "batting_stats" (
	"year"		int		NOT NULL,
	"stint"		int		NOT NULL,
	"player_id"	varchar(20)	NOT NULL,
	"team_id"	varchar(20)	NULL,
	"lg_id"		varchar(10)	NULL,
	"g"		int		NULL,
	"ab"		int		NULL,
	"R"		int		NULL,
	"h"		int		NULL,
	"2B"		int		NULL,
	"3B"		int		NULL,
	"hr"		int		NULL,
	"rbi"		int		NULL,
	"sb"		int		NULL,
	"cs"		int		NULL,
	"bb"		int		NULL,
	"so"		int		NULL,
	"ibb"		int		NULL,
	"hbp"		int		NULL,
	"sh"		int		NULL,
	"sf"		int		NULL,
	"gidp"		int		NULL
);
COMMENT ON COLUMN "batting_stats"."year" IS '시즌';
COMMENT ON COLUMN "batting_stats"."stint" IS '시즌 내 소속 순번 (트레이드 판별)';
COMMENT ON COLUMN "batting_stats"."player_id" IS '선수 식별자';
COMMENT ON COLUMN "batting_stats"."team_id" IS '팀 코드';
COMMENT ON COLUMN "batting_stats"."lg_id" IS 'AL/NL';
COMMENT ON COLUMN "batting_stats"."g" IS '출전 경기 G';
COMMENT ON COLUMN "batting_stats"."ab" IS '타수 AB';
COMMENT ON COLUMN "batting_stats"."R" IS '득점 R';
COMMENT ON COLUMN "batting_stats"."h" IS '안타 H';
COMMENT ON COLUMN "batting_stats"."2B" IS '2루타';
COMMENT ON COLUMN "batting_stats"."3B" IS '3루타';
COMMENT ON COLUMN "batting_stats"."hr" IS '홈런 HR';
COMMENT ON COLUMN "batting_stats"."rbi" IS '타점 RBI';
COMMENT ON COLUMN "batting_stats"."sb" IS '도루 SB';
COMMENT ON COLUMN "batting_stats"."cs" IS '도루자 CS';
COMMENT ON COLUMN "batting_stats"."bb" IS '볼넷 BB - OBP 계산 필수';
COMMENT ON COLUMN "batting_stats"."so" IS '삼진 SO';
COMMENT ON COLUMN "batting_stats"."ibb" IS '고의사구';
COMMENT ON COLUMN "batting_stats"."hbp" IS '몸에 맞는 볼 - OBP 계산 필수';
COMMENT ON COLUMN "batting_stats"."sh" IS '희생번트';
COMMENT ON COLUMN "batting_stats"."sf" IS '희생플라이 - OBP 계산 필수';
COMMENT ON COLUMN "batting_stats"."gidp" IS '병살타';

CREATE TABLE "pitching_stats" (
	"year"		int		NOT NULL,
	"stint"		int		NOT NULL,
	"player_id"	varchar(20)	NOT NULL,
	"team_id"	varchar(20)	NULL,
	"lg_id"		varchar(10)	NULL,
	"w"		int		NULL,
	"l"		int		NULL,
	"g"		int		NULL,
	"gs"		int		NULL,
	"sv"		int		NULL,
	"ipouts"	int		NULL,
	"h"		int		NULL,
	"er"		int		NULL,
	"hr"		int		NULL,
	"bb"		int		NULL,
	"so"		int		NULL,
	"era"		float		NULL,
	"hbp"		int		NULL,
	"r"		int		NULL
);
COMMENT ON COLUMN "pitching_stats"."year" IS '시즌';
COMMENT ON COLUMN "pitching_stats"."stint" IS '시즌 내 소속 순번';
COMMENT ON COLUMN "pitching_stats"."player_id" IS '선수 식별자';
COMMENT ON COLUMN "pitching_stats"."team_id" IS '팀 코드';
COMMENT ON COLUMN "pitching_stats"."lg_id" IS 'AL/NL';
COMMENT ON COLUMN "pitching_stats"."w" IS '승';
COMMENT ON COLUMN "pitching_stats"."l" IS '패';
COMMENT ON COLUMN "pitching_stats"."g" IS '등판 경기';
COMMENT ON COLUMN "pitching_stats"."gs" IS '선발 등판';
COMMENT ON COLUMN "pitching_stats"."sv" IS '세이브';
COMMENT ON COLUMN "pitching_stats"."ipouts" IS '이닝*3 (아웃카운트)';
COMMENT ON COLUMN "pitching_stats"."h" IS '피안타';
COMMENT ON COLUMN "pitching_stats"."er" IS '자책점';
COMMENT ON COLUMN "pitching_stats"."hr" IS '피홈런';
COMMENT ON COLUMN "pitching_stats"."bb" IS '볼넷 - WHIP 계산 필수';
COMMENT ON COLUMN "pitching_stats"."so" IS '탈삼진';
COMMENT ON COLUMN "pitching_stats"."era" IS '평균자책점';
COMMENT ON COLUMN "pitching_stats"."hbp" IS '몸에맞는볼';
COMMENT ON COLUMN "pitching_stats"."r" IS '실점';

CREATE TABLE "fielding_stats" (
	"year"		int		NOT NULL,
	"stint"		int		NOT NULL,
	"position"	varchar(10)	NOT NULL,
	"player_id"	varchar(20)	NOT NULL,
	"team_id"	varchar(20)	NULL,
	"lg_id"		varchar(10)	NULL,
	"g"		int		NULL,
	"po"		int		NULL,
	"a"		int		NULL,
	"e"		int		NULL,
	"dp"		int		NULL
);
COMMENT ON COLUMN "fielding_stats"."year" IS '시즌';
COMMENT ON COLUMN "fielding_stats"."stint" IS '시즌 내 소속 순번';
COMMENT ON COLUMN "fielding_stats"."position" IS '수비 포지션';
COMMENT ON COLUMN "fielding_stats"."player_id" IS '선수 식별자';
COMMENT ON COLUMN "fielding_stats"."team_id" IS '팀 코드';
COMMENT ON COLUMN "fielding_stats"."lg_id" IS 'AL/NL';
COMMENT ON COLUMN "fielding_stats"."g" IS '수비 출전';
COMMENT ON COLUMN "fielding_stats"."po" IS '자살(풋아웃) - 수비율 계산 필수';
COMMENT ON COLUMN "fielding_stats"."a" IS '보살(어시스트) - 수비율 계산 필수';
COMMENT ON COLUMN "fielding_stats"."e" IS '실책 - 수비율 계산 필수';
COMMENT ON COLUMN "fielding_stats"."dp" IS '병살 관여';

CREATE TABLE "appearances" (
	"year"		int		NOT NULL,
	"team_id"	varchar(20)	NOT NULL,
	"lg_id"		varchar(10)	NULL,
	"player_id"	varchar(20)	NOT NULL,
	"g_all"		int		NULL,
	"g_batting"	int		NULL,
	"g_defense"	int		NULL,
	"g_p"		int		NULL,
	"g_c"		int		NULL,
	"g_1b"		int		NULL,
	"g_2b"		int		NULL,
	"g_3b"		int		NULL,
	"g_ss"		int		NULL,
	"g_lf"		int		NULL,
	"g_cf"		int		NULL,
	"g_rf"		int		NULL,
	"g_of"		int		NULL,
	"g_dh"		int		NULL
);
COMMENT ON COLUMN "appearances"."year" IS '시즌';
COMMENT ON COLUMN "appearances"."team_id" IS '팀 코드';
COMMENT ON COLUMN "appearances"."lg_id" IS 'AL/NL';
COMMENT ON COLUMN "appearances"."player_id" IS '선수 식별자';
COMMENT ON COLUMN "appearances"."g_all" IS '전체 출전';
COMMENT ON COLUMN "appearances"."g_batting" IS '타자로 출전';
COMMENT ON COLUMN "appearances"."g_defense" IS '수비 출전';
COMMENT ON COLUMN "appearances"."g_p" IS '투수 출전 - role(P) 판별용';
COMMENT ON COLUMN "appearances"."g_c" IS '포수 출전';
COMMENT ON COLUMN "appearances"."g_1b" IS '1루수 출전';
COMMENT ON COLUMN "appearances"."g_2b" IS '2루수 출전';
COMMENT ON COLUMN "appearances"."g_3b" IS '3루수 출전';
COMMENT ON COLUMN "appearances"."g_ss" IS '유격수 출전';
COMMENT ON COLUMN "appearances"."g_lf" IS '좌익수 출전';
COMMENT ON COLUMN "appearances"."g_cf" IS '중견수 출전';
COMMENT ON COLUMN "appearances"."g_rf" IS '우익수 출전';
COMMENT ON COLUMN "appearances"."g_of" IS '외야수 출전';
COMMENT ON COLUMN "appearances"."g_dh" IS '지명타자 출전';

CREATE TABLE "allstar" (
	"player_id"	varchar(20)	NOT NULL,
	"year"		int		NOT NULL,
	"game_num"	int		NOT NULL,
	"team_id"	varchar(20)	NULL
);
COMMENT ON COLUMN "allstar"."player_id" IS '선수 식별자';
COMMENT ON COLUMN "allstar"."year" IS '시즌';
COMMENT ON COLUMN "allstar"."game_num" IS '올스타 경기 번호';
COMMENT ON COLUMN "allstar"."team_id" IS '팀 코드';

-- ── 가공 테이블 (D의 contract.py SCHEMA와 1:1 일치) ──────────

CREATE TABLE "player_season" (
	"player_id"		varchar(20)	NOT NULL,
	"season"		int		NOT NULL,
	"team_last"		varchar(20)	NOT NULL,
	"franch_id"		varchar(20)	NOT NULL,
	"league"		varchar(10)	NOT NULL,
	"role"			varchar(10)	NOT NULL,
	"age"			float		NULL,
	"exp"			int		NOT NULL,
	"n_stint"		int		NOT NULL,
	"g_ratio"		float		NULL,
	"g_ratio_prev"		float		NULL,
	"g_chg"			float		NULL,
	"off_score"		float		NULL,
	"pit_score"		float		NULL,
	"def_score"		float		NULL,
	"overall_score"		float		NULL,
	"ops_z"			float		NULL,
	"ops_z_prev"		float		NULL,
	"era_z"			float		NULL,
	"whip_z"		float		NULL,
	"team_wr"		float		NULL,
	"allstar"		int		NULL,
	"y_departed"		float		NULL,
	"y_path"		varchar(30)	NULL,
	"y_fa_release"		varchar(30)	NULL,
	"y_returned"		float		NULL,
	"y_next_score"		float		NULL
);
COMMENT ON COLUMN "player_season"."player_id" IS '선수 식별자';
COMMENT ON COLUMN "player_season"."season" IS '시즌';
COMMENT ON COLUMN "player_season"."team_last" IS '시즌 종료 시점 소속팀';
COMMENT ON COLUMN "player_season"."franch_id" IS '프랜차이즈 식별자 - L1 이탈판정 기준';
COMMENT ON COLUMN "player_season"."league" IS 'mlb 또는 kbo';
COMMENT ON COLUMN "player_season"."role" IS 'B, P 또는 TWO';
COMMENT ON COLUMN "player_season"."age" IS '나이';
COMMENT ON COLUMN "player_season"."exp" IS '리그 경력 시즌 수 - L2b 판별 키';
COMMENT ON COLUMN "player_season"."n_stint" IS '소속팀 수; 2 이상이면 시즌 중 트레이드';
COMMENT ON COLUMN "player_season"."g_ratio" IS 'G / team_games';
COMMENT ON COLUMN "player_season"."g_ratio_prev" IS '전년도 g_ratio';
COMMENT ON COLUMN "player_season"."g_chg" IS 'g_ratio 변화율';
COMMENT ON COLUMN "player_season"."off_score" IS '공격 전력 점수 0~100';
COMMENT ON COLUMN "player_season"."pit_score" IS '투구 전력 점수 0~100';
COMMENT ON COLUMN "player_season"."def_score" IS '수비 전력 점수 0~100';
COMMENT ON COLUMN "player_season"."overall_score" IS '종합 전력 점수 0~100';
COMMENT ON COLUMN "player_season"."ops_z" IS '리그 내 OPS z-score';
COMMENT ON COLUMN "player_season"."ops_z_prev" IS '전년도 ops_z';
COMMENT ON COLUMN "player_season"."era_z" IS '리그 내 ERA z-score';
COMMENT ON COLUMN "player_season"."whip_z" IS '리그 내 WHIP z-score';
COMMENT ON COLUMN "player_season"."team_wr" IS '소속팀 승률';
COMMENT ON COLUMN "player_season"."allstar" IS '올스타 선정 여부 0/1';
COMMENT ON COLUMN "player_season"."y_departed" IS 'L1 이탈 여부 0.0/1.0 (nullable); pandas NaN 적재 호환을 위해 float 유지';
COMMENT ON COLUMN "player_season"."y_path" IS 'L2 관측 경로: trade, offseason_move, league_exit (nullable)';
COMMENT ON COLUMN "player_season"."y_fa_release" IS '조건부 L2b: y_path=offseason_move인 행만 release_certain, fa_est, release_est (nullable)';
COMMENT ON COLUMN "player_season"."y_returned" IS 'L3 복귀 여부 0.0/1.0 (nullable); 은퇴·부상 원인 라벨이 아님';
COMMENT ON COLUMN "player_season"."y_next_score" IS '다음 시즌 전력 회귀 타깃 (nullable)';

CREATE TABLE "team_season" (
	"year"		int	NOT NULL,
	"team_id"	varchar(20)	NOT NULL,
	"bat_strength"	float	NULL,
	"pit_strength"	float	NULL,
	"def_strength"	float	NULL,
	"win_rate"	float	NULL,
	"pred_rank"	int	NULL,
	"risk_index"	float	NULL
);
COMMENT ON COLUMN "team_season"."year" IS '시즌';
COMMENT ON COLUMN "team_season"."team_id" IS '팀 식별자';
COMMENT ON COLUMN "team_season"."bat_strength" IS '팀 타격 전력';
COMMENT ON COLUMN "team_season"."pit_strength" IS '팀 투구 전력';
COMMENT ON COLUMN "team_season"."def_strength" IS '팀 수비 전력';
COMMENT ON COLUMN "team_season"."win_rate" IS '실제 또는 예측 승률';
COMMENT ON COLUMN "team_season"."pred_rank" IS '예상 순위';
COMMENT ON COLUMN "team_season"."risk_index" IS '팀 이탈 위험 지수';

CREATE TABLE "games" (
	"game_pk"	bigint	NOT NULL,
	"season"	int	NOT NULL,
	"game_date"	date	NOT NULL,
	"league"	varchar(10)	NOT NULL,
	"home_team"	varchar(20)	NOT NULL,
	"away_team"	varchar(20)	NOT NULL,
	"home_strength"	float	NULL,
	"away_strength"	float	NULL,
	"home_sp_era"	float	NULL,
	"away_sp_era"	float	NULL,
	"home_rest"	int	NULL,
	"away_rest"	int	NULL,
	"home_last10"	float	NULL,
	"away_last10"	float	NULL,
	"y_home_win"	float	NULL
);
COMMENT ON COLUMN "games"."game_pk" IS 'MLB Stats API 경기 고유 ID';
COMMENT ON COLUMN "games"."season" IS '팀 FK 연결용 시즌';
COMMENT ON COLUMN "games"."game_date" IS '경기일';
COMMENT ON COLUMN "games"."league" IS 'mlb 또는 kbo';
COMMENT ON COLUMN "games"."home_team" IS '홈팀';
COMMENT ON COLUMN "games"."away_team" IS '원정팀';
COMMENT ON COLUMN "games"."home_strength" IS '홈팀 전력';
COMMENT ON COLUMN "games"."away_strength" IS '원정팀 전력';
COMMENT ON COLUMN "games"."home_sp_era" IS '홈 선발 ERA';
COMMENT ON COLUMN "games"."away_sp_era" IS '원정 선발 ERA';
COMMENT ON COLUMN "games"."home_rest" IS '홈팀 휴식일';
COMMENT ON COLUMN "games"."away_rest" IS '원정팀 휴식일';
COMMENT ON COLUMN "games"."home_last10" IS '홈팀 최근 10경기 성적';
COMMENT ON COLUMN "games"."away_last10" IS '원정팀 최근 10경기 성적';
COMMENT ON COLUMN "games"."y_home_win" IS '홈팀 승리 여부 0.0/1.0; 미래 경기는 NULL; pandas NaN 적재 호환을 위해 float 유지';

-- ── 기본키 ────────────────────────────────────────────────

ALTER TABLE "players" ADD CONSTRAINT "PK_PLAYERS" PRIMARY KEY ("player_id");
ALTER TABLE "franchises" ADD CONSTRAINT "PK_FRANCHISES" PRIMARY KEY ("franch_id");
ALTER TABLE "teams" ADD CONSTRAINT "PK_TEAMS" PRIMARY KEY ("year", "team_id");
ALTER TABLE "batting_stats" ADD CONSTRAINT "PK_BATTING_STATS" PRIMARY KEY ("year", "stint", "player_id");
ALTER TABLE "pitching_stats" ADD CONSTRAINT "PK_PITCHING_STATS" PRIMARY KEY ("year", "stint", "player_id");
ALTER TABLE "fielding_stats" ADD CONSTRAINT "PK_FIELDING_STATS" PRIMARY KEY ("year", "stint", "position", "player_id");
ALTER TABLE "appearances" ADD CONSTRAINT "PK_APPEARANCES" PRIMARY KEY ("year", "team_id", "player_id");
ALTER TABLE "allstar" ADD CONSTRAINT "PK_ALLSTAR" PRIMARY KEY ("year", "game_num", "player_id");
ALTER TABLE "player_season" ADD CONSTRAINT "PK_PLAYER_SEASON" PRIMARY KEY ("player_id", "season");
ALTER TABLE "team_season" ADD CONSTRAINT "PK_TEAM_SEASON" PRIMARY KEY ("year", "team_id");
ALTER TABLE "games" ADD CONSTRAINT "PK_GAMES" PRIMARY KEY ("game_pk");

-- ── 외래키 (teams는 복합PK라 반드시 (year, team_id) 묶어서 참조) ──

ALTER TABLE "teams" ADD CONSTRAINT "FK_franchises_TO_teams" FOREIGN KEY ("franch_id")
	REFERENCES "franchises" ("franch_id");

ALTER TABLE "batting_stats" ADD CONSTRAINT "FK_players_TO_batting_stats" FOREIGN KEY ("player_id")
	REFERENCES "players" ("player_id");
ALTER TABLE "pitching_stats" ADD CONSTRAINT "FK_players_TO_pitching_stats" FOREIGN KEY ("player_id")
	REFERENCES "players" ("player_id");
ALTER TABLE "fielding_stats" ADD CONSTRAINT "FK_players_TO_fielding_stats" FOREIGN KEY ("player_id")
	REFERENCES "players" ("player_id");
ALTER TABLE "appearances" ADD CONSTRAINT "FK_players_TO_appearances" FOREIGN KEY ("player_id")
	REFERENCES "players" ("player_id");
ALTER TABLE "allstar" ADD CONSTRAINT "FK_players_TO_allstar" FOREIGN KEY ("player_id")
	REFERENCES "players" ("player_id");

ALTER TABLE "player_season" ADD CONSTRAINT "FK_players_TO_player_season" FOREIGN KEY ("player_id")
	REFERENCES "players" ("player_id");
ALTER TABLE "player_season" ADD CONSTRAINT "FK_teams_TO_player_season" FOREIGN KEY ("season", "team_last")
	REFERENCES "teams" ("year", "team_id");
ALTER TABLE "player_season" ADD CONSTRAINT "FK_franchises_TO_player_season" FOREIGN KEY ("franch_id")
	REFERENCES "franchises" ("franch_id");

ALTER TABLE "team_season" ADD CONSTRAINT "FK_teams_TO_team_season" FOREIGN KEY ("year", "team_id")
	REFERENCES "teams" ("year", "team_id");

ALTER TABLE "games" ADD CONSTRAINT "FK_teams_TO_games_home" FOREIGN KEY ("season", "home_team")
	REFERENCES "teams" ("year", "team_id");
ALTER TABLE "games" ADD CONSTRAINT "FK_teams_TO_games_away" FOREIGN KEY ("season", "away_team")
	REFERENCES "teams" ("year", "team_id");

ALTER TABLE "batting_stats" ADD CONSTRAINT "FK_teams_TO_batting_stats" FOREIGN KEY ("year", "team_id")
	REFERENCES "teams" ("year", "team_id");
ALTER TABLE "pitching_stats" ADD CONSTRAINT "FK_teams_TO_pitching_stats" FOREIGN KEY ("year", "team_id")
	REFERENCES "teams" ("year", "team_id");
ALTER TABLE "fielding_stats" ADD CONSTRAINT "FK_teams_TO_fielding_stats" FOREIGN KEY ("year", "team_id")
	REFERENCES "teams" ("year", "team_id");
ALTER TABLE "appearances" ADD CONSTRAINT "FK_teams_TO_appearances" FOREIGN KEY ("year", "team_id")
	REFERENCES "teams" ("year", "team_id");
ALTER TABLE "allstar" ADD CONSTRAINT "FK_teams_TO_allstar" FOREIGN KEY ("year", "team_id")
	REFERENCES "teams" ("year", "team_id");

-- ── 값 도메인 제약 (기존 contract.py 컬럼명·dtype 유지) ────────

ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_league"
	CHECK ("league" IN ('mlb', 'kbo'));
ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_role"
	CHECK ("role" IN ('B', 'P', 'TWO'));
ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_allstar"
	CHECK ("allstar" IS NULL OR "allstar" IN (0, 1));
ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_y_departed"
	CHECK ("y_departed" IS NULL OR "y_departed" IN (0.0, 1.0));
ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_y_path"
	CHECK ("y_path" IS NULL OR "y_path" IN ('trade', 'offseason_move', 'league_exit'));
ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_y_fa_release"
	CHECK ("y_fa_release" IS NULL OR "y_fa_release" IN ('release_certain', 'fa_est', 'release_est'));
ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_y_fa_release_scope"
	CHECK ("y_fa_release" IS NULL OR "y_path" = 'offseason_move');
ALTER TABLE "player_season" ADD CONSTRAINT "CK_player_season_y_returned"
	CHECK ("y_returned" IS NULL OR "y_returned" IN (0.0, 1.0));

ALTER TABLE "games" ADD CONSTRAINT "CK_games_league"
	CHECK ("league" IN ('mlb', 'kbo'));
ALTER TABLE "games" ADD CONSTRAINT "CK_games_y_home_win"
	CHECK ("y_home_win" IS NULL OR "y_home_win" IN (0.0, 1.0));

-- 은퇴·부상 원인 추정치는 DB 정답 라벨로 저장하지 않는다.
-- reason.py/화면에서 계산하며, 사용자에게 '추정'임을 명시한다.

-- ── RLS 읽기 정책 (없으면 조회 결과가 항상 0행) ────────────────

ALTER TABLE "players" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "franchises" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "teams" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "batting_stats" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "pitching_stats" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "fielding_stats" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "appearances" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "allstar" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "player_season" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "team_season" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "games" ENABLE ROW LEVEL SECURITY;

CREATE POLICY "public read" ON "players" FOR SELECT USING (true);
CREATE POLICY "public read" ON "franchises" FOR SELECT USING (true);
CREATE POLICY "public read" ON "teams" FOR SELECT USING (true);
CREATE POLICY "public read" ON "batting_stats" FOR SELECT USING (true);
CREATE POLICY "public read" ON "pitching_stats" FOR SELECT USING (true);
CREATE POLICY "public read" ON "fielding_stats" FOR SELECT USING (true);
CREATE POLICY "public read" ON "appearances" FOR SELECT USING (true);
CREATE POLICY "public read" ON "allstar" FOR SELECT USING (true);
CREATE POLICY "public read" ON "player_season" FOR SELECT USING (true);
CREATE POLICY "public read" ON "team_season" FOR SELECT USING (true);
CREATE POLICY "public read" ON "games" FOR SELECT USING (true);


-- ══════════════════════════════════════════════════════════════════
-- [2026-08-31 추가] 화면이 실제로 읽는 두 테이블
--
-- 앱(app/)은 지금까지 리포에 들어있는 data/final/features_v1.parquet 과
-- player_injury_stints.csv 를 직접 읽었다. Streamlit Cloud 배포 후 DB에서
-- 끌어오려면 이 둘이 테이블로 있어야 한다.
--
-- features_v1 은 player_season 에서 파생되는 "계약(contract) 산출물"이다.
-- src/features/contract.py 의 SCHEMA 와 컬럼·타입이 1:1로 대응해야 하며,
-- build.py 가 validate() 를 통과시킨 결과만 적재한다.
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE "features_v1" (
	"player_id"		varchar(20)	NOT NULL,
	"season"		int		NOT NULL,
	"team_last"		varchar(20)	NULL,
	"franch_id"		varchar(20)	NULL,
	"league"		varchar(10)	NULL,
	"role"			varchar(10)	NULL,
	"primary_position"	varchar(10)	NULL,
	"age"			float		NULL,
	"exp"			int		NULL,
	"n_stint"		int		NULL,
	"g_ratio"		float		NULL,
	"g_ratio_prev"		float		NULL,
	"g_chg"			float		NULL,
	"off_score"		float		NULL,
	"pit_score"		float		NULL,
	"def_score"		float		NULL,
	"overall_score"		float		NULL,
	"ops_z"			float		NULL,
	"ops_z_prev"		float		NULL,
	"era_z"			float		NULL,
	"whip_z"		float		NULL,
	"team_wr"		float		NULL,
	"y_departed"		float		NULL,
	"y_path"		varchar(30)	NULL,
	"y_fa_release"		varchar(30)	NULL,
	"y_returned"		float		NULL,
	"y_core_departed"	float		NULL,
	"y_next_score"		float		NULL,
	CONSTRAINT "PK_features_v1" PRIMARY KEY ("player_id", "season")
);

-- 시즌 필터가 가장 잦은 조회 패턴이다(화면은 최신 시즌만 본다)
CREATE INDEX "IX_features_v1_season" ON "features_v1" ("season");

-- mlb_injury_pipeline.py 산출물. injury_days_estimated 는 "추정으로 채운
-- 결장일수"라 실측(total_recovery_days)과 반드시 구분해서 보관한다.
CREATE TABLE "player_injury_stints" (
	"player_id"		varchar(20)	NOT NULL,
	"season"		int		NOT NULL,
	"il_stint_count"	int		NULL,
	"first_il_date"		date		NULL,
	"injury_note_sample"	text		NULL,
	"total_recovery_days"	float		NULL,
	"unresolved_stints"	int		NULL,
	"had_injury"		int		NULL,
	"injury_days_estimated"	float		NULL,
	"injury_effective_days"	float		NULL,
	"injury_risk_score"	float		NULL,
	CONSTRAINT "PK_player_injury_stints" PRIMARY KEY ("player_id", "season")
);

ALTER TABLE "features_v1" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "player_injury_stints" ENABLE ROW LEVEL SECURITY;
CREATE POLICY "public read" ON "features_v1" FOR SELECT USING (true);
CREATE POLICY "public read" ON "player_injury_stints" FOR SELECT USING (true);
