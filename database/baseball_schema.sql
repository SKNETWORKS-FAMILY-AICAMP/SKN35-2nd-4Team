CREATE TABLE "batting_stats" (
	"year"	int		NOT NULL,
	"stint"	int		NOT NULL,
	"player_id"	varchar(20)		NOT NULL,
	"team_id"	varchar(20)		NULL,
	"at_bats"	int		NULL,
	"hits"	int		NULL,
	"doubles"	int		NULL,
	"triples"	int		NULL,
	"home_runs"	int		NULL,
	"runs"	int		NULL,
	"total_bases"	float		NULL,
	"run_score"	float		NULL,
	"rbi"	int		NULL,
	"weighted_batting_score"	VARCHAR(255)		NULL,
	"weighted_run_production"	VARCHAR(255)		NULL,
	"plate_appearances"	VARCHAR(255)		NULL
);

COMMENT ON COLUMN "batting_stats"."year" IS '시즌';

COMMENT ON COLUMN "batting_stats"."stint" IS '시즌 내 소속 순번';

COMMENT ON COLUMN "batting_stats"."player_id" IS '선수 식별자';

COMMENT ON COLUMN "batting_stats"."team_id" IS '팀 코드';

COMMENT ON COLUMN "batting_stats"."at_bats" IS '타수 AB';

COMMENT ON COLUMN "batting_stats"."hits" IS '안타 H';

COMMENT ON COLUMN "batting_stats"."doubles" IS '2루타 2B';

COMMENT ON COLUMN "batting_stats"."triples" IS '3루타 3B';

COMMENT ON COLUMN "batting_stats"."home_runs" IS '홈런 HR';

COMMENT ON COLUMN "batting_stats"."runs" IS '득점 R';

COMMENT ON COLUMN "batting_stats"."total_bases" IS 'H + 2B + 2×3B + 3×HR';

COMMENT ON COLUMN "batting_stats"."run_score" IS '득점 가중치를 반영한 기여 점수';

COMMENT ON COLUMN "batting_stats"."rbi" IS '타점 RBI';

COMMENT ON COLUMN "batting_stats"."weighted_batting_score" IS '루타와 득점 가중치를 합산한 타격 점수';

COMMENT ON COLUMN "batting_stats"."weighted_run_production" IS '득점과 타점 생산성 가중 점수';

COMMENT ON COLUMN "batting_stats"."plate_appearances" IS '타석 PA';

CREATE TABLE "appearances" (
	"year"	int		NOT NULL,
	"team_id"	varchar(20)		NOT NULL,
	"player_id"	varchar(20)		NOT NULL
);

COMMENT ON COLUMN "appearances"."year" IS '시즌';

COMMENT ON COLUMN "appearances"."team_id" IS '팀 코드';

COMMENT ON COLUMN "appearances"."player_id" IS '선수 식별자';

CREATE TABLE "players" (
	"player_id"	varchar(20)		NOT NULL
);

COMMENT ON COLUMN "players"."player_id" IS '선수 식별자';

CREATE TABLE "franchises" (
	"franch_id"	varchar(20)		NOT NULL
);

COMMENT ON COLUMN "franchises"."franch_id" IS '프랜차이즈 식별자';

CREATE TABLE "teams" (
	"year"	int		NOT NULL,
	"team_id"	varchar(20)		NOT NULL,
	"franch_id"	varchar(20)		NOT NULL
);

COMMENT ON COLUMN "teams"."year" IS '시즌';

COMMENT ON COLUMN "teams"."team_id" IS '시즌별 팀 코드';

COMMENT ON COLUMN "teams"."franch_id" IS '프랜차이즈 식별자';

CREATE TABLE "games" (
	"game_pk"	bigint		NOT NULL,
	"season"	int		NOT NULL,
	"game_date"	date		NOT NULL,
	"league"	varchar(10)		NOT NULL,
	"home_team"	varchar(20)		NOT NULL,
	"away_team"	varchar(20)		NOT NULL,
	"home_strength"	float		NULL,
	"away_strength"	float		NULL,
	"home_sp_era"	float		NULL,
	"away_sp_era"	float		NULL,
	"home_rest"	int		NULL,
	"away_rest"	int		NULL,
	"home_last10"	float		NULL,
	"away_last10"	float		NULL,
	"y_home_win"	float		NULL
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

COMMENT ON COLUMN "games"."y_home_win" IS '홈팀 승리 여부 0/1; 미래 경기는 NULL';

CREATE TABLE "allstar" (
	"year"	int		NOT NULL,
	"game_num"	int		NOT NULL,
	"player_id"	varchar(20)		NOT NULL
);

COMMENT ON COLUMN "allstar"."year" IS '시즌';

COMMENT ON COLUMN "allstar"."game_num" IS '올스타 경기 번호';

COMMENT ON COLUMN "allstar"."player_id" IS '선수 식별자';

CREATE TABLE "fielding_stats" (
	"year"	int		NOT NULL,
	"stint"	int		NOT NULL,
	"position"	varchar(10)		NOT NULL,
	"player_id"	varchar(20)		NOT NULL
);

COMMENT ON COLUMN "fielding_stats"."year" IS '시즌';

COMMENT ON COLUMN "fielding_stats"."stint" IS '시즌 내 소속 순번';

COMMENT ON COLUMN "fielding_stats"."position" IS '수비 포지션';

COMMENT ON COLUMN "fielding_stats"."player_id" IS '선수 식별자';

CREATE TABLE "player_season" (
	"player_id"	varchar(20)		NOT NULL,
	"season"	int		NOT NULL,
	"team_last"	varchar(20)		NOT NULL,
	"league"	varchar(10)		NOT NULL,
	"n_stint"	int		NOT NULL,
	"exp"	int		NOT NULL,
	"role"	varchar(10)		NOT NULL,
	"age"	float		NULL,
	"g_ratio"	float		NULL,
	"off_score"	float		NULL,
	"pit_score"	float		NULL,
	"def_score"	float		NULL,
	"overall_score"	float		NULL,
	"y_departed"	float		NULL,
	"y_path"	varchar(30)		NULL,
	"y_fa_release"	varchar(30)		NULL,
	"y_returned"	float		NULL,
	"y_next_score"	float		NULL
);

COMMENT ON COLUMN "player_season"."player_id" IS '선수 식별자';

COMMENT ON COLUMN "player_season"."season" IS '시즌';

COMMENT ON COLUMN "player_season"."team_last" IS '시즌 종료 시점 소속팀';

COMMENT ON COLUMN "player_season"."league" IS 'mlb 또는 kbo';

COMMENT ON COLUMN "player_season"."n_stint" IS '소속팀 수; 2 이상이면 시즌 중 트레이드';

COMMENT ON COLUMN "player_season"."exp" IS '리그 경력 시즌 수';

COMMENT ON COLUMN "player_season"."role" IS 'B, P 또는 TWO';

COMMENT ON COLUMN "player_season"."age" IS '나이';

COMMENT ON COLUMN "player_season"."g_ratio" IS 'G / team_games';

COMMENT ON COLUMN "player_season"."off_score" IS '공격 전력 점수 0~100';

COMMENT ON COLUMN "player_season"."pit_score" IS '투구 전력 점수 0~100';

COMMENT ON COLUMN "player_season"."def_score" IS '수비 전력 점수 0~100';

COMMENT ON COLUMN "player_season"."overall_score" IS '종합 전력 점수 0~100';

COMMENT ON COLUMN "player_season"."y_departed" IS 'L1 이탈 여부 0/1';

COMMENT ON COLUMN "player_season"."y_path" IS 'L2 trade, offseason_move, league_exit';

COMMENT ON COLUMN "player_season"."y_fa_release" IS 'L2b release_certain, fa_est, release_est';

COMMENT ON COLUMN "player_season"."y_returned" IS 'L3 복귀 여부 0/1';

COMMENT ON COLUMN "player_season"."y_next_score" IS '다음 시즌 전력 회귀 타깃';

CREATE TABLE "team_season" (
	"year"	int		NOT NULL,
	"team_id"	varchar(20)		NOT NULL,
	"bat_strength"	float		NULL,
	"pit_strength"	float		NULL,
	"def_strength"	float		NULL,
	"win_rate"	float		NULL,
	"pred_rank"	int		NULL,
	"risk_index"	float		NULL
);

COMMENT ON COLUMN "team_season"."year" IS '시즌';

COMMENT ON COLUMN "team_season"."team_id" IS '팀 식별자';

COMMENT ON COLUMN "team_season"."bat_strength" IS '팀 타격 전력';

COMMENT ON COLUMN "team_season"."pit_strength" IS '팀 투구 전력';

COMMENT ON COLUMN "team_season"."def_strength" IS '팀 수비 전력';

COMMENT ON COLUMN "team_season"."win_rate" IS '실제 또는 예측 승률';

COMMENT ON COLUMN "team_season"."pred_rank" IS '예상 순위';

COMMENT ON COLUMN "team_season"."risk_index" IS '팀 이탈 위험 지수';

CREATE TABLE "pitching_stats" (
	"year"	int		NOT NULL,
	"stint"	int		NOT NULL,
	"player_id"	varchar(20)		NOT NULL,
	"team_id"	varchar(20)		NULL
);

COMMENT ON COLUMN "pitching_stats"."year" IS '시즌';

COMMENT ON COLUMN "pitching_stats"."stint" IS '시즌 내 소속 순번';

COMMENT ON COLUMN "pitching_stats"."player_id" IS '선수 식별자';

COMMENT ON COLUMN "pitching_stats"."team_id" IS '팀 코드';

ALTER TABLE "batting_stats" ADD CONSTRAINT "PK_BATTING_STATS" PRIMARY KEY (
	"year",
	"stint",
	"player_id"
);

ALTER TABLE "appearances" ADD CONSTRAINT "PK_APPEARANCES" PRIMARY KEY (
	"year",
	"team_id",
	"player_id"
);

ALTER TABLE "players" ADD CONSTRAINT "PK_PLAYERS" PRIMARY KEY (
	"player_id"
);

ALTER TABLE "franchises" ADD CONSTRAINT "PK_FRANCHISES" PRIMARY KEY (
	"franch_id"
);

ALTER TABLE "teams" ADD CONSTRAINT "PK_TEAMS" PRIMARY KEY (
	"year",
	"team_id"
);

ALTER TABLE "games" ADD CONSTRAINT "PK_GAMES" PRIMARY KEY (
	"game_pk"
);

ALTER TABLE "allstar" ADD CONSTRAINT "PK_ALLSTAR" PRIMARY KEY (
	"year",
	"game_num",
	"player_id"
);

ALTER TABLE "fielding_stats" ADD CONSTRAINT "PK_FIELDING_STATS" PRIMARY KEY (
	"year",
	"stint",
	"position",
	"player_id"
);

ALTER TABLE "player_season" ADD CONSTRAINT "PK_PLAYER_SEASON" PRIMARY KEY (
	"player_id",
	"season"
);

ALTER TABLE "team_season" ADD CONSTRAINT "PK_TEAM_SEASON" PRIMARY KEY (
	"year",
	"team_id"
);

ALTER TABLE "pitching_stats" ADD CONSTRAINT "PK_PITCHING_STATS" PRIMARY KEY (
	"year",
	"stint",
	"player_id"
);

ALTER TABLE "batting_stats" ADD CONSTRAINT "FK_players_TO_batting_stats_1" FOREIGN KEY (
	"player_id"
)
REFERENCES "players" (
	"player_id"
);

ALTER TABLE "appearances" ADD CONSTRAINT "FK_players_TO_appearances_1" FOREIGN KEY (
	"player_id"
)
REFERENCES "players" (
	"player_id"
);

ALTER TABLE "allstar" ADD CONSTRAINT "FK_players_TO_allstar_1" FOREIGN KEY (
	"player_id"
)
REFERENCES "players" (
	"player_id"
);

ALTER TABLE "fielding_stats" ADD CONSTRAINT "FK_players_TO_fielding_stats_1" FOREIGN KEY (
	"player_id"
)
REFERENCES "players" (
	"player_id"
);

ALTER TABLE "player_season" ADD CONSTRAINT "FK_players_TO_player_season_1" FOREIGN KEY (
	"player_id"
)
REFERENCES "players" (
	"player_id"
);

ALTER TABLE "player_season" ADD CONSTRAINT "FK_teams_TO_player_season_1" FOREIGN KEY (
	"season"
)
REFERENCES "teams" (
	"year"
);

ALTER TABLE "team_season" ADD CONSTRAINT "FK_teams_TO_team_season_1" FOREIGN KEY (
	"year"
)
REFERENCES "teams" (
	"year"
);

ALTER TABLE "team_season" ADD CONSTRAINT "FK_teams_TO_team_season_2" FOREIGN KEY (
	"team_id"
)
REFERENCES "teams" (
	"team_id"
);

ALTER TABLE "pitching_stats" ADD CONSTRAINT "FK_players_TO_pitching_stats_1" FOREIGN KEY (
	"player_id"
)
REFERENCES "players" (
	"player_id"
);

