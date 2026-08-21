# HANDOFF — Night-5 실험 야간 운용 (Codex용)

작성: 2026-08-21, Claude 캠페인 세션. 이 문서 하나로 밤샘 운용이 가능해야 한다.
모르는 것이 나오면 즉흥 대응하지 말고 그대로 기록하고 다음 태스크로 넘어갈 것.

## 0. 배경 한 장

- 목표: 동결된 HP100 flow-matching 정책(r0)의 **OOD 충돌율(CR)을 학습만으로**
  낮추는 것 (Validity↑, clearance↑, SR 유지; TtG는 희생 허용). 배포시
  best-of-N 스위치는 "최종병기"로 봉인 — 공식 결과에 사용 금지.
- 현재 챔피언: **N4A2_s12500**
  (`/data3/research1/claude_sfm2_predictive_cfc09ad/champions/N4A2_12500/snapshot_step12500.pt`,
  SHA `bb1c3e1544ccc4c9da8c40caeb3762047fb25f78bb3a327fe5b64500bf2e76f8`).
  레시피: 직전 챔피언 RC3_12500이 best-of-16 MPC-select로 롤아웃한 성공
  에피소드의 실행 스텝(정확 검증기 인증 통과분만, 339,026행)으로 RC3에서
  계속학습(all_open, α.02 hinge, per_gamma_balanced, lr 5e-6, E4)한 것의
  step-12500 스냅샷.
- 성적 (fresh M50, OOD ep0 920000 / ID 930000):
  | 모델 | OOD CR | Val | clr | SR | TO | time | ID CR/SR |
  |---|---|---|---|---|---|---|---|
  | r0 | .551 | .641 | .086 | .448 | — | 7.48 | .049/.941 |
  | RC3_12500 | .480 | .691 | .102 | .514 | .006 | 8.22 | .040/.954 |
  | **N4A2_s12500** | **.391** | .724 | .126 | .589 | .020 | 8.61 | .020/.980 |
- M20 스크린 기준선 (고정 CRN 뱅크 ep0 900000):
  r0 CR .443/Val .636/SR .557/TO .014 · RC3 .329/.710/.657/.014 ·
  N4A2_s12500 .329/.739/.643/.029.
- 확정된 교훈: (1) 증류 2라운드(교사=신챔피언)는 보수화 누적으로 실패
  (timeout 팽창) — **증류는 1라운드가 최적**. (2) 대조 목적함수(DPO/NFT)는
  이 과제에서 CR로 전이 안 됨 — 재시도 금지. (3) M20 승자는 fresh M50
  확인 전에는 승자가 아니다 (winner's curse 상습 확인됨).
- 오늘 밤 가설: 라운드2 실패는 "교사를 갈아탄 것"이 원인이고, **RC3-교사
  데이터의 양·선택압을 키우는 방향**은 아직 열려 있다.

## 1. 환경·접속

- Helios: `ssh dohyun@helios.robotics.caltech.edu` (비밀번호는 운용자가
  별도 전달 — 이 저장소에 절대 기록하지 말 것).
- 저장소: `~/projects/safe_flow_expansion_SFM2-claude-cfc09ad`,
  브랜치 `agent/claude-sfm2-predictive-20260814`. 시작 전 `git pull` 1회.
- Python: `~/miniforge3/envs/cfm_mppi/bin/python`.
- 데이터 루트: `OUT=/data3/research1/claude_sfm2_predictive_cfc09ad`,
  야간 작업 루트 `OUT/night4` (마커 `OUT/night4/markers`, 로그
  `OUT/night4/logs`).
- **GPU 규칙: 0번과 3번만 사용. 1·2번은 다른 사용자 — 절대 점유 금지.**
- 장기 실행은 반드시 `setsid nohup <cmd> > <log> 2>&1 < /dev/null &` 로
  분리 실행 (ssh 끊김·세션 정리에 살아남게).

## 2. 불가침 규칙 (위반 시 캠페인 무효)

1. 동결 파일 수정 금지: `sfm_hp100_predictive_execution.py`,
   `sfm_metrics2.py`, `sfm_hp100_eval.py`, `sfm_hp100_ball_adapter.py`,
   `grid_policy_sfm_hp100.py` (+ 기타 기존 파일 일체 — 이번 밤은 새 코드
   작성 없이 아래 커맨드만 실행한다).
2. 체크포인트 계약: `{state_dict, config}` 엄격 유지, 새 파라미터 추가
   금지. 학습은 전부 `sfm_hp100_raw_train.py` 커맨드로만.
3. 평가 뱅크 규율: M20은 ep0 900000 고정 CRN, fresh M50은 920000/930000.
   **ep0 940000/950000 (M100)은 어떤 이유로도 실행 금지** (사용자 승인
   게이트).
4. `OUT/champions/` 아래 파일 수정·삭제 금지 (복사만 가능).
5. **내가 띄우지 않은 프로세스를 절대 kill 하지 말 것.** GPU 메모리를
   잡고 있어도 살아있는 학습/평가일 수 있다. 어젯밤 실제로 로그가
   버퍼링된 채 조용히 돌던 작업자가 오인 사살된 사고가 있었다. 죽일 수
   있는 것은 자기가 이 문서의 태스크로 띄운 PID뿐.
6. 실패 시: 같은 커맨드 1회 재시도 → 그래도 실패면 로그 마지막 30줄을
   보고서에 붙이고 다음 태스크로. 스크립트·플래그 즉흥 수정 금지.
7. 결과 보고는 실측만. 추정치를 결과처럼 쓰지 말 것.

## 3. Task 0 — 선행 체인 종료 대기 (필수)

night-4 에필로그(챔피언 캘리브레이션 + 시드-3 재현)가 GPU를 쓰고 있을 수
있다. `OUT/night4/markers/epilogue_all.done` 이 생길 때까지 대기 (최대
3시간; Task 1 스크립트에 대기 루프가 이미 들어 있으므로 그냥 Task 1을
띄워도 안전하다). 생겼으면 다음 두 결과를 보고서에 기록:

- `OUT/bestofN_calibration/N4A2_12500_shard{A,B}.json` 의 N별
  pooled CR/SR/TO (신챔피언의 최종병기 곡선).
- `OUT/funnel/screen_m20_N4rep/STAGE_COMPLETE.json` 의 OOD pooled 행들
  (시드-3 재현 밴드; N4A2 본선 M20 .329/.739와 비교).

## 4. Task 1 — 반(反)보수화 증류 스윕 (주력)

준비된 스크립트 하나가 4개 암 학습 + M20 스크린까지 전부 한다:

```
cd ~/projects/safe_flow_expansion_SFM2-claude-cfc09ad && git pull
setsid nohup bash scripts/night5_codex_arms.sh \
  > /data3/research1/claude_sfm2_predictive_cfc09ad/night4/logs/night5_task1_launcher.log 2>&1 < /dev/null &
```

암 구성 (전부 기존 데이터, 새 수집 없음):

| Arm | 시작점 | 데이터 | 변화축 |
|---|---|---|---|
| N5D1 | RC3 | BoN r1+r2 (482k)+negs | RC3-거리 고정, 데이터 2배 |
| N5D2 | RC3 | BoN r1 (339k)+negs | positive_mass=progress_weighted (보수화 반작용) |
| N5D3 | r0 | 308k+BoN r1+r2 (790k)+negs | 전량 집계 재학습 |
| N5D4 | RC3 | BoN r1+negs | 반도즈 E2 (조기 최적 가설) |

완료 마커: `night5_task1.done`. 스크린 결과:
`OUT/funnel/screen_m20_{N5a,N5b}/STAGE_COMPLETE.json` — OOD
(`scene_profile == "double_density_velocity_ood"`) pooled를 표로 정리.

**M50 승격 규칙 (그대로 적용):** M20에서 CR ≤ .329 이면서 SR ≥ .55,
TO ≤ .03 인 체크포인트 중 CR 최저(동률이면 Val 최고) **최대 2개만** fresh
M50 실행:

```
cd ~/projects/safe_flow_expansion_SFM2-claude-cfc09ad
OUT=/data3/research1/claude_sfm2_predictive_cfc09ad
setsid nohup env OUTPUT=$OUT/funnel/shortlist_m50_N5_<LABEL> STAGE=shortlist-m50 \
  CHECKPOINTS="<LABEL>=<체크포인트절대경로>" PHYSICAL_GPU=<0또는3> \
  PYTHON_BIN=$HOME/miniforge3/envs/cfm_mppi/bin/python \
  bash scripts/run_expansion_funnel.sh \
  > $OUT/funnel/shortlist_m50_N5_<LABEL>.log 2>&1 < /dev/null &
```

판정: OOD CR < .391 (챔피언) 이면 "챔피언 후보"로 보고하고 해당
체크포인트를 `OUT/champions/<이름>/` 에 **복사**(cp) + `sha256sum` 기록.
통과 못 하면 수치만 기록. 어떤 경우에도 M100은 안 돌린다.

## 5. Task 2 — BoN-32 교사 수집 + 학생 (Task 1 종료 후)

선택압을 16→32로 올린 RC3-교사 데이터가 더 낮은 CR을 가르치는지 검증:

```
cd ~/projects/safe_flow_expansion_SFM2-claude-cfc09ad
setsid nohup bash scripts/night5_codex_bon32.sh \
  > /data3/research1/claude_sfm2_predictive_cfc09ad/night4/logs/night5_task2_launcher.log 2>&1 < /dev/null &
```

(Task 1 마커를 최대 6시간 기다렸다가 수집 120분 → N5E1 학습 → M20 스크린
`screen_m20_N5c`까지 자동.) 완료 마커 `night5_task2.done`. 수집 로그의
블록별 `chosen_valid_fraction`·`outcome`과, N5E1 스크린 표를 기록. M50
승격 규칙은 Task 1과 동일 (총 M50 예산: 밤 전체 3회 이내).

## 6. 진행 감시 방법

15–20분 간격으로:
```
ls /data3/research1/claude_sfm2_predictive_cfc09ad/night4/markers/
tail -3 /data3/research1/claude_sfm2_predictive_cfc09ad/night4/logs/<해당로그>
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```
학습 로그가 4시간 이상 무변화 + GPU 0% + 마커 없음 → 죽은 것으로 간주,
규칙 2-6 (1회 재시도) 적용. 스크립트는 마커 게이트라 재실행이 안전하다.

## 7. 아침 보고 형식 (필수 산출물)

`OUT/night4/NIGHT5_REPORT.md` 에:
1. Task 0 두 결과 (캘리브레이션 N-곡선, 시드-3 재현 표).
2. Task 1 전체 M20 표 (arm×스냅샷, OOD pooled 6지표) + M50 승격 결과.
3. Task 2 수집 통계 + N5E1 표 (+M50 결과 있으면).
4. 판정 3줄: 챔피언 교체 여부 / 어떤 축이 효과·무효였나 / 남긴 문제.
5. 사고·재시도 이력 전부 (숨기지 말 것).
