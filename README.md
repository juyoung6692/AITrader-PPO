# PPO US Day Trader

Proximal Policy Optimization (PPO) 기반의 미국 주식 **단타 매매 강화학습** 프로젝트입니다.
yfinance에서 5분봉 데이터를 가져와 기술적 지표를 만들고, 자체 Gymnasium 환경에서
PPO 에이전트(stable-baselines3)가 long / flat / short 결정을 학습합니다.

## 구조

```
ppo_trader/
├── config.py      # 하이퍼파라미터·심볼 설정
├── data.py        # yfinance 로더 + parquet 캐시
├── features.py    # 기술적 지표 (RSI, MACD, BB, ATR 등)
├── env.py         # DayTradingEnv (Gymnasium 호환)
├── train.py       # PPO 학습 진입점
└── evaluate.py    # 홀드아웃 백테스트
```

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 학습

```bash
python -m ppo_trader.train \
    --symbols AAPL MSFT NVDA TSLA AMD \
    --interval 5m --period 60d \
    --timesteps 300000 --n-envs 4
```

- 모델은 `models/ppo_daytrader.zip` 으로 저장됩니다.
- 학습 로그 / 체크포인트는 `logs/ppo_daytrader/` 하위에 쌓이고, TensorBoard로 볼 수 있습니다.

## 백테스트

```bash
python -m ppo_trader.evaluate \
    --model models/ppo_daytrader \
    --symbols AAPL MSFT NVDA TSLA AMD \
    --interval 5m --period 60d
```

심볼별 수익률·최대낙폭·샤프(분봉 환산)·최종 자산이 표로 출력됩니다.

## 환경 설계 요약

- **관측**: `window_size`개의 분봉 기술적 지표 + 현재 포지션 / 미실현 손익 / 현금비중
- **행동**: `Discrete(3)` — flat / long / short (`allow_short=False`면 2-action)
- **보상**: 자산 변화율 × `reward_scale` − 회전율·홀딩·드로다운 페널티
- **장 마감 강제 청산**: `end_of_day_flat=True` (날짜 바뀌면 포지션 자동 청산)
- **거래비용**: `commission` + `slippage` 양방향 반영

## 주의

- yfinance 5분봉은 보통 최근 60일까지만 제공됩니다 (`period` 더 길게 잡아도 잘림).
- 본 코드는 **연구용 백테스트** 입니다. 실거래 적용 시 데이터 품질·체결 모델·리스크 한도를 별도로 보강하세요.
