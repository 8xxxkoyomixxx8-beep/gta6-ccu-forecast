# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/8xxxkoyomixxx8-beep/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/8xxxkoyomixxx8-beep/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for Grand Theft Auto VI's launch-day Twitch audience. Given everything recorded up to the end of release day's first hour (that hour's concurrent-viewer count included), it predicts the "Grand Theft Auto VI" Twitch category's concurrent viewers for each of the day's 24 UTC hours as five quantiles (q05, q25, q50, q75, q95). The model restricts its comparable pool to rolling midnight console launches that were already live at hour 1, expresses each comparable's release day as log ratios to its own hour-1 viewers, and anchors the empirical quantiles of those ratios — blended with a lognormal fit and smoothed across neighbouring hours — on the actual first-hour print. Standard library only, no network, deterministic. In leave-one-out validation over the comparable launches it scores a log-pinball loss of 0.307 against the starter baseline's 0.360.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
# reads every *.in case in the input dir, writes one <id>.out per case
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
