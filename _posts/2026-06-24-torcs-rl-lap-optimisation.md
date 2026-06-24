---
title: "TORCS Corkscrew Revisited: Training a SAC Agent From Scratch With Cleaner Reward Engineering"
date: 2026-06-24
categories: [Reinforcement Learning, Autonomous Driving]
tags: [TORCS, SAC, Reward Engineering, Curriculum Learning, Deep RL]
author: Choeyunbeom
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

# TORCS Corkscrew Revisited: Training a SAC Agent From Scratch With Cleaner Reward Engineering

**TL;DR**: Started over on the TORCS Corkscrew challenge with a cleaner codebase and more disciplined reward design, informed by the failure modes from [the first attempt](https://choeyunbeom.github.io/reinforcement%20learning/autonomous%20driving/2026/01/28/torcs-rl-journey.html). Solved a 660-episode S-corner deadlock, hit a best lap of **1:34**, and found that reward function audits matter far more than hyperparameter tuning.

## Abstract

Having documented three distinct failure modes in the first attempt — perverse crash incentives, a parking exploit, and PPO catastrophic forgetting — this round started fresh with a self-written SAC implementation on the same track. Rather than retrofitting fixes onto the previous codebase, the goal was to build the reward function from the ground up with the lessons already in mind. The result: reliable S-corner passage, a 9.9% completion rate within 121 episodes from a focused spawn, and a best lap of 1:34.

**What changed from the first attempt:**
- Self-implemented SAC with auto-entropy tuning and twin-Q critic, replacing Stable-Baselines3
- Reward function rebuilt from scratch — no milestone bonuses, no flat crash penalties
- Three-schedule curriculum (centre-line weight, heading weight, progress weight) driven by completion rate EMA
- Spawn position used as a diagnostic and training tool, not just an environment setting

---

## 1. Why Start Over

The first run produced 37 completions over 4,349 episodes and a best lap of 1:48 on a 3,600m track. The headline number looks reasonable but the methodology was reactive: spot a failure, patch the reward, repeat. By the end, the reward function had accumulated several conflicting terms and hardcoded fixes that were difficult to reason about together.

The cleaner approach was to start with a minimal reward signal and add terms only when there was data justifying them. This post documents that process — what was kept, what was cut, and what the data said at each step.

---

## 2. SAC Implementation

Stable-Baselines3 was replaced with a self-written SAC to have full visibility into the update logic. The core configuration:

```python
STATE_DIM  = 34    # track(19) + speedXYZ(3) + trackPos(1) + angle(1) + rpm(1) + wheelSpin(4) + focus(5)
ACTION_DIM = 3     # [steer, accel, brake]
HIDDEN_DIM = 256

LR           = 3e-4
GAMMA        = 0.99
TAU          = 0.005
BATCH_SIZE   = 256
REPLAY_SIZE  = 500_000
WARMUP_STEPS = 5_000

base_entropy = -float(ACTION_DIM) * 2.0  # = -6.0
```

**Warmup speed limit.** The first attempt taught us that filling the replay buffer with 180 km/h terminal crashes during warmup is counterproductive — the agent learns "go fast and die" before learning basic track-following. During warmup, a hard brake is applied whenever `sp_x > 80 km/h`. The buffer gets populated with survivable, low-speed transitions first.

**Auto-entropy tuning.** Rather than fixing the entropy coefficient, SAC adjusts it automatically to hit `base_entropy = -6.0`. This means the policy self-regulates how exploratory it should be: in easy sections it becomes more deterministic; in novel sections (like the first time it encounters the S-corner) it increases entropy to keep options open.

---

## 3. Reward Function: Built From Lessons

### 3.1 The Core Formula

The first attempt's locomotion reward had a subtle bug:

```python
# First attempt — buggy
track_progress = abs(sp_x * np.cos(angle) - sp_y * np.sin(angle))
```

`abs()` made reverse driving rewarding. A car backing up fast accumulates positive track progress indefinitely, and the wrong-way termination fires too slowly to catch it. This was replaced with a clamp:

```python
# This attempt — fixed
track_progress = sp_x * np.cos(angle) - sp_y * np.sin(angle)
track_progress = max(track_progress, 0.0)
reward = progress_weight * track_progress
```

### 3.2 Survival Bonus

A flat `+1.0` per step for staying on the track. This guarantees that surviving is always better than crashing, regardless of speed. Without this, a crash early in an episode (small negative) can be preferable to driving slowly (small positive minus penalties).

```python
reward += 1.0
```

### 3.3 No Flat Milestone Bonuses

The first attempt used milestone bonuses at 1000m, 2000m, 3000m. The lesson there: flat bonuses at specific distances create local optima. The agent learns to reach the bonus location and then behave suboptimally because the reward topology has a spike at that distance. This time, no milestone bonuses were added.

### 3.4 Termination Boundary: 1.3

The first attempt used `|trackPos| > 1.0`. TORCS's nominal edge is ±1.0, but physical track width extends past this. Terminating at exactly 1.0 was ending episodes on drivable track, which trained the agent to treat the 0.9–1.0 zone as a cliff.

```python
if abs(track_pos) > 1.3:
    reward -= 20.0
    done = True
```

Using the full available width produced noticeably better cornering lines within 50 episodes.

---

## 4. The 2500m Exploit (Discovered Mid-Training)

Despite building more carefully this time, one exploit slipped through. During S-corner debugging, a +50 flat bonus was added at `distRaced > 2500m` to incentivise pushing past that point. The result:

```
Observed behaviour (logged):
  - Episode reaches ~2480m at 120 km/h
  - Policy applies full brake
  - Car decelerates to <5 km/h
  - Stuck timer fires after 150 steps → episode ends
  - Agent collects +50 bonus on the way down, then -20 stuck penalty
  - Net: +30 per episode from this exploit
```

The agent had worked out that stopping before 2500m, collecting the trigger bonus, and dying to the stuck timeout was more profitable than attempting the corner. Exactly the class of exploit the first attempt had taught us to watch for — and it appeared again anyway.

Fix: removed the bonus entirely.

```python
# 2500m milestone bonus removed — was causing deliberate stopping exploit
# _passed_2400 flag kept for future diagnostic use only
```

---

## 5. The S-Corner Problem

### 5.1 660 Episodes of Deadlock

The S-corner at ~2400m was a complete wall for 660 episodes. The intuitive diagnosis was "not braking hard enough." The telemetry said otherwise:

```
Analysis of crash logs at 2400m:
  - Brake input at crash: 0.85+ (near maximum)
  - Steer input at crash: -1.0 (full lock)
  - Angle (yaw vs track) at crash: 38–45°
  - Track position at crash: -0.9 to -1.0
```

The agent was already at full brake and full steer lock. The problem was entry timing: the car was arriving at the apex already rotated 40° off-axis, at which point full braking and steering is physically too late to correct the trajectory.

This reframed the problem from "braking" to "anticipation." The agent needed to begin braking 100–200m before the corner, not at it.

### 5.2 Why the Standard Setup Couldn't Solve It

With a 0m spawn on a 3,600m track, the agent reaches the S-corner only in the final third of a successful episode. In practice, most episodes die before 2,400m, so the S-corner appears in the replay buffer very infrequently — perhaps once every few episodes. The policy has almost no gradient signal for that specific situation.

This is a sampling problem, not a capacity problem.

### 5.3 Spawn as a Training Tool

Moving the spawn to 1600m means every episode starts ~800m before the S-corner. Every episode includes S-corner exposure. The replay buffer fills with S-corner transitions within 20–30 episodes and the policy starts receiving meaningful gradient signal for that section.

```
# corkscrew.xml Starting Grid:
# distance to start = 1600.0  (S-corner focused training)
```

The tradeoff is distributional: a policy trained from 1600m will be competent from that point forward but will have almost no experience of the 0–1600m section. Returning to 0m spawn after S-corner mastery caused the policy to lose S-corner ability due to replay buffer shift. This is the core tension in position-based curriculum learning.

---

## 6. Curriculum Design

Three weights are annealed concurrently over training:

### 6.1 Centre-Line Weight

Starts at 1.0 (tight centre-line enforcement) and anneals to 0.0 over 1,500 episodes, allowing the policy to learn the racing line.

```python
centre_weight = max(0.0, 1.0 - episode / 1500)
reward -= effective_cw * 0.4 * abs(track_pos)
```

After 2500m, the weight is hard-floored at 0.8 to counteract a specific right-side drift observed at the post-S-corner bend (2500–2600m). This is track-specific and brittle — it will likely need revising if the spawn position changes.

### 6.2 Heading Weight

Annealed to zero over 500 episodes. The heading bonus rewards pointing forward:

```python
heading_bonus = heading_weight * 0.5 * np.cos(angle)
```

Keeping this active long-term creates a perverse incentive: the policy maximises heading reward by driving slowly in a straight line, as high speed introduces lateral forces that increase the heading angle. Removing it early forces the agent to find that speed itself is the signal.

### 6.3 Progress Weight (Completion-Rate Driven)

This weight scales the speed component of the reward. Rather than annealing on a fixed schedule, it's driven by a rolling completion rate EMA:

```python
PHASE1_W = 0.5    # floor — survival focus
PHASE2_W = 1.0    # ceiling — speed focus
alpha    = 0.05   # EMA smoothing

completion_rate = alpha * completed + (1 - alpha) * completion_rate
progress_weight = PHASE1_W + (PHASE2_W - PHASE1_W) * min(completion_rate / 0.7, 1.0)
```

The policy doesn't get pushed toward speed until it's completing >70% of episodes. This prevents the speed reward from destabilising a policy that hasn't yet learned to stay on the track.

---

## 7. Monitoring: Alpha as a Signal

SAC's entropy coefficient alpha is one of the more informative training signals to watch. A rising alpha means the policy is encountering states where its Q-estimates have high variance — it's hedging by staying exploratory.

```
Run 20260609_123328 alpha progression:
  ep 1–50:    alpha ≈ 0.35
  ep 50–100:  alpha ≈ 0.40
  ep 100–121: alpha ≈ 0.45
```

During the first attempt, a rising alpha would have prompted hyperparameter intervention. This time, knowing that the alpha rise coincided with the policy starting to encounter the S-corner regularly (due to the 1600m spawn), the correct interpretation was: "the agent is in new territory, let it explore." The alpha stabilised once the Q-estimates for those states matured.

---

## 8. Results

### 8.1 Training Run (1600m Spawn)

```
Total episodes:  121
Completions:     12   (9.9% completion rate)
Best lap time:   1:34
Alpha range:     0.35 → 0.45

Section speeds (km/h):
  0–500m:     142
  500–1000m:  138
  1000–1500m: 128   (approach to S-corner)
  1500–2000m: 119
  2000–2500m: 122   (S-corner clearance)
  2500–3000m: 118
  3000–3500m: 113   ← slowest section
  3500–3598m: 126   (final straight)
```

The 3000–3500m section at 113 km/h is the remaining bottleneck. It contains a sequence of medium-speed corners where the current policy brakes earlier than necessary.

### 8.2 Comparison With First Attempt

| Metric | First attempt | This attempt |
|--------|---------------|--------------|
| Algorithm | SAC (SB3) | SAC (self-written) |
| Training episodes | 4,349 | 121 |
| Completion rate | 0.85% | 9.9% |
| Best lap time | 1:48 | **1:34** |
| S-corner strategy | Patched post-failure | Spawn-targeted from the start |
| Milestone bonuses | Yes (caused exploit) | None |

The episode count comparison is not apples-to-apples — the 1600m spawn simplifies the task by removing the pre-S-corner section from every episode. But the lap time improvement is on shared track and reflects a genuinely faster policy.

---

## 9. What Didn't Work

### 9.1 Buffer Focus Sampling

At one point, a replay buffer filter was added: always store transitions from `distRaced > 2200m`, discard 70% of earlier transitions. The idea was to force more S-corner gradient signal.

This was removed after training loss became unstable. The issue: the Q-networks are trained on a distribution that's heavily weighted toward late-episode states, but the policy is evaluated on the full episode from 1600m. When early-episode states appear during evaluation, the Q-estimates are poorly calibrated. The filter created a training/evaluation distributional gap that undermined learning.

### 9.2 distFromStart vs distRaced

An early version of the 2500m penalty used `distFromStart` rather than `distRaced`. `distFromStart` resets to 0 at the finish line; `distRaced` is cumulative. On a 3598m track with a 1600m spawn, a car that completes a lap and continues into a second lap will have `distFromStart` pass through 2500m again — triggering the penalty at the wrong physical location.

```python
# Wrong: resets at finish line
dist = float(obs.get('distFromStart', 0.0))

# Correct: always cumulative
dist = float(obs.get('distRaced', 0.0))
```

---

## 10. Lessons Revisited

The first attempt produced a methodology: train → observe failure → analyse data → form hypothesis → implement solution → validate → repeat. That process still applies, but this round added one refinement:

**Build the reward function before you need to.** The first attempt's reward function was shaped entirely by failures. This time, knowing the exploit classes in advance (flat bonuses create local optima, stuck grace periods get gamed, milestone bonuses cause parking) allowed several of them to be avoided rather than discovered and patched.

Some still appeared anyway — the 2500m bonus exploit was added and caught mid-training. The difference is that catching it took one episode of telemetry review rather than 200 episodes of confused debugging.

The 3000–3500m section remains unresolved. The next step is to raise `progress_weight` gradually — currently at 0.5 — to push the speed reward higher and incentivise faster cornering in that section, while monitoring alpha for signs of destabilisation.

---

## Appendix: Full Reward Function

```python
# Locomotion (core signal)
track_progress = sp_x * np.cos(angle) - sp_y * np.sin(angle)
track_progress = max(track_progress, 0.0)
reward = progress_weight * track_progress

# Heading alignment (annealed to 0 over 500 episodes)
reward += heading_weight * 0.5 * np.cos(angle)

# Survival bonus
reward += 1.0

# Centre-line penalty (curriculum annealed; hard floor ≥0.8 after 2500m)
effective_cw = max(centre_weight, 0.8) if dist_raced > 2500 else centre_weight
reward -= effective_cw * 0.4 * abs(track_pos)

# Edge proximity warning
if abs(track_pos) > 0.8:
    reward -= min(10.0 * (abs(track_pos) - 0.8), 5.0)
if track.min() < 2.0:
    reward -= min((2.0 - track.min()) * 3.0, 5.0)

# Post-S-corner right-drift penalty (2500–2600m)
if 2500 < dist_raced < 2600 and track_pos > 0.3:
    reward -= (track_pos - 0.3) * 5.0

# Damage
if damage_delta > 0:
    reward -= 5.0 + damage_delta * 0.1

# Low-speed penalty (skip launch phase)
if time_step > 100 and sp_x < 40:
    reward -= progress_weight * (40 - sp_x) * 0.5

# Corner approach
if front_dist < 70 and sp_x > 75:
    reward -= min((sp_x - 75) * 0.08, 2.0)
if 50 < front_dist < 120 and 40 < sp_x < 75:
    reward += 2.0

# Exploration bonus (new max distFromStart)
if dist_from_start > max_dist:
    reward += (dist_from_start - max_dist) * 0.02

# Lap completion
reward += 100.0
reward += progress_weight * max(0, 240 - last_lap)

# Termination penalties
if abs(track_pos) > 1.3:   reward -= 20.0; done = True
if slow_steps >= 150:       reward -= 20.0; done = True
if cos(angle) < -0.3:      reward -= 20.0; done = True
```

---

- **Code**: [github.com/choeyunbeom/geek_pit_crew](https://github.com/choeyunbeom)
- **Part 1**: [TORCS Corkscrew Challenge: A Journey Through Reinforcement Learning Failures and Breakthroughs](https://choeyunbeom.github.io/reinforcement%20learning/autonomous%20driving/2026/01/28/torcs-rl-journey.html)
- **Date**: June 24, 2026
