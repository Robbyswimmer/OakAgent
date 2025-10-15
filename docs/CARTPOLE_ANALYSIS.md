# OaK-CartPole Architecture Deep Dive & Continual Learning Analysis

**Date**: 2025-10-13
**Purpose**: Comprehensive validation of learning cycle and continual learning capabilities

---

## Executive Summary

**Architecture Status**: ✅ **Logically Sound**
**Continual Learning Readiness**: ⚠️ **Significant Issues Identified**

The OaK architecture implements a sophisticated hierarchical RL system with predictive knowledge. All components are correctly implemented and follow OaK principles. However, several fundamental issues prevent effective continual learning across non-stationary regimes.

---

## Part 1: Architecture Validation

### 1.1 Experience Collection & Replay ✅

**Implementation** (main.py:190-250, replay.py:10-108):
- FIFO circular buffer (50K capacity for real experience)
- Separate buffers for real (rb_real) and simulated (rb_sim) experience
- Option trajectories stored with SMDP quantities (R_total, duration)
- State/action history tracked for FC-STOMP

**Validation**: ✅ **Correct**
- Proper FIFO semantics with wraparound
- SMDP trajectory computation is accurate
- Recent state sampling works for Dyna planning

**Continual Learning Issue** 🔴:
- **Catastrophic forgetting**: 50K capacity ≈ 167 episodes at 300 steps/ep
- When regime shifts at episode 200, only ~17 episodes of old regime remain
- No mechanism to identify or preserve "important" transitions
- No episodic memory for long-term retention

---

### 1.2 Dynamics Model Learning ✅

**Implementation** (models/dyn_model.py:80-262):
- Ensemble of 3 MLPs predicting (Δs, r) from (s, a)
- Supervised learning with MSE loss
- Meta-learning (IDBD) for adaptive step sizes
- 20 minibatch updates per environment step

**Validation**: ✅ **Correct**
- Ensemble provides uncertainty estimates
- Delta prediction (Δs = s' - s) is more stable than direct s' prediction
- Xavier initialization with small gain prevents explosion
- Gradient clipping prevents instability

**Continual Learning Issue** 🟡:
- Model trained on ALL data in replay buffer uniformly
- No distribution shift detection
- When regime shifts, model must "unlearn" old dynamics gradually
- **Prediction error will spike** after regime transition
- Planning quality degrades until model adapts (~100-200 steps)

---

### 1.3 Q-Learning (Primitive & Option) ✅

**Implementation**:
- **Primitive Q** (models/q_primitive.py:47-206): Double Q-learning with target network
- **Option Q** (models/q_option.py:46-200): SMDP Q-learning with gamma^τ discounting

**Validation**: ✅ **Correct**
- Double Q-learning reduces overestimation bias
- Target network synced every 500 steps
- SMDP discounting properly accounts for option duration
- IDBD meta-learning adapts step sizes

**Continual Learning Issues** 🔴:
- **Q-value staleness**: Q(s,a) learned for old dynamics become incorrect after shift
- No mechanism to detect or reset stale Q-values
- Agent makes poor decisions based on outdated value estimates
- **Bootstrap from wrong values**: TD learning propagates errors
- Target network further delays adaptation (500 step lag)

---

### 1.4 GVF Learning & Prediction ✅

**Implementation** (knowledge/gvf.py:19-200):
- 4 GVFs predicting cumulants: uprightness, centering, stability, survival
- TD learning with eligibility traces
- Normalized predictions for feature mining
- Meta-learning (IDBD) for adaptive step sizes

**Validation**: ✅ **Correct**
- GVF predictions properly normalized to [0, 1/(1-γ)]
- Cumulant definitions match OaK principles
- TD error computation is correct
- Small weight initialization prevents explosion

**Continual Learning Issues** 🟡:
- **Non-stationary policy**: GVFs predict under current policy, but policy changes continuously
- **Non-stationary dynamics**: When regime shifts, predictions become incorrect
- **Feature mining dependency**: FC-STOMP uses GVF variance to identify features
  - Variance reflects policy changes + dynamics changes (confounded)
  - May identify spurious features after regime shift
- **No mechanism to re-calibrate** GVF predictions after shift

---

### 1.5 Option Policy & Model Learning ✅

**Implementation**:
- **Policy** (options/option.py:137-229): Actor-critic with pseudo-rewards
- **Model** (models/option_model.py:103-262): SMDP model predicting (Δs, R, τ)

**Validation**: ✅ **Correct** (after pseudo-reward fix)
- Actor-critic uses TD error as advantage estimate
- Pseudo-reward computed from START state (avoids double-counting)
- Option model uses normalized state space for stability
- Tanh activation bounds delta predictions

**Continual Learning Issues** 🔴:
- **Policy brittleness**: Option policies optimized for old dynamics fail in new regime
- **Model inaccuracy**: Option models predict wrong outcomes after shift
- **Slow pruning**: Options need 15 executions + 2500 step age before pruning
  - During regime shift, bad options persist for ~83 episodes (2500 / 30 steps/ep)
  - Agent suffers from executing failing options
- **Success rate lag**: Success rate computed over ALL executions (cumulative)
  - Option with 90% success in R1 might have 10% in R2
  - Cumulative: (180 successes / 210 executions) = 85.7% → won't prune!

---

### 1.6 FC-STOMP Cycle ✅

**Implementation** (knowledge/feature_construct.py:594-750):
1. **Feature Mining**: Identify GVFs with high variance + low error
2. **Controllability**: Test if feature is actionable via model-based lookahead
3. **Subtask Formation**: Create goal specification from feature
4. **Option Creation**: Instantiate option with policy/model
5. **Pruning**: Remove options with <5% success rate

**Validation**: ✅ **Correct**
- Feature variance threshold (0.02) appropriately relaxed for early training
- Controllability check uses H=3 lookahead with dynamics model
- Bootstrap mode (no options) uses relaxed controllability (0.05 vs 0.08)
- Pruning has minimum age (2500 steps) to avoid premature removal

**Continual Learning Issues** 🟡:
- **Cooldown delays**: 300 step cooldown between feature spawns
- **Pruning lag**: 800 episode window + minimum age delays adaptation
- **Controllability confusion**: After regime shift, controllability scores are computed with stale dynamics model
  - May think features are controllable when they're not (or vice versa)
- **No regime-aware reset**: No mechanism to clear options when shift detected

---

### 1.7 Dyna Planning ✅

**Implementation** (planner.py:8-120):
- Sample recent states (last 100)
- Simulate H=11 step rollouts using dynamics model
- Mix primitive actions (70%) and options (30%)
- Update Q-function directly from simulated experience (OAK_PURITY_MODE)

**Validation**: ✅ **Correct**
- On-demand generation avoids storing stale simulated experience
- Horizon (11) matches config
- Recent state sampling focuses planning on relevant regions

**Continual Learning Issues** 🟡:
- **Model-based planning with wrong model**: After regime shift, dynamics model is inaccurate
- **Planning propagates errors**: Simulated experience has wrong (s, a, r, s') tuples
- **Q-learning from bad data**: Updates Q-function with incorrect targets
- Could actually **harm learning** immediately after shift (60 imagined transitions per step!)

---

## Part 2: Continual Learning Assessment

### Critical Issue #1: Catastrophic Forgetting 🔴

**Problem**: FIFO replay buffer overwrites old regime data quickly.

**Math**:
- Buffer capacity: 50,000 transitions
- Late-training episodes: ~300 steps/episode
- Buffer coverage: 50,000 / 300 = **167 episodes**
- Regime duration: 200 episodes
- After 33 episodes in new regime, **all old data is gone**

**Impact**:
- Backward transfer impossible (returning to R1 after R2)
- Agent must relearn from scratch if regime returns
- No long-term memory of past regimes

**Potential Solutions**:
1. **Reservoir sampling**: Preserve some old-regime data
2. **Episodic memory**: Store full episodes, sample uniformly
3. **Experience replay with priorities**: Keep high-TD-error transitions
4. **Separate regime buffers**: Detect regime shifts, maintain per-regime buffer

---

### Critical Issue #2: Epsilon Decay Problem 🔴

**Problem**: Epsilon decays monotonically; no re-exploration after regime shift.

**Math**:
- EPSILON_START = 1.0, EPSILON_DECAY = 0.995, EPSILON_END = 0.01
- After 200 episodes: ε = 1.0 × 0.995^200 = **0.368**
- After 400 episodes: ε = **0.135**
- At regime shift (R1→R2 at ep 200), ε = 0.368 (not terrible)
- At next shift (R2→R3 at ep 400), ε = 0.135 (too low for exploration!)

**Impact**:
- Agent exploits learned policy even when it's wrong
- Insufficient exploration to discover new optimal actions
- Later regimes suffer more than earlier ones

**Potential Solutions**:
1. **Epsilon boost on shift detection**: Increase ε to 0.5 when shift detected
2. **Per-regime epsilon**: Reset ε when regime changes
3. **Uncertainty-driven exploration**: Explore more when model uncertainty is high
4. **Optimistic initialization**: Reset Q-values to encourage exploration

---

### Critical Issue #3: No Distribution Shift Detection 🔴

**Problem**: Agent has no awareness of regime changes.

**Current Behavior**:
- All components learn incrementally/gradually
- No special "adaptation mode" triggered
- Agent treats regime shift as gradual drift

**Impact**:
- Slow adaptation (wastes ~50-100 episodes per regime)
- Harmful decisions based on stale models
- Missed opportunity for targeted adaptation

**Potential Solutions**:
1. **Dynamics model error monitoring**: Spike in prediction error → shift detected
2. **GVF prediction error monitoring**: Sudden increase → shift detected
3. **Performance monitoring**: Return drops by >20% → trigger adaptation
4. **Explicit regime signal**: Use ground-truth regime transitions (for research)

**Detection-Based Adaptations**:
- Boost epsilon for exploration
- Clear or reset option library
- Increase learning rates temporarily
- Sample more uniformly from buffer (vs recent states)

---

### Critical Issue #4: Option Success Rate Accumulation 🟡

**Problem**: Success rate computed cumulatively, not per-regime.

**Example Scenario**:
- Option achieves 90% success in R1 over 200 executions (180 successes)
- Regime shifts to R2, option now has 10% success (20 executions, 2 successes)
- Cumulative success: (180 + 2) / (200 + 20) = **82.7%**
- Pruning threshold: 5%
- **Option won't be pruned despite being harmful!**

**Impact**:
- Bad options persist across regime boundaries
- Agent wastes actions executing failing options
- Delays convergence in new regime

**Potential Solutions**:
1. **Windowed success rate**: Only count last W executions (e.g., W=50)
2. **Exponential decay**: Weight recent executions more heavily
3. **Regime-aware reset**: Clear option statistics on shift detection
4. **Rapid pruning mode**: Lower threshold/age requirement after shift

---

### Critical Issue #5: Planning with Stale Models 🟡

**Problem**: Dyna planning uses 60 imagined transitions per step with wrong model.

**Timeline After Regime Shift**:
- t=0: Regime shifts, dynamics model is accurate for old regime
- t=0-100: Model prediction error is high, but planning continues
- t=100-200: Model gradually adapts, but still learning
- t=200+: Model reasonably accurate

**Impact**:
- Early planning (t=0-100) actively **harms** learning
  - Q-function updated with wrong (s, a, r, s') tuples
  - Bad Q-values propagate via bootstrapping
- Could be better to **disable planning** temporarily after shift

**Potential Solutions**:
1. **Model error gating**: Disable planning when model error > threshold
2. **Conservative planning**: Use ensemble disagreement to filter bad simulations
3. **Adaptive plan horizon**: Reduce H when model is uncertain
4. **Real-only learning**: Temporarily learn only from real experience after shift

---

### Moderate Issue #6: GVF Non-Stationarity 🟡

**Problem**: GVFs predict under non-stationary policy and dynamics.

**Sources of Non-Stationarity**:
1. **Policy improvement**: As Q-function updates, policy changes
2. **Option creation**: New options change action space
3. **Regime shifts**: Environment dynamics change

**Impact**:
- GVF variance reflects both dynamics changes AND policy changes
- Feature mining may identify spurious features
- Controllability scores are confounded

**Potential Solutions**:
1. **Off-policy GVF learning**: Decouple GVF learning from behavior policy
2. **Importance sampling**: Correct for policy changes
3. **Separate evaluation policy**: Use fixed policy for GVF evaluation
4. **Regime-aware GVFs**: Detect shift, re-initialize GVFs

---

### Moderate Issue #7: Q-Value Staleness 🟡

**Problem**: Q-values learned for old dynamics are incorrect after shift.

**Example**:
- R1: Q(s, left) = 100 (optimal)
- R2 (heavy pole): Q(s, left) = 50 (suboptimal)
- But Q-function still has Q(s, left) ≈ 100 from R1
- Agent overestimates value of "left" action

**Impact**:
- Exploration-exploitation tradeoff is distorted
- Agent exploits suboptimal actions
- Bootstrapping propagates errors slowly

**Potential Solutions**:
1. **Q-value reset**: Reset Q-network on shift detection
2. **Conservative Q-learning**: Lower Q-values when uncertain
3. **Ensemble Q-networks**: Maintain multiple Q-functions, detect disagreement
4. **Optimistic initialization**: Reset to high values to encourage exploration

---

## Part 3: Recommendations for Continual Learning

### High Priority (Blocking Issues)

1. **✅ Pseudo-Reward Fix** [COMPLETED]
   - Changed from t[3] (next state) to t[0] (current state)
   - Prevents double-counting in value bootstrapping

2. **🔴 Epsilon Boost on Regime Transition**
   ```python
   # In train_continual() when regime changes:
   if regime_changed:
       self.epsilon = max(self.epsilon, 0.5)  # Boost exploration
   ```

3. **🔴 Windowed Option Success Rate**
   ```python
   # In option.py: Track recent successes separately
   self.recent_executions = deque(maxlen=50)
   self.recent_successes = deque(maxlen=50)

   def get_recent_success_rate(self):
       if len(self.recent_executions) == 0:
           return 0.0
       return sum(self.recent_successes) / len(self.recent_executions)
   ```

4. **🟡 Model Error Gating for Planning**
   ```python
   # In _update_all_components(): Check model error before planning
   if self.dyn_model.get_average_error() < 0.5:  # threshold
       sim_transitions = self.planner.imagine_transitions(self.rb_real)
       # ... update Q from simulations
   ```

### Medium Priority (Performance Improvements)

5. **🟡 Experience Replay with Reservoir Sampling**
   - Preserve 10% of buffer for old-regime data
   - Sample uniformly from all regimes, not just recent

6. **🟡 Distribution Shift Detection**
   - Monitor dynamics model prediction error
   - Trigger adaptation mode when error spikes

7. **🟡 Regime-Aware FC-STOMP**
   - Clear option library on shift detection
   - Reduce pruning age requirement after shift

8. **🟡 Conservative Q-Learning After Shift**
   - Use min over ensemble for conservative estimates
   - Reduces overestimation from stale Q-values

### Low Priority (Future Work)

9. **Episodic Memory System**
   - Store full episodes with regime metadata
   - Sample episodes uniformly across regimes

10. **Multi-Head Networks**
    - Separate Q-networks per regime
    - Detect regime via forward pass ensemble

11. **Meta-Learning Regime Adaptation**
    - Learn a "regime adaptation" policy
    - Optimize for rapid adaptation speed

---

## Part 4: Validation Summary

### What's Working ✅

1. **Core Architecture**: All components correctly implement OaK principles
2. **Experience Collection**: FIFO buffer, SMDP trajectories, state history
3. **Dynamics Model**: Ensemble with uncertainty, meta-learning, gradient clipping
4. **Q-Learning**: Double Q, SMDP discounting, target networks
5. **GVFs**: Normalized predictions, TD learning, feature mining
6. **Option Learning**: Actor-critic, pseudo-rewards (FIXED), option models
7. **FC-STOMP**: Feature → subtask → option → model → planning pipeline
8. **Planning**: Dyna with on-demand generation, OAK purity mode

### What's Broken for Continual Learning 🔴

1. **Catastrophic Forgetting**: FIFO buffer overwrites old regime data
2. **Epsilon Decay**: No re-exploration after regime shifts
3. **No Shift Detection**: Agent unaware of distribution changes
4. **Option Success Accumulation**: Cumulative stats mask regime-specific failures
5. **Planning with Stale Models**: Harmful simulated experience early after shift
6. **GVF Non-Stationarity**: Predictions confounded by policy + dynamics changes
7. **Q-Value Staleness**: Outdated value estimates persist across regimes

### Overall Assessment

**Single-Regime Learning**: ✅ **Ready to Test**
**Continual Learning**: ⚠️ **Needs Fixes (esp. #2, #3, #4)**

The architecture is sound for learning a single task. However, continual learning across regime shifts requires at minimum:
1. Epsilon boost on transitions
2. Windowed option success rate
3. Model error gating for planning

Without these fixes, the agent will:
- Exploit stale policies (low epsilon)
- Execute failing options (cumulative success rate)
- Learn from wrong simulations (bad planning)

**Estimated performance without fixes**: ~30-40% worse than single-regime learning.
**Estimated performance with fixes**: Within 10-20% of single-regime learning.

---

## Part 5: Testing Recommendations

### Test 1: Baseline (Single Regime)
- Run for 500 episodes in R1_base
- Verify solving (>475 return over 100 episodes)
- Establish performance ceiling

### Test 2: Two-Regime (R1 → R2)
- Run TwoRegimeConfig (100 episodes each)
- Measure adaptation time in R2
- Check option creation/pruning behavior

### Test 3: Full Five-Regime
- Run ContinualConfig (200 episodes per regime)
- Track performance across all transitions
- Measure forward transfer (does R1 help R2?)

### Test 4: Backward Transfer (R1 → R2 → R1)
- Run BackAndForthConfig
- Critical test for catastrophic forgetting
- Expect poor performance on return to R1 (buffer overwriting)

### Metrics to Track
1. **Adaptation time**: Episodes until solved in new regime
2. **Forward transfer**: Does R1 experience help R2?
3. **Backward transfer**: Performance when returning to R1
4. **Option churn**: Creation/deletion rate across transitions
5. **Model error**: Prediction error immediately after shift

---

## Conclusion

The OaK-CartPole architecture is **logically sound and correctly implemented**. All learning components follow established RL principles and OaK design philosophy.

However, the system is **not ready for continual learning** without modifications. The three critical issues (epsilon decay, option success accumulation, planning with stale models) will significantly degrade performance across regime shifts.

**Recommended next steps**:
1. Apply high-priority fixes (#2, #3, #4)
2. Run two-regime test to validate improvements
3. Add distribution shift detection for automatic adaptation
4. Consider episodic memory for backward transfer

The codebase is clean, well-documented, and modular. Adding continual learning capabilities is feasible with targeted modifications.
