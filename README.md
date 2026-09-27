# cfar-bench
A Rust library for CFAR detection and pulse compression, validated against closed-form Pfa/Pd theory via Monte Carlo simulation.

Everything in here is automatically tested: a Monte Carlo measurement, a confidence interval with a stated error rate. An assertion, that the theoretical value falls inside it.

Scope: cell-averaging (CA) and ordered-statistic (OS) CFAR single pulse detection of Swerling I targets in complex Gaussian noise, including interfering targets and clutter edges.

## Results

CA-CFAR false alarm rate matches theory

| N \ Pfa | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| 8  | 1.021 | 1.037 | 0.990 |
| 16 | 0.922 | 0.943 | 1.043 |
| 32 | 0.955 | 1.012 | 0.987 |
 
The Pfa is also unchanged across six decades of noise power. That is the "constant false alarm rate" property, tested directly.

CFAR loss vs window size (Swerling I, Pd = 0.9; extra SNR CA-CFAR needs compared with an ideal detector that knows the noise power exactly):

| N | Pfa 1e-4: theory | measured | Pfa 1e-6: theory | measured |
|---|---|---|---|---|
| 4  | 5.90 dB | 5.91 ± 0.01 | 9.45 dB | 9.44 ± 0.02 |
| 8  | 2.73 dB | 2.73 ± 0.01 | 4.27 dB | 4.26 ± 0.01 |
| 16 | 1.31 dB | 1.30 ± 0.01 | 2.01 dB | 2.01 ± 0.01 |
| 32 | 0.64 dB | 0.63 ± 0.01 | 0.97 dB | 0.98 ± 0.01 |
| 64 | 0.32 dB | 0.32 ± 0.01 | 0.48 dB | 0.47 ± 0.01 |


The loss roughly halves every time N doubles. A larger window gives a better noise estimate.
 
**Interfering targets** (N = 16, OS rank k = 12, target and interferers at 20 dB):
 
| Interferers in window | CA-CFAR Pd | OS-CFAR Pd |
|---|---|---|
| 0 | 0.917 | 0.908 |
| 1 | 0.600 | 0.898 |
| 2 | 0.391 | 0.880 |
| 4 | 0.168 | 0.806 |
| 5 | 0.109 | **0.386** |

CA-CFAR loses most of its detections to a single interferer. OS-CFAR holds up to exactly N − k = 4 interferers, then breaks down, as the theory predicts.

Clutter edge (20 dB step): CA-CFAR's false alarm rate spikes to 31× the design value at the first clutter cell (measured 3.06e-2, theory 3.03e-2). OS-CFAR masks far less before the edge but spikes higher at it (46×).

## Quick start

```bash
cargo test --release              # full validation suite (~20 s)
cargo run --release --example cfar_loss            # CFAR loss table
cargo run --release --example interfering_targets  # CA vs OS with interferers
cargo run --release --example clutter_edge         # false alarms across a clutter edge
cargo run --release --example correlated_profile   # correlation along a profile
```

All randomness uses `ChaCha8Rng` with fixed seeds. Every run reproduces the same numbers.

## Repository layout 

```
src/
├── sim/ Noise, Swerling I targets, single-cell trials, range-profile scenes
├── cfar/ CaCfar, OsCfar, FixedThreshold (known-noise benchmark), window + edge policy
├── theory/ Closed-form answer key: CA-CFAR, OS-CFAR, known-noise detector, exponential law
├── stats/ Clopper-Pearson & Wilson intervals, Bonferroni families, trial planning, KS test
├── special/ ln Γ, regularized incomplete beta, normal quantile (checked against SciPy)
└── units.rs    dB conversions
tests/          Statistical validation suites (one Bonferroni family per file)
examples/       Report experiments that print tables
```

A design rule runs through the repo. `theory` never calls into `cfar` or `sim`. The code under test and answer key share no logic, a bug cannot hide in the other.

For the same reason, the detectors take a threshold multiplier α, not a target Pfa. Converting Pfa to α is a theory question, and if that conversion were wrong, the Monte Carlo would show it.

## The mathematics

### 1. Noise model

Receiver noise at complex baseband is circularly-symmetric complex Gaussian, $x \sim \mathcal{CN}(0, P)$: I and Q are independent with power $P/2$ each. After square-law detection, $|x|^2$ is the sum of two squared Gaussians, which makes it **exponential with mean $P**. Every CFAR result below depends on this, so it is validated first: power, I/Q independence, a KS test and the first four moments.

### 2. CA-CFAR and its false alarm probability

The detector estimates the noise from $N$ reference cells around the cell under test (CUT), skipping guard cells, and declares a detection when
 
$$\text{CUT} > \alpha \cdot \frac{1}{N}\sum_{i=1}^{N} X_i .$$
 
With noise only, the CUT is $\text{Exp}(P)$ and the reference sum $Z$ is $\text{Gamma}(N, P)$. Conditioning on $Z$ and averaging, which is exactly the moment-generating function of the Gamma distribution, gives
 
$$P_{fa} = \left(1 + \frac{\alpha}{N}\right)^{-N}.$$
 
The noise power $P$ cancels, so the false alarm rate does not depend on how strong the noise is. That is what "constant false alarm rate" means. Inverting gives the multiplier for a design Pfa: $\alpha = N\left(P_{fa}^{-1/N} - 1\right)$.



### 3. Swerling I targets, Pd, and CFAR loss
 
A Swerling I target has a complex Gaussian echo, so target plus noise in the CUT is still exponential, now with mean $P(1+S)$ at SNR $S$. The same derivation gives
 
$$P_d = \left(1 + \frac{\alpha}{N(1+S)}\right)^{-N}.$$
 
The benchmark is a detector that knows $P$ exactly and uses a fixed threshold: $P_d = P_{fa}^{1/(1+S)}$. CA-CFAR has to *estimate* the noise, and the estimate is noisy, so it needs more SNR to reach the same Pd. That extra SNR is the **CFAR loss**. It shrinks roughly like $1/N$ (a large-N expansion of the formulas above, also checked in the tests).
 
### 4. OS-CFAR
 
OS-CFAR estimates the noise from the **k-th smallest** reference cell $X_{(k)}$ instead of the mean. The spacings between exponential order statistics are independent (the Rényi representation), which gives a product formula:
 
$$P_{fa} = \prod_{i=0}^{k-1} \frac{N-i}{N-i+\alpha}, \qquad P_d = \text{same with } \alpha \to \frac{\alpha}{1+S}.$$
 
There is no closed-form inverse, so α is found by bisection; Pfa is strictly decreasing in α, so the root is unique. The formula is checked three independent ways in unit tests: against a gamma-function form, against numerical integration over the order-statistic density, and against the $k=1$ special case.
 
**Why it is robust:** up to $N-k$ reference cells can be arbitrarily large (other targets) and the k-th smallest is still a noise cell. **The price:** in clean noise the k-th value is a worse noise estimate than the mean, so OS-CFAR has 0.3–1.9 dB more loss than CA-CFAR, shrinking as N grows.
 
### 5. Non-uniform backgrounds
 
The CA-CFAR derivation never needed the cells to have equal power. If the CUT has mean power $p_c$ and reference cell $i$ has mean power $p_i$, all independent and exponential:
 
$$P(\text{detect}) = \prod_{i=1}^{N} \left(1 + \frac{\alpha\, p_i}{N\, p_c}\right)^{-1}.$$
 
This single formula covers interfering targets (some $p_i$ large) and clutter edges (a step in the $p_i$). That is why CA-CFAR's behaviour in both scenarios is predicted exactly, not just measured.

## How the validation works

**Independent trials**. Each Monte Carlo trial draws a fresh CUT and fresh reference cells. The detection count is exactly binomial. Sliding a detector a long one long profile would make neighbouring decisions correlated.

**Exact confidence intervals (Clopper–Pearson).** The textbook $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ interval breaks down at small probabilities: at Pfa = 1e-4 you observe a handful of counts from a very skewed distribution. Clopper–Pearson inverts exact binomial tests and **guarantees** coverage ≥ 1 − α for every true p. The tests prove that property by computing coverage exactly, not by simulation.
 
**Multiple comparisons (Bonferroni)**. If you check 20 points at 99% confidence each, the chance that some checks will fail by luck is 18%. Each test file is one family with a family-wise error rate of 1e-3, and each assertion runs at 10^-3/m. The threhsolds are derived from the number of assertions, not chosen by hand.

**Trial planning.** The number of trials needed to measure a probability $p$ to relative precision $r$ is about $z^2 (1-p)/(p\,r^2)$. At Pfa = 1e-4 that is millions of trials, which is why the planner exists.


**Negative controls.** Each suite includes realistic bugs that must be *rejected*:
 
- the known-noise multiplier used in CA-CFAR
- envelope $|x|$ fed in instead of power $|x|^2$
- SNR converted with $10^{dB/20}$ instead of $10^{dB/10}$
- a non-fluctuating target instead of Swerling I
- the CA-CFAR multiplier reused in OS-CFAR
- an off-by-one OS rank
- real-valued noise, and a 1 % noise-power miscalibration

**Calibration of the machinery itself.** Under a correct null hypothesis, p-values should be uniform. Over 10,000 seeds, the KS test rejected at 0.11 %, 1.12 % and 5.36 % for α = 0.1 %, 1 % and 5 %: the validation tools are themselves calibrated.


## Limitations
 
- **Single pulse, Swerling I only.** Non-fluctuating (Swerling 0) targets need the Marcum Q function and are not implemented; they appear only as a negative control.
- **White noise only.** Real processing chains (matched filtering, oversampling) correlate neighbouring cells; the effect on CFAR is discussed above but not measured.
- **OS-CFAR in non-uniform backgrounds is measured, not predicted.** Only a large-interferer approximation is compared.
- **Numerical accuracy.** The incomplete beta function is accurate to about 1e-8 relative at the largest arguments used (~1e6), which is far tighter than any Monte Carlo interval here.
- **Reproducibility.** Seeds make results reproducible to the printed digits across platforms. Bit-for-bit identity is not guaranteed, because `ln`/`exp` come from each platform's maths library.

## References
 
- M. A. Richards, *Fundamentals of Radar Signal Processing*, McGraw-Hill.
- S. M. Kay, *Fundamentals of Statistical Signal Processing, Vol. II: Detection Theory*, Prentice Hall.
- H. Rohling, "Radar CFAR thresholding in clutter and multiple target situations", *IEEE Trans. Aerospace and Electronic Systems*, 1983.
- P. P. Gandhi and S. A. Kassam, "Analysis of CFAR processors in nonhomogeneous background", *IEEE Trans. Aerospace and Electronic Systems*, 1988.
- C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial", *Biometrika*, 1934.
- L. D. Brown, T. T. Cai and A. DasGupta, "Interval estimation for a binomial proportion", *Statistical Science*, 2001.
- M. J. Wichura, "Algorithm AS 241: The percentage points of the normal distribution", *Applied Statistics*, 1988.