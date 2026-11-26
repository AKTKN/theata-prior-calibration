# Recovering mechanism priors from detector moments

This derivation indexes every error mechanism by a general index e. It does
not require a spatial classification of mechanisms. It explains the current
graphlike estimator as a special case of a real linear inverse problem.
The general matrix inversion described here is a mathematical extension;
the executable estimator continues to use its validated graphlike formulas.

## 1. Known supports and unknown priors

Let M be the number of observed detectors and E the number of candidate error
mechanisms. Their known detector incidence matrix is

$$
H=[h_1,\ldots,h_E]\in\{0,1\}^{M\times E},
\qquad H_{ie}=1\ \Longleftrightarrow\ i\in S_e,
$$

where S_e is the set of detectors flipped by mechanism e. For one shot, let

$$
F=(F_1,\ldots,F_E)^\top\in\mathbb F_2^E,
\qquad F_e\sim\operatorname{Bernoulli}(q_e),
\qquad x=HF\pmod 2.
$$

The components F_e are mutually independent. The unknown prior vector is
q in [0,1/2)^E. The observed detector vector x has M components. The restriction
q_e<1/2 fixes the positive-bias branch used below. Logical labels do not enter
these detector-moment identities.

Define the population moments

$$
\mu_i=\mathbb E[x_i],
\qquad \mu_{ij}=\mathbb E[x_ix_j].
$$

The symbol q_e denotes an individual mechanism prior. The symbol p_ij below
denotes the pair quantity obtained from the paper's equation S11. For general
supports, these are different indexed objects.

## 2. From XOR observations to products

For one Bernoulli variable,

$$
\mathbb E[(-1)^{F_e}]=(1-q_e)-q_e=1-2q_e.
$$

Because an XOR is a sum modulo two,

$$
(-1)^{x_i}=\prod_{e=1}^E(-1)^{H_{ie}F_e}.
$$

Independence therefore gives the exact identity

$$
b_i:=1-2\mu_i
=\mathbb E[(-1)^{x_i}]
=\prod_{e=1}^E(1-2q_e)^{H_{ie}}.
\tag{1}
$$

Similarly,

$$
\begin{aligned}
b_{ij}
&:=1-2\mu_i-2\mu_j+4\mu_{ij}\\
&=\mathbb E[(1-2x_i)(1-2x_j)]\\
&=\mathbb E[(-1)^{x_i\oplus x_j}]\\
&=\prod_{e=1}^E(1-2q_e)^{H_{ie}\oplus H_{je}}.
\end{aligned}
\tag{2}
$$

A mechanism shared by detectors i and j cancels from their XOR; a mechanism
affecting just one of them remains. For binary numbers u,v,

$$
u+v-(u\oplus v)=2uv.
$$

Using this identity in (1) and (2) yields

$$
\frac{b_i b_j}{b_{ij}}
=\prod_{e=1}^E(1-2q_e)^{2H_{ie}H_{je}}.
\tag{3}
$$

All factors are positive on the selected branch. Taking the positive square
root and applying S11 gives

$$
\begin{aligned}
1-2p_{ij}
&=\sqrt{\frac{b_i b_j}{b_{ij}}}\\
&=\prod_{e=1}^E(1-2q_e)^{H_{ie}H_{je}},
\end{aligned}
\tag{4}
$$

or, equivalently,

$$
p_{ij}
=\frac12\left[
1-\prod_{e:\,i,j\in S_e}(1-2q_e)
\right].
\tag{5}
$$

Thus p_ij is the probability that an odd number of mechanisms shared by i
and j occur. If their shared set contains exactly one mechanism e, p_ij=q_e.
If it contains several, p_ij combines their priors.

Equation S11 in terms of measured moments is

$$
p_{ij}=\frac12\left[
1-\sqrt{\frac{(1-2\mu_i)(1-2\mu_j)}
{1-2\mu_i-2\mu_j+4\mu_{ij}}}
\right].
\tag{6}
$$

Equations (1)--(6) use no low-error approximation.

## 3. A single matrix equation for all mechanisms

First restore the detector means on the diagonal:

$$
\widetilde P_{ij}=
\begin{cases}
p_{ij},&i\ne j,\\
\mu_i,&i=j.
\end{cases}
\tag{7}
$$

The package saves zero on the diagonal of p_ij.npy for visualization. Its
diagonal detector means must instead be read from moments.npz. Off-diagonal
pair data alone cannot determine priors of single-detector mechanisms.

Define

$$
a_e=-\log(1-2q_e),
\qquad
\Gamma_{ij}=-\log(1-2\widetilde P_{ij}).
\tag{8}
$$

The logarithm defining Gamma is entrywise, not a matrix logarithm. Taking
logarithms in (4), and in (1) for the diagonal, gives

$$
\Gamma_{ij}=\sum_{e=1}^E H_{ie}H_{je}a_e
\qquad(1\le i,j\le M).
\tag{9}
$$

The diagonal satisfies the same formula because H_ie^2=H_ie. Consequently,

$$
\Gamma
=H\operatorname{diag}(a)H^\top
=\sum_{e=1}^E a_e h_eh_e^\top.
\tag{10}
$$

For each mechanism e, the matrix h_e h_e^T is known from the assumed DEM
structure; its scalar coefficient a_e is unknown. Equation (10) expresses
the entire prior-estimation problem without mechanism-specific labels.
Although the observation x=HF is a binary parity equation, every operation
in (8)--(10) is over the real numbers.

To solve for the coefficients, stack the upper-triangular entries, including
the diagonal. Set R=M(M+1)/2 and define

$$
A\in\mathbb R^{R\times E},
\qquad A_{(i,j),e}=H_{ie}H_{je},
\qquad y_{(i,j)}=\Gamma_{ij},
\qquad 1\le i\le j\le M.
\tag{11}
$$

Then the unknown vector a in R^E satisfies

$$
Aa=y.
\tag{12}
$$

If A has full column rank, exact compatible moments give

$$
a=A^\dagger y,
\qquad q_e=\frac{1-\exp(-a_e)}2.
\tag{13}
$$

Here A^\dagger is the real Moore--Penrose pseudoinverse. A numerical
implementation should use a QR/SVD solve rather than explicitly invert
A^T A. The required rank condition concerns the matrices h_e h_e^T, not the
columns h_e alone; H itself need not have full column rank.

## 4. Identifiability from first and second moments

The condition rank_R(A)=E is sufficient for unique recovery of every prior.
It is also necessary for identifiability throughout the interior q_e>0:
if Av=0 for a nonzero vector v, a and a+t v produce identical first and
second moments whenever both vectors remain nonnegative. For interior a,
sufficiently small nonzero t always preserves nonnegativity. At the boundary
of the parameter set, nonnegativity can occasionally give uniqueness despite
rank deficiency; that is not a general identifiability guarantee.

The existence of hyperedges by itself does not settle this rank condition.
For example, the known supports

$$
S_1=\{1\},\quad S_2=\{2\},\quad S_3=\{3\},\quad
S_4=\{1,2,3\}
$$

give rank(A)=4. The only mechanism shared by any two detectors is mechanism
4, so p_12=p_13=p_23=q_4, after which the diagonal equations recover q_1,
q_2, and q_3. Thus this hypergraph is identifiable using first and second
moments alone.

If instead all seven nonempty detector supports on three detectors are
unknown mechanisms, A has six rows, seven columns, and rank six. First and
second moments do not identify all seven priors in the interior. Additional
statistics or parameter constraints are then necessary.

Identical detector supports produce identical columns in A. Detector-only
data cannot separate their individual priors, even when their logical labels
differ. An empty detector support produces a zero column. These limitations
already follow from the general matrix equation.

## 5. The graphlike formulas as explicit elimination

For the implemented graphlike model, each S_e has size one or two and every
nonempty support appears at most once after identifiable aggregation.

If S_e={i,j} with i not equal to j, only this mechanism contributes to the
off-diagonal entry Gamma_ij. Hence

$$
a_e=\Gamma_{ij},
\qquad q_e=p_{ij}.
\tag{14}
$$

For a mechanism S_e={i}, use the diagonal equation

$$
\Gamma_{ii}
=a_e+\sum_{\substack{f\ne e\\H_{if}=1}}a_f.
$$

The other incident mechanisms have support size two, so their coefficients
are already determined by (14). Therefore

$$
a_e=\Gamma_{ii}
-\sum_{\substack{f\ne e\\H_{if}=1}}a_f,
\tag{15}
$$

and

$$
q_e=\frac12\left[
1-\frac{1-2\mu_i}
{\prod_{\substack{f\ne e\\H_{if}=1}}(1-2q_f)}
\right].
\tag{16}
$$

Equation (16) is the general event-index form of S14--S16. It uses only the
known incidence relation and already recovered incident priors. An empty
product equals one. No label tied to the location of the mechanism is needed.

More generally, rearranging (1) gives (16) for any e incident to detector i
when all other incident priors are known. For a general hypergraph these
equations may need to be solved jointly through (12).

## 6. Empirical data and the implemented estimator

For N independent shots, substitute

$$
\widehat\mu_i=\frac1N\sum_{n=1}^N x_i^{(n)},
\qquad
\widehat\mu_{ij}=\frac1N\sum_{n=1}^N x_i^{(n)}x_j^{(n)}
$$

into (6), construct empirical Gamma, and infer a before applying (13).
Finite-sample moments need not lie exactly in the assumed model. A possible
general extension is nonnegative least squares,

$$
\widehat a
=\arg\min_{a\ge0}
\|\widehat\Gamma-H\operatorname{diag}(a)H^\top\|_F^2.
$$

This proposal presumes that the empirical logarithms are defined; invalid
empirical domains require a specified treatment. A rank-deficient fit is not
made identifiable merely by an optimization routine selecting a solution.

The current executable package uses the graphlike elimination in (14)--(16),
with the finite-sample regularization described in method.md. It does not
silently replace that estimator with the proposed general matrix fit.

## References

1. Chen et al., Exponential suppression of bit or phase flip errors with
   repetitive error correction, Supplement VI.1, equations S11--S16:
   https://arxiv.org/html/2102.06132v1
2. Stim detector error model specification (independent error instructions):
   https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md

The general support-index derivation and rank analysis above follow directly
from the stated independent Bernoulli model; they are not claims that the
original repetition-code paper implemented this general inverse solver.
