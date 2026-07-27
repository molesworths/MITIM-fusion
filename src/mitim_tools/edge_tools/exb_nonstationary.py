"""
Nonstationary flux surrogate for the L->H transition (ITG turn-on + ExB-shear suppression).

This is a *drop-in* variant of the standard MITIM flux GP (SURROGATEtools.surrogate_model /
BOTORCHtools.ExactGPcustom): same local inputs [aLte, aLti, aLne, aLw0_n, nuei, tite, w0_n,
beta_e], same gyroBohm-normalised flux output, same .fit()/.predict()/.gpmodel interface. It
is selected through ``edge_options: {nonstationary_exb: true}`` and touches nothing in the
optimizer / STEP / ModelList machinery.

It implements the construction in ``nonstationary_gp_construction.md``:

  * TARGET  -> ln(f):  a turbulent flux that turns on above the ITG critical gradient and is
    suppressed by ExB shear is a *non-monotonic bump* in the underlying drive; taking logs turns
    the turn-on x suppression *product* into an additive "tent" that GP machinery represents
    stably, and tames the exp() dynamic range so resolution is freed for the knee.
    -> ``Transformation_Outcomes_LogGB``: model ln( f/factorGB + offset ), lognormal back-transform.
    The offset is 0 for the (positive) heat channels and a small data-set shift for Ge, whose
    turbulent particle flux can dip negative (inward pinch) -- ln(f+offset) keeps the argument
    positive (asinh would be the alternative if Ge ran strongly negative).

  * MEAN    -> physics-shaped backbone with a LEARNABLE knee (``ExBPhysicsDriverMean``):

        d(x)      = w_lin * gamma_lin_itg  -  w_exb * gamma_ExB_dia  +  c0     (all normalized feats)
        gamma_eff = softplus(d)                                                (smooth ITG/ExB knee)
        m(x)      = a * ln( gamma_eff^2 + floor )  +  c_grad * aL_drive  +  b

    The features are *normalized*, but because w_lin/w_exb/c0 are LEARNED they absorb the
    per-feature normalization, so the knee LOCATION (where gamma_lin ~ w_exb/w_lin * gamma_ExB)
    is fit from data rather than frozen at the analytic corner -- this is what a stationary
    Matern residual cannot reliably do for a sharp, mis-located knee (spec Sec 3.2). w_exb is
    anchored by a prior (spec: anchor the suppression, learn the turn-on). c_grad * aL_drive is
    the diffusive linear-in-gradient backbone (aLti for Qi, aLte for Qe, aLne for Ge), off until
    earned (zero init + shrinkage prior), so the mean is "physics gate x diffusive drive", not a
    replacement that discards the standard linear mean.

  * RESIDUAL -> a global ARD Matern-5/2 (TypeKernel=0) over all normalized features absorbs the
    remaining model mismatch (now only the residual SHAPE, not the knee location).

  * NOISE   -> heteroscedastic (FixedNoise), so the log floor just below the knee is not
    down-weighted.

The turn-on surface (Guo-Romanelli) and the ExB suppression (diamagnetic Er) are precomputed in
physical (c_s/a) units in STATEtools.calculateProfileFunctions and consumed here as the features
``gamma_lin_itg`` / ``gamma_ExB_dia``; ``mu_exb`` (their analytic combination) is kept as an extra
residual coordinate.
"""

import torch
import gpytorch
import botorch

from mitim_tools.opt_tools import BOTORCHtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

# TypeMean id for the physics-driver mean (BOTORCHtools.ExactGPcustom dispatches on it).
TYPEMEAN_EXB = 4

# Precomputed physical driver features (STATEtools plasma keys) appended to the turbulent channels'
# surrogate_transformation_variables. They feed the learnable-knee MEAN only (see MEAN_ONLY_FEATURES):
# they are deterministic functions of the base inputs, so putting them in the ARD kernel would just
# add redundant lengthscales (extra sample burden); the spec routes the structured coordinates to the
# mean and keeps the residual ARD over the base inputs. (mu_exb was dropped -- the learnable mean now
# recomputes the gate from these two, so it is redundant.)
EXB_FEATURES = ["gamma_lin_itg", "gamma_ExB_dia"]

# Features excluded from the ARD residual kernel (mean-only). Kept identical to EXB_FEATURES so the
# residual kernel dimensionality is exactly the standard flux GP's -- no extra samples needed for it.
MEAN_ONLY_FEATURES = tuple(EXB_FEATURES)

# Turbulent channels this surrogate serves. Heat fluxes (positive) are modelled in ln space; Ge's
# turbulent particle flux can be large and inward (initial TGLF), so it uses a signed asinh transform.
EXB_CHANNELS = ("Qe_tr_turb", "Qi_tr_turb", "Ge_tr_turb")

# Channel prefix -> its driving gradient feature (diffusive linear-in-gradient backbone term).
_DRIVE_GRAD = {"Qe": "aLte", "Qi": "aLti", "Ge": "aLne"}


def is_exb_channel(output):
    """True if ``output`` (e.g. 'Qi_tr_turb_3') is a channel the ExB surrogate should serve."""
    if output is None:
        return False
    return "_".join(output.split("_")[:-1]) in EXB_CHANNELS


def is_signed_channel(output):
    """Ge's turbulent particle flux can be large and inward (initial TGLF) -> signed asinh, not log."""
    return (output is not None) and (output[:2] == "Ge")


# ----------------------------------------------------------------------------------------------
# Mean: learnable ITG/ExB knee (gate) x diffusive drive, in ln-flux space
# ----------------------------------------------------------------------------------------------

class ExBPhysicsDriverMean(gpytorch.means.mean.Mean):
    """
    m(x) = a * ln( softplus(w_lin*g_lin - w_exb*g_exb + c0)^2 + floor ) + c_grad*aL_drive + b

    g_lin = gamma_lin_itg feature, g_exb = gamma_ExB_dia feature (both normalized). The knee
    location is learned through (w_lin, w_exb, c0), which absorb the feature normalization; w_exb
    (the ExB suppression strength) is anchored by a prior, the turn-on is free. c_grad*aL_drive is
    the diffusive linear-in-gradient backbone (off until earned). Degrades to constant if the gate
    features are absent.
    """

    _SOFTPLUS_BETA = 8.0     # sharp-but-smooth knee; the razor-thin true corner is unresolvable anyway
    _FLOOR = 1.0e-3          # ln floor: sets the off-state level and knee softness

    def __init__(self, batch_shape=torch.Size(), variables=None, output=None,
                 enable_grad_term=True, exb_prior_sigma=0.5, **kwargs):
        super().__init__()

        def _idx(name):
            if variables is None:
                return None
            for i, v in enumerate(variables):
                if v == name:
                    return i
            return None

        self.i_lin = _idx("gamma_lin_itg")
        self.i_exb = _idx("gamma_ExB_dia")
        drive_name = _DRIVE_GRAD.get(output[:2]) if output is not None else None
        self.i_drive = _idx(drive_name) if (enable_grad_term and drive_name is not None) else None
        self.has_gate = (self.i_lin is not None) and (self.i_exb is not None)

        if not self.has_gate:
            print("\t\t- ExBPhysicsDriverMean: gate features (gamma_lin_itg/gamma_ExB_dia) not found, "
                  "falling back to constant mean", typeMsg="w")

        _sp_inv_1 = 0.5413248  # softplus^-1(1): so weights start at ~1 (trust the physics scale)
        # Gate: turn-on weight (free) and ExB-suppression weight (prior-anchored), + knee offset c0.
        self.register_parameter("raw_w_lin", torch.nn.Parameter(torch.full((*batch_shape, 1), _sp_inv_1)))
        self.register_parameter("raw_w_exb", torch.nn.Parameter(torch.full((*batch_shape, 1), _sp_inv_1)))
        self.register_parameter("c0", torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        # ln-flux scale (>0) and bias.
        self.register_parameter("raw_a", torch.nn.Parameter(torch.full((*batch_shape, 1), _sp_inv_1)))
        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(*batch_shape, 1)))

        # Anchor the ExB suppression strength (spec Sec 3.2: anchor x_cut, learn the turn-on).
        if self.has_gate:
            self.register_prior("w_exb_prior",
                                gpytorch.priors.NormalPrior(_sp_inv_1, float(exb_prior_sigma)),
                                "raw_w_exb")

        # Diffusive linear-in-gradient backbone: zero init + shrinkage prior (off until earned).
        if self.i_drive is not None:
            self.register_parameter("c_grad", torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_prior("c_grad_prior", gpytorch.priors.NormalPrior(0.0, 0.5), "c_grad")

    def forward(self, x):
        bias = self.bias.squeeze(-1)
        if not self.has_gate:
            out = bias * torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
        else:
            sp = torch.nn.functional.softplus
            g_lin = x[..., self.i_lin:self.i_lin + 1]
            g_exb = x[..., self.i_exb:self.i_exb + 1]
            d = sp(self.raw_w_lin) * g_lin - sp(self.raw_w_exb) * g_exb + self.c0
            gamma_eff = sp(d, beta=self._SOFTPLUS_BETA)                       # smooth ITG/ExB knee
            m = sp(self.raw_a) * torch.log(gamma_eff ** 2 + self._FLOOR)      # (*b, n, 1)
            out = m.squeeze(-1) + bias

        if self.i_drive is not None:
            out = out + (self.c_grad * x[..., self.i_drive:self.i_drive + 1]).squeeze(-1)
        return out


# ----------------------------------------------------------------------------------------------
# Outcome transform: model ln(gyroBohm flux + offset); back-transform to (shifted) lognormal flux
# ----------------------------------------------------------------------------------------------

# Hard cap on warp-space arguments before exp/sinh in the back-transforms. Extrapolated warp-space
# means can run away (the spec's Sec-4 extrapolation hazard); exp/sinh then overflow to inf. e^40 ~
# 2e17 GB is already absurd but finite -- directionally "huge flux" for the optimizer, never inf/nan.
_WARP_CLAMP = 40.0


class _Transformation_Outcomes_GBbase(BOTORCHtools.Transformation_Outcomes):
    """Shared GB-factor plumbing (tf1 of the chain). Subclasses set the scalar warp / back-transform."""

    _EPS = 1e-8

    def _factor(self, X):
        if (self.output is not None) and self.flag_to_evaluate:
            return self.surrogate_parameters["transformationOutputs"](
                X, self.surrogate_parameters, self.output
            ).to(X.device)
        return None

    def _mark_identity(self, factor):
        # keep the Standardize base-state self-consistent (identity here; tf2 does the scaling)
        self.stdvs = torch.ones_like(factor)
        self.means = torch.zeros_like(factor)
        self._stdvs_sq = self.stdvs
        self._is_trained = torch.tensor(True)
        self.training = False


class Transformation_Outcomes_LogGB(_Transformation_Outcomes_GBbase):
    """
    Heat-flux transform: forward Y -> ln( Y/factorGB ); untransform -> lognormal flux posterior.
    Noise maps by the delta method Var[ln g] ~= Var[Y]/(factorGB g)^2, g = Y/factorGB.
    """

    def forward(self, X, Y, Yvar):
        factor = self._factor(X)
        if factor is None:
            factor = torch.ones_like(Y)
        g = (Y / factor).clamp_min(self._EPS)
        Ylog = torch.log(g)
        Ylog_var = Yvar / (factor ** 2 * g ** 2) if Yvar is not None else None
        self._mark_identity(factor)
        return Ylog, Ylog_var

    def untransform_posterior(self, X, posterior):
        factor = self._factor(X)
        if factor is None:
            factor = torch.ones(1, dtype=X.dtype, device=X.device)
        c = _WARP_CLAMP
        return botorch.posteriors.transformed.TransformedPosterior(
            posterior,
            sample_transform=lambda s: factor * torch.exp(s.clamp(-c, c)),
            mean_transform=lambda m, v: factor * torch.exp((m + 0.5 * v).clamp(-c, c)),
            variance_transform=lambda m, v: (factor ** 2) * torch.exp((2 * m + v).clamp(-c, c))
            * (torch.exp(v.clamp(max=c)) - 1.0),
        )


class Transformation_Outcomes_AsinhGB(_Transformation_Outcomes_GBbase):
    """
    Signed transform for Ge (turbulent particle flux, which can be large and inward):

        forward:      Y  ->  asinh( (Y/factorGB) / s )        (linear near 0, +-log in both tails)
        untransform:  Gaussian(mu,var) in warp space  ->  factorGB * s * sinh(.)  flux posterior

    sinh-of-Gaussian moments are analytic:
        E[sinh X]  = e^{var/2} sinh(mu)
        E[sinh^2 X]= (e^{2var} cosh(2mu) - 1) / 2
    Noise maps by the delta method: d asinh((Y/f)/s)/dY = 1 / sqrt(Y^2 + (f s)^2).
    """

    def __init__(self, m, output, surrogate_parameters, scale=1.0):
        super().__init__(m, output, surrogate_parameters)
        self.register_buffer("scale", torch.as_tensor(float(scale)).clamp_min(1e-8))

    def forward(self, X, Y, Yvar):
        factor = self._factor(X)
        if factor is None:
            factor = torch.ones_like(Y)
        s = self.scale
        Yw = torch.asinh((Y / factor) / s)
        Yw_var = Yvar / (Y ** 2 + (factor * s) ** 2) if Yvar is not None else None
        self._mark_identity(factor)
        return Yw, Yw_var

    def untransform_posterior(self, X, posterior):
        factor = self._factor(X)
        if factor is None:
            factor = torch.ones(1, dtype=X.dtype, device=X.device)
        s = self.scale

        c = _WARP_CLAMP

        def _mean(m, v):
            return factor * s * torch.exp(0.5 * v.clamp(max=c)) * torch.sinh(m.clamp(-c, c))

        def _var(m, v):
            mc, vc = m.clamp(-c, c), v.clamp(max=c)
            e_sinh2 = 0.5 * (torch.exp(2.0 * vc) * torch.cosh(2.0 * mc) - 1.0)
            return (factor * s) ** 2 * (e_sinh2 - (torch.exp(0.5 * vc) * torch.sinh(mc)) ** 2).clamp_min(0.0)

        return botorch.posteriors.transformed.TransformedPosterior(
            posterior,
            sample_transform=lambda z: factor * s * torch.sinh(z.clamp(-c, c)),
            mean_transform=_mean,
            variance_transform=_var,
        )


def compute_asinh_scale(train_Y, factor, floor=1e-3):
    """Per-fit asinh scale ~ robust magnitude of the GB flux (sets the linear->log knee of asinh)."""
    if train_Y is None or train_Y.numel() == 0:
        return 1.0
    g = (train_Y / factor).detach().abs()
    return max(g.median().item(), floor)


def build_outcome_transform(dimY, output, surrogate_parameters, dfT, scale=1.0, signed=None):
    """Select the ExB physics outcome transform (tf1): signed asinh vs ln.

    ``signed`` is decided by the caller from the DATA (any negative training flux -> asinh);
    if None, falls back to the channel default (Ge signed, heat channels ln).
    """
    if signed is None:
        signed = is_signed_channel(output)
    if signed:
        return Transformation_Outcomes_AsinhGB(dimY, output, surrogate_parameters, scale=scale).to(dfT)
    return Transformation_Outcomes_LogGB(dimY, output, surrogate_parameters).to(dfT)
