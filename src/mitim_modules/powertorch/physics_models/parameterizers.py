"""
Parameterization models for plasma profiles (n_e, T_e, T_i, ...).

Provides a common interface via ParameterModel and concrete implementations:
- SplineParameterModel (Akima or PCHIP), parameterizing a/Ly at user-defined knots
- MTanhParameterModel (stub)
- GaussianRBFParameterModel (stub)

A factory function create_parameter_model(config) instantiates the requested model.

Conventions
-----------
- Coordinate x: parameterizer classes operate on normalized radius x = r/a ("roa").
    In this coordinate, a/Ly = - d(ln y)/d x, which integrates naturally.
- Boundary condition: to reconstruct y from gradients, a boundary value y_sep at the
  outermost grid point is required. Pass explicitly to .y(..., y_sep=...), or provide
  bc_field in options to read from state.BC.<bc_field> when state is supplied.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union, List
import numpy as np
import torch
import copy
import math
from scipy.interpolate import InterpolatedUnivariateSpline as linear, Akima1DInterpolator as akima, PchipInterpolator as pchip, CubicSpline, BSpline
from scipy.special import erf
from scipy.optimize import curve_fit
import scipy as sp  
from scipy.optimize import least_squares
from scipy.special import gamma as Gamma
from scipy.integrate import cumulative_trapezoid
from mitim_modules.powertorch.utils import CALCtools
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# -------------------------
# Legacy form for powerstate and portals_main
# -------------------------


def piecewise_linear(
    x_coord,
    y_coord_raw,
    x_coarse_tensor,
    parameterize_in_aLx=True,
    multiplier_quantity=1.0,
    ):
    """
    Notes:
        - x_coarse_tensor must be torch
    """

    # **********************************************************************************************************
    # Define the integrator and derivator functions (based on whether I want to parameterize in aLx or in gradX)
    # **********************************************************************************************************

    if parameterize_in_aLx:
        # 1/Lx = -1/X*dX/dr
        integrator_function, derivator_function = (
            CALCtools.integration_Lx,
            CALCtools.derivation_into_Lx,
        )
    else:
        # -dX/dr
        integrator_function, derivator_function = (
            CALCtools.integration_dxdr,
            CALCtools.derivation_into_dxdr,
        )

    y_coord = torch.from_numpy(y_coord_raw).to(x_coarse_tensor) * multiplier_quantity

    ygrad_coord = derivator_function( torch.from_numpy(x_coord).to(x_coarse_tensor), y_coord )

    # **********************************************************************************************************
    # Get control points
    # **********************************************************************************************************

    x_coarse = x_coarse_tensor[1:].cpu().numpy()

    """
    Define region to get control points from
    ------------------------------------------------------------
	Trick: Addition of extra point
		This is important because if I don't, when I combine the trailing edge and the new
		modified profile, there's going to be a discontinuity in the gradient.
	"""
    
    ir_end = np.argmin(np.abs(x_coord - x_coarse[-1]))

    if ir_end < len(x_coord) - 1:
        ir = ir_end + 2  # To prevent that TGYRO does a 2nd order derivative
        x_coarse = np.append(x_coarse, [x_coord[ir]])
    else:
        ir = ir_end

	# Definition of trailing edge. Any point after, and including, the extra point
    x_trail = torch.from_numpy(x_coord[ir:]).to(x_coarse_tensor)
    y_trail = y_coord[ir:]
    x_notrail = torch.from_numpy(x_coord[: ir + 1]).to(x_coarse_tensor)

    # Produce control points, including a zero at the beginning
    aLy_coarse = [[0.0, 0.0]]
    for cont, i in enumerate(x_coarse):
        yValue = ygrad_coord[np.argmin(np.abs(x_coord - i))]
        aLy_coarse.append([i, yValue.cpu().item()])

    aLy_coarse = torch.from_numpy(np.array(aLy_coarse)).to(ygrad_coord)

    # Since the last one is an extra point very close, I'm making it the same
    aLy_coarse[-1, 1] = aLy_coarse[-2, 1]

    # Boundary condition at point moved by gridPointsAllowed
    y_bc = torch.from_numpy(interpolation_function([x_coarse[-1]], x_coord, y_coord.cpu().numpy())).to(ygrad_coord)

    # Boundary condition at point (ACTUAL THAT I WANT to keep fixed, i.e. roa=0.8)
    y_bc_real = torch.from_numpy(interpolation_function([x_coarse[-2]], x_coord, y_coord.cpu().numpy())).to(ygrad_coord)

    # **********************************************************************************************************
    # Define profile_constructor functions
    # **********************************************************************************************************

    def profile_constructor_coarse(x, y, multiplier=multiplier_quantity):
        """
        Construct curve in a coarse grid
        ----------------------------------------------------------------------------------------------------
        This constructs a curve in any grid, with any batch given in y=y.
        Useful for surrogate evaluations. Fast in a coarse grid. For HF evaluations,
        I need to do in a finer grid so that it is consistent with TGYRO.
        x, y must be (batch, radii),	y_bc must be (1)
        """
        return x, integrator_function(x, y, y_bc_real) / multiplier

    def profile_constructor_middle(x, y, multiplier=multiplier_quantity):
        """
        Deparamterizes a finer profile based on the values in the coarse.
        Reason why something like this is not used for the full profile is because derivative of this will not be as original,
                which is needed to match TGYRO
        """
        yCPs = CALCtools.Interp1d_torch()(aLy_coarse[:, 0][:-1].repeat((y.shape[0], 1)), y, x)
        return x, integrator_function(x, yCPs, y_bc_real) / multiplier

    def profile_constructor_fine(x, y, multiplier=multiplier_quantity):
        """
        Notes:
            - x is a 1D array, but y can be a 2D array for a batch of individuals: (batch,x)
            - I am assuming it is 1/LT for parameterization, but gives T
        """

        y = torch.atleast_2d(y)
        x = x[0, :] if x.dim() == 2 else x

        # Add the extra trick point
        x = torch.cat((x, aLy_coarse[-1][0].repeat((1))))
        y = torch.cat((y, aLy_coarse[-1][-1].repeat((y.shape[0], 1))), dim=1)

        # Model curve (basically, what happens in between points)
        yBS = CALCtools.Interp1d_torch()(x.repeat(y.shape[0], 1), y, x_notrail.repeat(y.shape[0], 1))

        """
        ---------------------------------------------------------------------------------------------------------
            Trick 1: smoothAroundCoarsing
                TGYRO will use a 2nd order scheme to obtain gradients out of the profile, so a piecewise linear
                will simply not give the right derivatives.
                Here, this rough trick is to modify the points in gradient space around the coarse grid with the
                same value of gradient, so in principle it doesn't matter the order of the derivative.
        """
        num_around = 1
        for i in range(x.shape[0] - 2):
            ir = torch.argmin(torch.abs(x[i + 1] - x_notrail))
            for k in range(-num_around, num_around + 1, 1):
                yBS[:, ir + k] = yBS[:, ir]
        # --------------------------------------------------------------------------------------------------------

        yBS = integrator_function(x_notrail.repeat(yBS.shape[0], 1), yBS.clone(), y_bc)

        """
        Trick 2: Correct y_bc
            The y_bc for the profile integration started at gridPointsAllowed, but that's not the real
            y_bc. I want the temperature fixed at my first point that I actually care for.
            Here, I multiply the profile to get that.
            Multiplication works because:
                1/LT = 1/T * dT/dr
                1/LT' = 1/(T*m) * d(T*m)/dr = 1/T * dT/dr = 1/LT
            Same logarithmic gradient, but with the right boundary condition

        """
        ir = torch.argmin(torch.abs(x_notrail - x[-2]))
        yBS = yBS * torch.transpose((y_bc_real / yBS[:, ir]).repeat(yBS.shape[1], 1), 0, 1)

        # Add trailing edge
        y_trailnew = copy.deepcopy(y_trail).repeat(yBS.shape[0], 1)

        x_notrail_t = torch.cat((x_notrail[:-1], x_trail), dim=0)
        yBS = torch.cat((yBS[:, :-1], y_trailnew), dim=1)

        return x_notrail_t, yBS / multiplier

    # **********************************************************************************************************

    return (
        aLy_coarse,
        profile_constructor_fine,
        profile_constructor_coarse,
        profile_constructor_middle,
    )


# -------------------------
# Base parameter model
# -------------------------

class ParameterBase:
    """Abstract base class for parameterizing a scalar profile y(x).

    Expected common methods:
    - get_aLy(x_eval, params) -> np.ndarray: return a/Ly on x_eval
    - get_y(x_eval, params) -> np.ndarray: return y on x_eval
    - get_curvature(x_eval, params) -> np.ndarray: return d2y/dx2 on x_eval
    - _build_interpolator(x_data, y_data) -> store interpolator
    - _build_bc_dict(boundary_model, state) -> store dict of BC values from boundary model
    - update_all(boundary_model, state) -> recalculate attributes if dirty flag is set
    
    Attributes:
    - options: Dict[str, Any] of model options
    - interpolator: callable interpolator object
    - bcs: Dict[str, float] of boundary condition values 
    - params: Dict[str, np.ndarray] of model parameters
    - y: Dict[str, np.ndarray] of reconstructed profiles
    - aLy: Dict[str, np.ndarray] of a/Ly profiles
    - dirty: bool flag indicating if model needs re-initialization

    Notes
    -----
    - x_eval is 1D normalized radius in roa (x = r/a).

    """

    def __init__(self, options: Dict[str, Any]):
        options = dict(options or {})

        self.predicted_profiles = list(options.get("predicted_profiles", []))
        self.include_zero_grad_on_axis = bool(options.get("include_zero_grad_on_axis", True))
        self.sigma = float(options.get("sigma", 0.05))
        self.bounds = options.get("bounds", None)

        self.params: Dict[str, np.ndarray] = {}
        self.param_std: Dict[str, np.ndarray] = {}
        self.param_names: List[str] = []
        self.bc_dict: Dict[str, List[BCEntry]] = {}
        self.bc_tensors: Dict[str, Dict[str, Any]] = {}

    # ------------------------------
    # Abstract-like interface
    # ------------------------------
    def add_bc(self, key: str, bc: BCEntry):
        self.bc_dict.setdefault(key, []).append({"val": float(bc['val']), "loc": float(bc['loc'])})

    def build_bcs(self, bc_dict: Dict[str, Any]) -> Dict[str, List[BCEntry]]:
        """
        Normalize and store BCs. Accepts:
          bc_dict = {"ne": (1e19, 1.0), "aLne": {"val": -2.0, "loc": 1.0},
                     "aLne": [( -1.5, 0.95), (-2.0, 1.0 )] }
        Stored form: self.bc_dict['aLne'] = [ {'val':..., 'loc':...}, {...} ]
        """
        self.bc_dict = {}

        for key, val in bc_dict.items():
            # allow list of entries
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple, dict)):
                for v in val:
                    self.add_bc(key, _normalize_single_bc(v))
            else:
                # single entry (tuple/list or dict)
                self.add_bc(key, _normalize_single_bc(val))

        # Optionally ensure aL<prof> has an axis BC at 0.0 (append only if no exact axis entry)
        if self.include_zero_grad_on_axis:
            for prof in self.predicted_profiles:
                key = f"aL{prof}"
                entries = self.bc_dict.get(key, [])
                has_axis = any(np.isclose(e['loc'], 0.0) for e in entries)
                if not has_axis:
                    # append axis zero-gradient BC (do not overwrite existing BCs)
                    self.add_bc(key, {"val": 0.0, "loc": 0.0})

        return self.bc_dict

    def get_nearest_bc(self, key: str, location: float) -> Union[BCEntry, None]:
        """Return the BC entry with location nearest to the requested location."""
        entries = self.bc_dict.get(key, [])
        if not entries:
            return None
        locs = np.array([e['loc'] for e in entries], dtype=float)
        idx = int(np.argmin(np.abs(locs - location)))
        return entries[idx]

    @staticmethod
    def _to_numpy_any(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _is_batched_bc_input(self, bc_dict: Dict[str, Any], batch_size: int) -> bool:
        if not isinstance(bc_dict, dict):
            return False
        for val in bc_dict.values():
            if not (isinstance(val, dict) and ("val" in val) and ("loc" in val)):
                continue
            v = self._to_numpy_any(val["val"])
            l = self._to_numpy_any(val["loc"])
            if v.ndim >= 1 and l.ndim >= 1 and v.shape[0] == batch_size and l.shape[0] == batch_size:
                return True
        return False

    def _slice_batched_bc_dict(self, bc_dict: Dict[str, Any], batch_idx: int, batch_size: int) -> Dict[str, Any]:
        """Convert batched BC tensors into a scalar/list BC dict for one batch index."""
        sliced: Dict[str, Any] = {}

        for key, val in bc_dict.items():
            if not (isinstance(val, dict) and ("val" in val) and ("loc" in val)):
                sliced[key] = val
                continue

            v = self._to_numpy_any(val["val"])
            l = self._to_numpy_any(val["loc"])

            if not (v.ndim >= 1 and l.ndim >= 1 and v.shape[0] == batch_size and l.shape[0] == batch_size):
                sliced[key] = val
                continue

            if "mask" in val:
                m = self._to_numpy_any(val["mask"]).astype(bool)
            else:
                m = np.ones_like(v, dtype=bool)

            v_row = np.ravel(v[batch_idx])
            l_row = np.ravel(l[batch_idx])
            m_row = np.ravel(m[batch_idx]) if m.ndim >= 1 and m.shape[0] == batch_size else np.ones_like(v_row, dtype=bool)

            entries = []
            for j in range(min(len(v_row), len(l_row), len(m_row))):
                if m_row[j]:
                    entries.append((float(v_row[j]), float(l_row[j])))

            if len(entries) == 1:
                sliced[key] = entries[0]
            elif len(entries) > 1:
                sliced[key] = entries

        return sliced
    
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract parameters from a PlasmaState given boundary conditions."""
        raise NotImplementedError

    def get_aLy(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute scale length a/Ly = -a * (dy/dx) / y for each profile."""
        raise NotImplementedError

    def get_y(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute profile y(x) from parameter set."""
        raise NotImplementedError

    def get_curvature(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² for each profile."""
        raise NotImplementedError

    def update(self, params: Dict[str, np.ndarray], bc_dict: Dict[str, Any], x_eval: np.ndarray):
        """Convenience method returning y, aLy, and curvature."""
        self.bc_tensors = {}

        def _to_numpy(value: Any) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                # Support torch tensors without requiring torch as a hard dependency.
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def _slice_params_for_batch(
            all_params: Dict[str, Any], batch_idx: int, batch_size: int
        ) -> Dict[str, Any]:
            sliced: Dict[str, Any] = {}
            for prof, prof_params in all_params.items():
                if isinstance(prof_params, dict):
                    prof_sliced: Dict[str, Any] = {}
                    for name, value in prof_params.items():
                        arr = _to_numpy(value)
                        # Dict-valued parameters in these models are scalar per profile;
                        # a leading batch axis indicates per-batch values.
                        if arr.ndim == 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = float(arr[batch_idx])
                        elif arr.ndim > 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = arr[batch_idx]
                        else:
                            prof_sliced[name] = value
                    sliced[prof] = prof_sliced
                else:
                    arr = _to_numpy(prof_params)
                    # For array-valued params, only treat as batched when a leading
                    # batch dimension is explicit (ndim > 1).
                    if arr.ndim > 1 and arr.shape[0] == batch_size:
                        sliced[prof] = arr[batch_idx]
                    else:
                        sliced[prof] = prof_params
            return sliced

        x_arr = _to_numpy(x_eval)

        if x_arr.ndim <= 1:
            self.build_bcs(bc_dict)
            y = self.get_y(params, x_arr)
            aLy = self.get_aLy(params, x_arr)
            curvature = self.get_curvature(params, x_arr)
            return y, aLy, curvature

        # Batched evaluation: iterate over batch dimension and stack results.
        batch_size = x_arr.shape[0]
        use_batched_bcs = self._is_batched_bc_input(bc_dict, batch_size)
        if use_batched_bcs:
            self.bc_tensors = bc_dict
        else:
            self.build_bcs(bc_dict)

        y_batches: Dict[str, List[np.ndarray]] = {}
        aLy_batches: Dict[str, List[np.ndarray]] = {}
        curv_batches: Dict[str, List[np.ndarray]] = {}

        for i in range(batch_size):
            if use_batched_bcs:
                self.build_bcs(self._slice_batched_bc_dict(bc_dict, i, batch_size))
            batch_params = _slice_params_for_batch(params, i, batch_size)
            y_i = self.get_y(batch_params, x_arr[i])
            aLy_i = self.get_aLy(batch_params, x_arr[i])
            curv_i = self.get_curvature(batch_params, x_arr[i])

            for prof, vals in y_i.items():
                y_batches.setdefault(prof, []).append(_to_numpy(vals))
            for prof, vals in aLy_i.items():
                aLy_batches.setdefault(prof, []).append(_to_numpy(vals))
            for prof, vals in curv_i.items():
                curv_batches.setdefault(prof, []).append(_to_numpy(vals))

        y_out = {prof: np.stack(vals, axis=0) for prof, vals in y_batches.items()}
        aLy_out = {prof: np.stack(vals, axis=0) for prof, vals in aLy_batches.items()}
        curv_out = {prof: np.stack(vals, axis=0) for prof, vals in curv_batches.items()}

        self.y = y_out
        self.aLy = aLy_out
        self.curv = curv_out
        return y_out, aLy_out, curv_out


# -------------------------
# Spline parameter model
# -------------------------


class Spline(ParameterBase):
    """Spline-based parameterization of a/Ly with control points at user-defined knots.

    Design parameters: self.defined_on + i for i in range(len(knots))

    Parameters (options)
    --------------------
    knots : Sequence[float]
        Locations in x (roa=r/a) where parameters define a/Ly values.
    spline_type : str
        'akima' (default) or 'pchip'. Determines the interpolator.
    include_zero_grad_on_axis : bool
        If True (default) and knots do not include x=0, a virtual control point with a/Ly=0 at x=0
        is prepended for smooth behavior at the magnetic axis.
    bc_field : Optional[str]
        Name of boundary condition value on state.BC (e.g., 'ne', 'te', 'ti') to use as y_sep if
        not explicitly provided to y()/curvature().
    """

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.spline_type = options.get('spline_type', 'linear').lower()
        self.knots = np.array(options.get('knots', []))
        self.defined_on = options.get('defined_on', 'aLy')
        if self.spline_type not in ('akima', 'pchip', 'cubic', 'linear'):
            raise ValueError("spline_type must be 'akima', 'pchip', 'cubic', or 'linear'")
        self.param_names = [self.defined_on+str(i) for i in range(len(self.knots))]
        self.n_params_per_profile = len(self.knots)
        self.splines: Dict[str, Any] = {}
        self._trailing_edge: Dict[str, Any] = {}
        self._exp_full: Dict[str, Any] = {}   # length-consistent raw experimental profile (fix_tail)
        # fix_tail_experimental: for defined_on='aLy', parameterize a/Ly ONLY on the knot
        # region and HOLD x > last-knot at the captured experimental profile (isolates the
        # flux-match region; the untrusted ~0.96<rho<1 tail stays at experiment, preserving
        # the true pedestal a/Ly peak and y_ped). The optimizer DVs remain the knots only.
        self.fix_tail_experimental = bool(options.get('fix_tail_experimental', False))

    # ------------------------------
    # Internal utilities
    # ------------------------------
    def _make_spline(self, x: np.ndarray, y: np.ndarray, prof: str):
        """Return a spline object of chosen type.
        
        If include_zero_grad_on_axis=True and x doesn't start at 0, prepend axis BC.
        """
        x_spline = np.asarray(x)
        y_spline = np.asarray(y)

        # Traceable guard: scipy's spline constructors raise an opaque
        # "`y` must contain only finite values" from deep in the C layer, which
        # hides the culprit during edge-UQ scans. Surface it here with the
        # profile and the offending values so the source is immediately clear.
        if not np.all(np.isfinite(x_spline)) or not np.all(np.isfinite(y_spline)):
            raise ValueError(
                f"_make_spline('{prof}', type={self.spline_type}): non-finite input to spline "
                f"construction. x_finite={np.all(np.isfinite(x_spline))}, "
                f"y_finite={np.all(np.isfinite(y_spline))}; "
                f"x={np.asarray(x_spline, dtype=float)}, y={np.asarray(y_spline, dtype=float)}"
            )

        if self.include_zero_grad_on_axis and not np.isclose(x_spline[0], 0.0) and self.defined_on == 'aLy':
            x_spline = np.insert(x_spline, 0, 0.0)
            y_spline = np.insert(y_spline, 0, 0.0)
        
        # Build spline
        if self.spline_type == "akima":
            spline = akima(x_spline, y_spline, extrapolate=True)
        elif self.spline_type == "pchip":
            spline = pchip(x_spline, y_spline, extrapolate=True)
        elif self.spline_type in ("cubic", "cspline"):
            spline = CubicSpline(x_spline, y_spline, extrapolate=True)
        elif self.spline_type == "linear":
            spline = linear(x_spline, y_spline, k=1)
        else:
            raise ValueError(f"Unknown spline_type: {self.spline_type}")
        
        self.splines[prof] = spline
        return spline
    
    def _get_spline(self, prof: str):
        """Retrieve a cached spline."""
        spline = self.splines.get(prof)
        if spline is None:
            raise KeyError(f"Spline for profile '{prof}' not initialized.")
        return spline

    def _integrate_aLy(self, prof: str, x_eval: np.ndarray, spl: Any, bc_value: float, bc_loc: float) -> np.ndarray:
        """
        Integrate spline of a/Ly to recover y(x) via
            dy/dx = -aLy * y   (for x = roa = r/a dimensionless)

            => y(x) = y_bc * exp(-∫[bc_loc to x] aLy dx')

        For bc_loc = 1.0 (edge), integrating inward (decreasing x) gives positive integral.
        For bc_loc = 0.0 (axis), integrating outward (increasing x) gives negative integral.
        """

        if not hasattr(spl, "antiderivative"):
            raise TypeError(f"Spline type {type(spl)} has no .antiderivative()")

        # y(x) = y_bc * exp(-∫[bc_loc→x] aLy dx')
        # aLy = a/Ly = -d(ln y)/d(roa), so the integral is already dimensionless.
        F = spl.antiderivative()
        phase = -(F(x_eval) - F(bc_loc))

        return bc_value * np.exp(phase)

    # ------------------------------
    # Implement required methods
    # ------------------------------
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract spline coefficients (y values at knots) from a state.
        
        Incorporates boundary conditions by merging BC points into the spline
        construction data before extracting parameters at knots.
        """

        if self.bounds is None:
            self.bounds = {name: (0.0, 100.0) for name in self.param_names}
        
        self.a = state.a  # store for conversions

        self.build_bcs(bc_dict)
        params = {}
        roa_vals = getattr(state, 'roa')
        
        for prof in self.predicted_profiles:
            prof_name = f"aL{prof}" if self.defined_on == "aLy" else prof
            y_prof = getattr(state, prof_name)
            if y_prof.ndim == 2:
                y_prof = y_prof[:, 0].flatten()
            else:
                y_prof = np.asarray(y_prof).flatten()
            
            # Merge boundary conditions into spline data
            x_data = np.asarray(roa_vals).flatten()
            y_data = np.asarray(y_prof).flatten()
            
            # Get BC entries for this profile
            bc_entries = self.bc_dict.get(prof_name, [])
            
            #Add/replace BC points in the data
            for bc in bc_entries:
                bc_loc = bc['loc']
                bc_val = bc['val']
                
                #Find if this location already exists in data (within tolerance)
                existing_idx = np.where(np.isclose(x_data, bc_loc, atol=1e-6))[0]
                
                if len(existing_idx) > 0:
                    # Replace existing point
                    y_data[existing_idx[0]] = bc_val
                else:
                    # Insert new point in sorted order
                    insert_idx = np.searchsorted(x_data, bc_loc)
                    x_data = np.insert(x_data, insert_idx, bc_loc)
                    y_data = np.insert(y_data, insert_idx, bc_val)
            
            # Store trailing edge data (from last knot onward) for bc=None fallback in get
            # methods and for fix_tail_experimental. Index on the RAW state grid (roa_vals),
            # NOT the BC-augmented x_data, so 'x' stays the same length as 'y'/'aLy' even when
            # an LCFS BC point was inserted above (x_data == raw grid on the ghost path, so no
            # behavior change there).
            last_knot = self.knots[-1]
            raw_roa = np.asarray(roa_vals).flatten()
            split = np.searchsorted(raw_roa, last_knot, side='right')
            te_start = max(0, split - 1)  # include one overlap point at/before last_knot
            te_x = raw_roa[te_start:]
            def _te_arr(attr):
                try:
                    arr = np.asarray(getattr(state, attr)).flatten()
                    return arr[te_start:]
                except AttributeError:
                    return None
            self._trailing_edge[prof] = {
                'x':   te_x,
                'y':   _te_arr(prof),
                'aLy': _te_arr(f'aL{prof}'),
            }

            # fix_tail_experimental needs a LENGTH-CONSISTENT raw experimental profile
            # (the _trailing_edge x is BC-augmented while its y/aLy come from the raw
            # state -> lengths diverge once an LCFS BC is inserted). Capture all three
            # straight off the raw state grid so np.interp is well-posed.
            def _raw(attr):
                arr = np.asarray(getattr(state, attr))
                return (arr[:, 0] if arr.ndim == 2 else arr).flatten()
            self._exp_full[prof] = {
                'x':   np.asarray(roa_vals).flatten(),
                'y':   _raw(prof),
                'aLy': _raw(f'aL{prof}'),
            }

            # Build spline with BC-augmented data
            spline = self._make_spline(x_data, y_data, prof)
            params[prof] = dict(zip(self.param_names, spline(self.knots)))
        
        self.params = params
        std_dict = {prof: {name: abs(val)*self.sigma for name, val in params[prof].items()} for prof in params}
        self.param_std = std_dict

        return params, std_dict  # return nominal and std dev
    
    def _fixed_tail_reconstruct(self, prof: str, vals: np.ndarray, x_eval: np.ndarray):
        """fix_tail_experimental (defined_on='aLy'): parameterize a/Ly ONLY on the knot
        region and hold x > last-knot at the captured experimental profile. Returns
        (y, aLy, curvature) as a self-consistent triple. The interior is integrated inward
        from the EXPERIMENTAL y at the last knot, so the knot DVs never touch the untrusted
        tail and the true pedestal a/Ly peak / y_ped are preserved."""
        last = float(self.knots[-1])
        te = self._exp_full.get(prof) or {}
        te_x, te_y, te_aLy = te.get('x'), te.get('y'), te.get('aLy')
        if te_x is None or te_y is None or te_aLy is None or len(np.asarray(te_x)) < 2:
            raise ValueError(f"fix_tail_experimental: no captured experimental profile for '{prof}' "
                             "(parameterize() must run on the experimental state first)")
        te_x = np.asarray(te_x); te_y = np.asarray(te_y); te_aLy = np.asarray(te_aLy)
        spline = self._make_spline(self.knots, np.asarray(vals), prof)   # interior only, no LCFS BC
        x = np.asarray(x_eval, dtype=float)
        y_anchor = float(np.interp(last, te_x, te_y))                    # experimental y at last knot
        y_int  = self._integrate_aLy(prof, x, spline, y_anchor, last)    # y = y_anchor*exp(-∫[last→x] aLy)
        aLy_int = spline(x)                                              # spline represents a/Ly
        curv_int = -y_int * (spline.derivative(1)(x) - aLy_int**2)
        y_tail   = np.interp(x, te_x, te_y)
        aLy_tail = np.interp(x, te_x, te_aLy)
        daLy_tail = np.gradient(aLy_tail, x) if x.size > 1 else np.zeros_like(x)
        curv_tail = -y_tail * (daLy_tail - aLy_tail**2)
        m = x > last
        y    = np.where(m, y_tail,   y_int)
        aLy  = np.where(m, aLy_tail, aLy_int)
        curv = np.where(m, curv_tail, curv_int)
        return np.clip(y, 0, None), np.clip(aLy, 0, None), curv

    def _use_fixed_tail(self) -> bool:
        return self.fix_tail_experimental and self.defined_on == "aLy"

    def get_y(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute profiles y(x) on x_eval."""
        out = {}
        for prof, prof_params in params.items():

            if self._use_fixed_tail():
                vals = np.array([prof_params[n] for n in self.param_names]) if isinstance(prof_params, dict) else np.asarray(prof_params)
                out[prof], _, _ = self._fixed_tail_reconstruct(prof, vals, x_eval)
                continue

            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)

            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            use_ghost = False
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                # Ghost point: add a point just beyond the last knot with the same value;
                # from there to roa=1 y is held constant (trailing edge stitching).
                _eps = 1e-4
                knots = np.append(self.knots, self.knots[-1] + _eps)
                vals = np.append(vals, vals[-1])
                use_ghost = True

            ghost_knot = knots[-1]
            spline = self._make_spline(knots, vals, prof)

            if self.defined_on == "y":
                y_spl = spline(x_eval)
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_y = te.get('x'), te.get('y')
                    if te_x is not None and te_y is not None and len(te_x) >= 2:
                        y_trail = np.interp(x_eval, te_x, te_y)
                    else:
                        y_trail = np.full_like(x_eval, float(spline(self.knots[-1])))
                    y = np.where(x_eval > self.knots[-1], y_trail, y_spl)
                else:
                    y = y_spl
            elif self.defined_on == "aLy":
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is None:
                    raise ValueError(f"No boundary condition found for profile '{prof}' at x=1.0")
                y_full = self._integrate_aLy(prof, x_eval, spline, bc_y['val'], bc_y['loc'])
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_y = te.get('x'), te.get('y')
                    if te_x is not None and te_y is not None and len(te_x) >= 2:
                        y_trail = np.interp(x_eval, te_x, te_y)
                    else:
                        y_trail = np.full_like(x_eval, self._integrate_aLy(prof, np.array([ghost_knot]), spline, bc_y['val'], bc_y['loc'])[0])
                    y = np.where(x_eval > self.knots[-1], y_trail, y_full)
                else:
                    y = y_full
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")

            out[prof] = np.clip(y, a_min=0, a_max=None)
        self.y = out
        return out

    def get_aLy(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute a/Ly(x) on x_eval.
        
        For roa (normalized): aLy = -(dy/dx) / y where x = r/a
        """
        out = {}
        for prof, prof_params in params.items():
            if self._use_fixed_tail():
                vals = np.array([prof_params[n] for n in self.param_names]) if isinstance(prof_params, dict) else np.asarray(prof_params)
                _, out[prof], _ = self._fixed_tail_reconstruct(prof, vals, x_eval)
                continue
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)

            # get aLy boundary condition to update vals if needed
            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            use_ghost = False
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                # Ghost point: add a point just beyond the last knot with the same value;
                # from there to roa=1 y is held constant (aLy=0 in that region).
                _eps = 1e-4
                knots = np.append(self.knots, self.knots[-1] + _eps)
                vals = np.append(vals, vals[-1])
                use_ghost = True

            ghost_knot = knots[-1]
            spline = self._make_spline(knots, vals, prof)

            if self.defined_on == "aLy":
                aLy_spl = spline(x_eval)
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_aLy = te.get('x'), te.get('aLy')
                    if te_x is not None and te_aLy is not None and len(te_x) >= 2:
                        aLy_trail = np.interp(x_eval, te_x, te_aLy)
                    else:
                        aLy_trail = np.zeros_like(x_eval)
                    aLy = np.where(x_eval > self.knots[-1], aLy_trail, aLy_spl)
                else:
                    aLy = aLy_spl
            elif self.defined_on == "y":
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_y = te.get('x'), te.get('y')
                    te_aLy_arr = te.get('aLy')
                    if te_x is not None and te_y is not None and len(te_x) >= 2:
                        y_trail = np.interp(x_eval, te_x, te_y)
                        if te_aLy_arr is not None:
                            aLy_trail = np.interp(x_eval, te_x, te_aLy_arr)
                        else:
                            te_spl = pchip(te_x, te_y)
                            y_t = te_spl(x_eval)
                            dy_t = te_spl.derivative(1)(x_eval)
                            y_t_safe = np.where(np.abs(y_t) < 1e-12, 1e-12, y_t)
                            aLy_trail = -dy_t / y_t_safe
                        y_spl = spline(x_eval)
                        dy_spl = spline.derivative(1)(x_eval)
                        y_spl_safe = np.where(np.abs(y_spl) < 1e-12, 1e-12, y_spl)
                        aLy_interior = -dy_spl / y_spl_safe
                        aLy = np.where(x_eval > self.knots[-1], aLy_trail, aLy_interior)
                    else:
                        y = spline(x_eval)
                        dy = spline.derivative(1)(x_eval)
                        y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
                        aLy = -dy / y_safe
                else:
                    y = spline(x_eval)
                    dy = spline.derivative(1)(x_eval)
                    # aLy = a/Ly = -(dy/dx)/y  for x = r/a dimensionless
                    y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
                    aLy = -dy / y_safe
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = np.clip(aLy, a_min=0, a_max=None)
        self.aLy = out
        return out

    def get_curvature(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² on x_eval.

        For roa (x = r/a dimensionless), aLy = -d(ln y)/dx, so:
            y' = -aLy * y
            y'' = -(aLy' * y + aLy * y') = -y * (aLy' - aLy²)
        """
        out = {}
        for prof, prof_params in params.items():
            if self._use_fixed_tail():
                vals = np.array([prof_params[n] for n in self.param_names]) if isinstance(prof_params, dict) else np.asarray(prof_params)
                _, _, out[prof] = self._fixed_tail_reconstruct(prof, vals, x_eval)
                continue
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)

            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            use_ghost = False
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                # Ghost point: add a point just beyond the last knot with the same value;
                # from there to roa=1 y is held constant (aLy=0, curvature=0 in that region).
                _eps = 1e-4
                knots = np.append(self.knots, self.knots[-1] + _eps)
                vals = np.append(vals, vals[-1])
                use_ghost = True

            ghost_knot = knots[-1]
            spline = self._make_spline(knots, vals, prof)
            te = self._trailing_edge.get(prof) or {}
            te_x, te_y = te.get('x'), te.get('y')
            _has_te = use_ghost and te_x is not None and te_y is not None and len(te_x) >= 3

            if self.defined_on == "y":
                curv_interior = spline.derivative(2)(x_eval)
                if use_ghost:
                    if _has_te:
                        curv_trail = pchip(te_x, te_y).derivative(2)(x_eval)
                    else:
                        curv_trail = np.zeros_like(x_eval)
                    curv = np.where(x_eval > self.knots[-1], curv_trail, curv_interior)
                else:
                    curv = curv_interior
            elif self.defined_on == "aLy":
                # Get boundary condition to integrate aLy → y
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is None:
                    raise ValueError(f"No boundary condition found for profile '{prof}' to compute curvature")

                aLy_spl = spline
                aLy_interior = aLy_spl(x_eval)
                aLy_prime_interior = aLy_spl.derivative(1)(x_eval)
                y_interior = self._integrate_aLy(prof, x_eval, aLy_spl, bc_y['val'], bc_y['loc'])

                if use_ghost:
                    if _has_te:
                        te_curv_spl = pchip(te_x, te_y)
                        y_trail = np.interp(x_eval, te_x, te_y)
                        aLy_trail_arr = te.get('aLy')
                        if aLy_trail_arr is not None:
                            aLy_trail = np.interp(x_eval, te_x, aLy_trail_arr)
                            aLy_prime_trail = pchip(te_x, aLy_trail_arr).derivative(1)(x_eval)
                        else:
                            aLy_trail = -te_curv_spl.derivative(1)(x_eval) / np.where(np.abs(te_curv_spl(x_eval)) < 1e-12, 1e-12, te_curv_spl(x_eval))
                            aLy_prime_trail = np.gradient(aLy_trail, x_eval)
                        aLy = np.where(x_eval > self.knots[-1], aLy_trail, aLy_interior)
                        aLy_prime = np.where(x_eval > self.knots[-1], aLy_prime_trail, aLy_prime_interior)
                        y = np.where(x_eval > self.knots[-1], y_trail, y_interior)
                    else:
                        aLy = aLy_interior
                        aLy_prime = aLy_prime_interior
                        y = y_interior
                else:
                    aLy = aLy_interior
                    aLy_prime = aLy_prime_interior
                    y = y_interior

                # Avoid division issues and NaN propagation
                y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)

                # y'' = -y * (aLy' - aLy²)  for x = r/a dimensionless
                curv = -y_safe * (aLy_prime - aLy**2)

                # Clean up any remaining NaN/inf values
                curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = curv
        self.curv = out
        return out


class Mtanh(ParameterBase):
    """Modified-tanh profile model with position-dependent width.

    Profile form:
        y(x) = A * (1 - tanh(u)) - m * (x - 1) + b
        u(x) = (x - c) / (Delta_0 * (1 + delta * (x - c)))

    Derivatives:
        dy/dx = -A * sech^2(u) / (Delta_0 * (1 + delta * (x - c))^2) - m
        a/Ly  = -(dy/dx) / y

    Solver-space parameters:
        [log_A, log_u1, delta, m]
    with:
        A  > 0       (pedestal amplitude)
        u1 = u(1) > 0
        delta        (width skewness, unconstrained)
        m            (linear slope, unconstrained)

    Boundary conditions y(1), aLy(1) are enforced algebraically (no root-find):
        R = y(1) * aLy(1) - m
        target = A * sech^2(u1) * u1 / R
        c solved from: delta*(1-c)^2 + (1-c) = target
        Delta_0 = (1-c) / (u1 * (1 + delta*(1-c)))
        b = y(1) - A * (1 - tanh(u1))
    """

    _WIDTH_FLOOR = 1e-6
    _Y_FLOOR = 1e-12

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.defined_on = "y"

        self.param_names = [
            'log_A',
            'log_u1',
            'delta',
            'm',
        ]

        self.n_params_per_profile = len(self.param_names)
        self.include_zero_grad_on_axis = False

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _solve_c(delta: float, target: float) -> Optional[float]:
        """Solve delta*(1-c)^2 + (1-c) = target for c in (0, 1)."""
        if abs(delta) < 1e-10:
            c = 1.0 - target
        else:
            discriminant = 1.0 + 4.0 * delta * target
            if discriminant < 0.0:
                return None
            root1 = (-1.0 + np.sqrt(discriminant)) / (2.0 * delta)
            root2 = (-1.0 - np.sqrt(discriminant)) / (2.0 * delta)
            valid = [r for r in (root1, root2) if r > 0.0]
            if not valid:
                return None
            one_minus_c = min(valid)
            c = 1.0 - one_minus_c
        if not (0.0 < c < 1.0):
            return None
        return float(c)

    def _to_physical(
        self,
        p: Union[np.ndarray, Dict[str, float]],
        y_bc: float,
        aLy_bc: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Map solver-space parameters to physical mtanh parameters.

        Parameters
        ----------
        p
            Solver-space parameter vector or dict:

                [log_A, log_u1, delta, m]

        y_bc
            Boundary value y(1).

        aLy_bc
            Boundary value a/Ly(1).

        Returns
        -------
        A, Delta_0, delta, m, c, b
            Fully reconstructed physical parameter set satisfying the
            boundary conditions exactly.
        """
        if isinstance(p, dict):
            vec = np.array([p['log_A'], p['log_u1'], p['delta'], p['m']], dtype=float)
        else:
            vec = np.asarray(p, dtype=float)

        log_A, log_u1, delta, m = [float(v) for v in vec[:4]]

        A  = float(np.exp(log_A))
        u1 = float(np.exp(log_u1))

        R     = float(y_bc * aLy_bc - m)
        R_eff = max(R, 1e-10)

        sech2  = 1.0 - np.tanh(u1) ** 2
        target = A * sech2 * u1 / R_eff

        c = self._solve_c(delta, target)
        if c is None:
            c = 0.95

        f1      = max(1.0 + delta * (1.0 - c), self._WIDTH_FLOOR)
        Delta_0 = (1.0 - c) / (u1 * f1)
        b       = float(y_bc - A * (1.0 - np.tanh(u1)))

        return A, Delta_0, delta, m, c, b

    def _physical_to_solver(
        self,
        A: float,
        Delta_0: float,
        delta: float,
        m: float,
        c: float,
        y_bc: float = None,
        aLy_bc: float = None,
    ) -> Dict[str, float]:
        """Map physical mtanh parameters to solver space.

        Parameters
        ----------
        A, Delta_0, delta, m, c
            Physical mtanh parameters.

        Returns
        -------
        dict
            Solver-space parameter dictionary {log_A, log_u1, delta, m}.
        """
        s  = 1.0 - c
        f1 = max(1.0 + delta * s, self._WIDTH_FLOOR)
        u1 = s / (Delta_0 * f1)

        return {
            'log_A':  float(np.log(A)),
            'log_u1': float(np.log(u1)),
            'delta':  float(delta),
            'm':      float(m),
        }

    def _resolve_bcs(self, prof: str) -> Tuple[float, float]:
        """Return (y_bc, aLy_bc) at x=1 for the given profile."""
        bc_y   = self.get_nearest_bc(prof, 1.0)
        bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
        if bc_y is None:
            raise ValueError(f"No y BC found for profile '{prof}'")
        if bc_aLy is None:
            raise ValueError(f"No aLy BC found for profile '{prof}'")
        return float(bc_y['val']), float(bc_aLy['val'])

    def _w_eval(self, x: np.ndarray, Delta_0: float, delta: float, c: float) -> np.ndarray:
        """Spatially varying width w(x) = Delta_0*(1+delta*(x-c)), floored at 1e-6."""
        return np.maximum(Delta_0 * (1.0 + delta * (x - c)), self._WIDTH_FLOOR)

    def _y_eval(self, x: np.ndarray, A: float, Delta_0: float, delta: float,
                m: float, c: float, b: float) -> np.ndarray:
        w  = self._w_eval(x, Delta_0, delta, c)
        u  = (x - c) / w
        return A * (-np.tanh(u) + 1.0) - m * (x - 1.0) + b

    def _dydx_eval(self, x: np.ndarray, A: float, Delta_0: float, delta: float,
                   m: float, c: float) -> np.ndarray:
        """dy/dx = -A * sech^2(u) / (Delta_0*(1+delta*(x-c))^2) - m"""
        f     = np.maximum(1.0 + delta * (x - c), self._WIDTH_FLOOR)
        u     = (x - c) / (Delta_0 * f)
        sech2 = 1.0 - np.tanh(u) ** 2
        return -A * sech2 / (Delta_0 * f ** 2) - m

    def _physical_for_profile(
        self,
        prof: str,
        p: Union[np.ndarray, Dict[str, float]],
    ) -> Tuple[float, float, float, float, float, float]:
        """Resolve BCs for a profile and map solver-space params to physical params."""
        y_bc, aLy_bc = self._resolve_bcs(prof)
        return self._to_physical(p, y_bc, aLy_bc)

    # ─────────────────────────────────────────────────────────
    # ParameterBase interface
    # ─────────────────────────────────────────────────────────

    def parameterize(
        self,
        state: Any,
        bc_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Fit mtanh parameters directly in solver space {log_A, log_u1, delta, m}."""
        self.build_bcs(bc_dict)

        params: Dict[str, Dict[str, float]] = {}
        params_std: Dict[str, Dict[str, float]] = {}

        x_data = np.asarray(getattr(state, 'roa')).flatten()

        # Keep bounds aligned with SplineMtanh's feasible search box.
        bounds_lo = np.array([np.log(1e-3), np.log(1e-3), -50.0, 0.0], dtype=float)
        bounds_hi = np.array([np.log(5.0),  np.log(10.0), 50.0, 10.0], dtype=float)
        default_p0 = np.array([np.log(0.5), np.log(1.0), 0.0, 1.0], dtype=float)

        for prof in self.predicted_profiles:
            y_data = np.asarray(getattr(state, prof)).flatten()
            y_bc, aLy_bc = self._resolve_bcs(prof)

            y_scale = np.maximum(np.abs(y_data), 1e-3)

            def _residuals(p: np.ndarray) -> np.ndarray:
                log_A, log_u1, delta, m = [float(v) for v in np.asarray(p, dtype=float)[:4]]

                # Soft penalties keep optimizer inside physically meaningful region.
                R = y_bc * aLy_bc - m
                v_r = max(0.0, 1e-10 - R)

                A = np.exp(log_A)
                u1 = np.exp(log_u1)
                sech2 = 1.0 - np.tanh(u1) ** 2
                target = A * sech2 * u1 / max(R, 1e-10)
                c = self._solve_c(delta, target)
                root_violation = 0.0 if c is not None else 1.0
                if c is None:
                    c = 0.95

                f1 = max(1.0 + delta * (1.0 - c), self._WIDTH_FLOOR)
                D0 = (1.0 - c) / (u1 * f1)
                v_d0 = max(0.0, 1e-2 - D0) + max(0.0, D0 - 1.0)

                A_, D0_, delta_, m_, c_, b_ = self._to_physical(
                    np.array([log_A, log_u1, delta, m], dtype=float),
                    y_bc,
                    aLy_bc,
                )
                y_model = self._y_eval(x_data, A_, D0_, delta_, m_, c_, b_)
                res_data = (y_model - y_data) / y_scale

                penalty = np.sqrt(1e3) * np.array([v_r, root_violation, v_d0], dtype=float)
                return np.concatenate([res_data, penalty])

            starts = [
                default_p0,
                np.array([np.log(1e-3), np.log(1e-2), 0.0, 1.0], dtype=float),
                np.array([np.log(1.0), np.log(0.05), 0.0, 1.0], dtype=float),
            ]

            best = None
            best_cost = np.inf
            for p0 in starts:
                p0_clip = np.minimum(np.maximum(p0, bounds_lo), bounds_hi)
                try:
                    fit = least_squares(
                        _residuals,
                        p0_clip,
                        method='trf',
                        bounds=(bounds_lo, bounds_hi),
                        ftol=1e-3,
                        gtol=1e-12,
                        xtol=1e-3,
                        diff_step=5e-2,
                        x_scale='jac',
                        max_nfev=100,
                    )
                except Exception:
                    continue

                if fit.cost < best_cost:
                    best = np.asarray(fit.x, dtype=float)
                    best_cost = float(fit.cost)

            popt = best if best is not None else default_p0

            params[prof] = {
                'log_A': float(popt[0]),
                'log_u1': float(popt[1]),
                'delta': float(popt[2]),
                'm': float(popt[3]),
            }

            params_std[prof] = {
                k: abs(v) * self.sigma
                for k, v in params[prof].items()
            }

        self.params = params
        self.params_std = params_std

        return params, params_std


    def get_y(
        self,
        params: Dict[str, Dict[str, float]],
        x_eval: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate y(x) for each profile."""
        x_eval = np.asarray(x_eval)

        out: Dict[str, np.ndarray] = {}

        for prof, p in params.items():
            A, Delta_0, delta, m, c, b = self._physical_for_profile(prof, p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            R = y_bc * aLy_bc - m

            if m <= 0 or R <= 0:
                out[prof] = np.full_like(x_eval, fill_value=1e3)  # large penalty for unphysical parameters
            else:
                out[prof] = np.clip(
                    self._y_eval(x_eval, A, Delta_0, delta, m, c, b),
                    0.0,
                    None,
                )

        self.y = out
        return out

    def get_aLy(
        self,
        params: Dict[str, Dict[str, float]],
        x_eval: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate a/Ly(x) = -(dy/dx)/y for each profile."""
        x_eval = np.asarray(x_eval)

        out: Dict[str, np.ndarray] = {}

        for prof, p in params.items():
            A, Delta_0, delta, m, c, b = self._physical_for_profile(prof, p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            R = y_bc * aLy_bc - m

            if m <= 0 or R <= 0:
                out[prof] = np.full_like(x_eval, fill_value=1e3)  # large penalty for unphysical parameters
            else:
                y = self._y_eval(x_eval, A, Delta_0, delta, m, c, b)
                dydx = self._dydx_eval(x_eval, A, Delta_0, delta, m, c)
                y_safe = np.where(np.abs(y) < self._Y_FLOOR, self._Y_FLOOR, y)

                out[prof] = np.clip(-dydx / y_safe, 0.0, None)

        self.aLy = out
        return out

    def get_curvature(
        self,
        params: Dict[str, Dict[str, float]],
        x_eval: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate d²y/dx² on x_eval for each profile.

        With:

            w(x) = Delta_0 * (1 + delta*(x-c))
            f(x) = 1 + delta*(x-c)
            u(x) = (x-c) / (Delta_0*f(x))

        the curvature is:

            d²y/dx² =
                (2*A*sech²(u)/Delta_0)
                * [tanh(u)/(Delta_0*f^4) + delta/f^3]
        """
        x_eval = np.asarray(x_eval)

        out: Dict[str, np.ndarray] = {}

        for prof, p in params.items():
            A, Delta_0, delta, m, c, _ = self._physical_for_profile(prof, p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            R = y_bc * aLy_bc - m

            if m <= 0 or R <= 0:
                out[prof] = np.full_like(x_eval, fill_value=1e3)  # large penalty for unphysical parameters
            else:
                f = np.maximum(1.0 + delta * (x_eval - c), self._WIDTH_FLOOR)

                u = (x_eval - c) / (Delta_0 * f)

                tanh_ = np.tanh(u)
                sech2 = 1.0 - tanh_**2

                out[prof] = (
                    (2.0 * A * sech2 / Delta_0)
                    * (
                        tanh_ / (Delta_0 * f**4)
                        + delta / f**3
                    )
                )

        self.curv = out
        return out


# -------------------------------------------------------------------------
# Analytic-Jacobian pedestal forward model (value + d(aLy)/d[s,c,w1,r]).
#
# Convention (sign-flipped w.r.t. SplineMtanh):
#     w(x) = w1 * exp(-r*(x - c)/(1 - x0))      so r >= 0  <=>  width(x<c) > width(x>c)
#     y(x) = A*(1 - tanh(u)) - m*(x - 1) + b,   u = (x - c)/w(x)
# BCs y(1)=y1, aLy(1)=aLy1 are eliminated analytically; A = s*A_max(c,w1,r)
# with s in (0,1) so that m = y1*aLy1*(1 - s) >= 0 *by construction* (no penalty).
# Source: user-provided gen_exprs.eval_all (symbolically generated & tested).
# -------------------------------------------------------------------------

def _pedestal_eval_all(x, s, c, w1, r, x0, y1, aLy1):
    numpy = np
    x = np.asarray(x, dtype=float)
    x1 = x - 1
    x2 = c - 1
    x3 = x0 - 1
    x4 = x3**(-1.0)
    x5 = r*x4
    x6 = x2*x5
    x7 = w1**(-1.0)
    x8 = x2*x7
    x9 = x8*numpy.exp(x6)
    x10 = numpy.cosh(x9)
    x11 = numpy.exp(-x6)
    x12 = x6 + 1
    x13 = x12**(-1.0)
    x14 = x11*x13
    x15 = x10**2*x14
    x16 = -x2
    x17 = -1/x3
    x18 = r*x17
    x19 = x16*x18
    x20 = numpy.exp(x19)
    x21 = x16*x20
    x22 = numpy.tanh(x21*x7)
    x23 = x22 - 1
    x24 = w1*x23
    x25 = x15*x24
    x26 = aLy1*s
    x27 = c - x
    x28 = -x27
    x29 = x18*x28
    x30 = numpy.exp(x29)
    x31 = numpy.tanh(x28*x30*x7)
    x32 = x31 - 1
    x33 = w1*x15*x32
    x34 = aLy1*x1*(s - 1) + x25*x26 - x26*x33 + 1
    x35 = x27*x5
    x36 = x35 + 1
    x37 = numpy.exp(x35)
    x38 = x37*x7
    x39 = x27*x38
    x40 = numpy.cosh(x39)
    x41 = x40**(-2.0)
    x42 = x37*x41
    x43 = x15*x36*x42
    x44 = -aLy1/x34
    x45 = x44*(s*x43 - s + 1)
    x46 = numpy.sinh(x9)
    x47 = 2*x46
    x48 = x36*x47
    x49 = x48*x7
    x50 = x10*x11
    x51 = x50*(x35 + 2)
    x52 = x13*x5
    x53 = x6 + 2
    x54 = x50*x53
    x55 = 2*x50*numpy.sinh(x39)/x40
    x56 = x38*x55
    x57 = x19 + 1
    x58 = x31**2 - 1
    x59 = x30*(x29 + 1)
    x60 = x22**2 - 1
    x61 = s*x10*x44
    x62 = x47*x8
    x63 = x13*x61
    x64 = x2**2
    x65 = x27**2
    x66 = numpy.tanh(x9)
    x67 = x66 + 1
    x68 = numpy.tanh(x39)
    x69 = x68 + 1
    y_val = x34*y1
    aLy_val = -x45
    d_s = x44*(-x43 - x45*(x1 + x25 - x33) + 1)
    d_c = -x61*(x14*x45*(r*w1*x10*x13*x32*x4*x53 + x10*x20*x57*x60 - x10*x24*x52*x53 - x10*x58*x59 + 2*x20*x23*x46*x57 - x20*x32*x47*x57) + x42*(-x13*x36**2*x56 + x49 + x51*x52 - x36*x5*x54/x12**2))
    d_w1 = x63*(x41*x7*(-x27*x36*x55*x7*numpy.exp(2*x35) - x36*x37*x50 + x37*x48*x8 + x50*x59) - x45*(-x10*x60*x8 + x23*x50 - x23*x62 - x32*x50 + x32*x62 + x39*x50*x58))
    d_r = x63*(x4*x45*(w1*x10*x11*x13*x2*x53*x69 - w1*x13*x2*x54*x67 + x10*x11*x37*x65*(x68**2 - 1) - x10*x64*(x66**2 - 1) + 2*x46*x64*x67 - x47*x64*x69) + x42*(x10*x13*x17*x21*x36*(x19 + 2)*numpy.exp(-2*x6) - x27*x4*x51 + x36*x4*x56*x65 - x4*x49*x64))
    return y_val, aLy_val, d_s, d_c, d_w1, d_r


class SplineMtanhAnalytic(ParameterBase):
    """Analytic-Jacobian mtanh + linear-corrector aLy parameterization (single class).

    This is the sole mtanh-spline parameterizer: the former ``SplineMtanh`` base is
    merged in (``__init__`` knot/fit setup, ``parameterize`` and the batched
    ``update`` now live here; ``SplineMtanh`` survives only as a module-level alias).
    Knot-local ``aLy*`` design variables are reconstructed as a cancellation-free
    mtanh backbone plus a corrector that pins the interior knots; the interior fit
    solves ``theta = [s, c, w1, rho]`` with an analytic Jacobian:

    * **Speed** -- ``least_squares(method='trf')`` is handed the *exact* Jacobian
      ``d(aLy)/d[s,c,w1,r]`` (:func:`_pedestal_eval_all`), removing the
      finite-difference sweeps (``diff_step``) that dominated the base fitter.
    * **Robustness** -- amplitude is parameterized as ``A = s*A_max(c,w1,r)`` with
      ``s in (0,1)``, so the background slope ``m = y1*aLy1*(1-s) >= 0`` holds *by
      construction*. The base class's smooth ``m in [0.01,10]`` feasibility
      penalties (and their weight tuning) are gone.
    * **Fit quality / determinism** -- a weak ``lam_r * r`` L2 residual restores an
      effectively unique basin when ``#knots < 4`` (the base-function fit is
      rank-deficient there), and ``c`` is seeded from a rational sigmoid in
      ``R = aLy1/aLy(x_last)``.

    The fit is a deterministic pure function of (knots, BCs): a fixed c-seed sweep,
    no resolve cache and no history warm-start.

    The resolved shape is carried as the *finite* reconstruction tuple
    ``(g, Delta_0, delta, m, c, y_bc)`` (:meth:`_theta_to_phys`): the tanh's share of
    the LCFS gradient ``g = s*y1*aLy1`` and the linear background ``m = (1-s)*y1*aLy1``
    (>= 0 by construction), never the exploding explicit ``A``/``b``. The
    ``_y_mtanh``/``_dydx_mtanh``/``_d2ydx2_mtanh`` do the cancellation-free
    reconstruction; the corrector ``Delta = B @ resid`` (:meth:`_corrector_ops`) is a
    constant linear operator (piecewise-linear, monotone-safe) that pins the interior
    knots and carries the aLy-knot-DV gradient.
    """

    # Solver space here is [s, c, w1, rho] (NOT [log_A, c, log_w1, r]).
    # The 4th component is rho in [0,1): the width-asymmetry as a FRACTION of the
    # u'(x)>0 feasibility limit,  r = rho * _R_SAFETY * (1-x0)/(c-x0).  This makes
    # the monotone-reconstruction constraint (the parent class's smooth "u'>0 on
    # [x0,1]" penalty, C1) STRUCTURAL: the foot can no longer turn over (dy/dx>0
    # inside x_turn=c-(1-x0)/r) no matter how the knots move.  rho=0 is the flat,
    # non-peaking edge (width ~constant); rho->1 is the steepest feasible foot
    # (narrow width at the LCFS -> sub-separatrix aLy peak).  The old raw-r bound
    # (0, ln10) had no c-coupling, so ~5% of wide knot draws overflowed it.
    _Y_FLOOR = 1e-12
    _PENALTY = 1e3
    _S_BOUNDS = (1e-3, 0.999)
    # Foot width w1 = w(x0). Physical range: below ~0.01 the tanh foot is a razor-thin
    # spike (a flat interior with a steep last-knot jump used to collapse w1 to ~0.006 and
    # send the sub-separatrix aLy to ~10x its neighbours); above ~0.1 the "foot" is wider
    # than the pedestal. Bounding [0.01, 0.1] keeps every reconstruction physical.
    _W1_BOUNDS = (0.01, 0.1)
    _RHO_BOUNDS = (0.0, 0.999)
    _R_SAFETY = 0.999                        # rho=1 hits u'(x0)=0 exactly; stay just inside
    _C_BOUNDS = (0.9, 1.1)
    _DENOM_FLOOR = 1e-9

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        # --- knot/domain setup (merged from the former SplineMtanh base) ---
        self.include_zero_grad_on_axis = False
        self.knots = np.array(options.get('knots', []) or [], dtype=float)
        self.defined_on = str(options.get('defined_on', 'aLy'))
        if self.defined_on != 'aLy':
            raise ValueError("SplineMtanhAnalytic only supports defined_on='aLy'")
        self.x0 = float(options.get('x0', 0.85))
        # TRF fit controls (analytic Jacobian; deterministic fixed-seed sweep, no cache).
        self.fit_max_nfev = int(options.get('fit_max_nfev', 100))
        self.fit_trf_ftol = float(options.get('fit_trf_ftol', 1e-3))
        self.fit_trf_gtol = float(options.get('fit_trf_gtol', 1e-12))
        self.fit_trf_xtol = float(options.get('fit_trf_xtol', 1e-3))
        self.fit_max_rel_error = float(options.get('fit_max_rel_error', 2e-2))
        self.fit_warm_accept_rel = float(options.get('fit_warm_accept_rel', 0.30))
        # Hand-rolled LM fit controls (replaces scipy least_squares for O(1 ms) fits).
        self.fit_lm_iters = int(options.get('fit_lm_iters', 12))
        self.fit_lm_damping_tries = int(options.get('fit_lm_damping_tries', 4))
        self.fit_lm_ctol = float(options.get('fit_lm_ctol', 1e-3))   # rel-cost stop
        self.fit_lm_gtol = float(options.get('fit_lm_gtol', 1e-4))   # KKT-grad stop
        self.fit_ridge_cost = float(options.get('fit_ridge_cost', 0.05))
        # Per-profile previous fit, reused as a warm-start seed (speed only; the fit
        # still runs to convergence so the result stays deterministic).
        self._last_theta_guess: Dict[str, np.ndarray] = {}
        self.param_names = [f'aLy{i}' for i in range(len(self.knots))]
        self.n_params_per_profile = len(self.param_names)
        # --- analytic-fit specifics ---
        self.lam_r = float(options.get('lam_r', 0.3))
        # c-seed sigmoid in R = aLy1/aLy(x_last):
        #   c(R) = c_lo + (c_hi-c_lo) * R^p / (R^p + a),  a chosen so c(1)=c_mid.
        # Defaults recalibrated against a 3-knot pedestal/shortfall benchmark:
        # c_hi 1.10->1.15 (headroom for the c>1 "tail" regime of near-flat edges,
        # where c_fit reaches ~1.10) and p 1->2 (flat near R~1, steeper for high R)
        # halved the mean |c_seed - c_fit| (0.042 -> 0.019). c is only a seed (the
        # analytic-Jacobian fit corrects it), so this buys warm-accept/convergence
        # speed, not final accuracy.
        self.c_seed_lo = float(options.get('c_seed_lo', 0.9))
        self.c_seed_hi = float(options.get('c_seed_hi', 1.15))
        self.c_seed_mid = float(options.get('c_seed_mid', 0.95))
        self.c_seed_p = float(options.get('c_seed_p', 2.0))
        # Interior-knot residual corrector. ON by default: the analytic mtanh is not
        # guaranteed to hit the requested aLy knots (up to ~10-18% off on non-mtanh
        # interior shapes), so Delta(x) pins them exactly. Set use_corrector=False to
        # report the BARE mtanh backbone (knots then only approximately matched via the
        # fit theta) -- the tail-peak guard + w1 bound still apply to the backbone, so
        # the reported profile stays physical. NB with the corrector off the aLy-knot
        # DVs enter only through the (detached) fit, so the straight-through torch DV
        # gradient vanishes (the solver's FD-through-parameterizer path is unaffected).
        self.use_corrector = bool(options.get('use_corrector', True))
        # Fixed defaults for the non-c solver components.
        self.s0 = float(options.get('s0', 0.5))
        self.w1_0 = float(options.get('w1_0', 0.02))
        # Runaway peak guard. A flat interior with a steep last-knot/LCFS jump can make
        # the mtanh backbone overshoot to many x its neighbours (e.g. aLne ~100 with knots
        # ~20) even at the physical w1 floor. When the fitted backbone peak exceeds
        # fit_peak_kappa * max(aLy(knots), aLy(1)) (see _peak_cap), a single penalized LM
        # re-fit (soft ReLU on the excess at _peak_x) pulls it back under the cap. Gated on
        # detection -> real, non-runaway fits never re-fit (byte-identical) and pay only one
        # cheap peak evaluation. The corrector still pins the interior knots, so the re-fit
        # only reshapes the over-peaked region.
        # The sample grid spans the WHOLE parameterizer domain [x0,1], not just the
        # near-LCFS foot: a solver-driven peak sitting on an interior knot (aLy(0.97) >>
        # aLy_lcfs is legitimate, if not pretty) was invisible to a foot-only window, and
        # so was any UQ excursion that moved the peak inboard of it.
        self._peak_penalty_on = bool(options.get('fit_peak_penalty', True))
        self.fit_peak_kappa = float(options.get('fit_peak_kappa', 2.0))
        self.fit_peak_weight = float(options.get('fit_peak_weight', 5.0))
        self.peak_proj_iters = int(options.get('fit_peak_proj_iters', 12))
        # Also reject the dip-then-peak (turnover) shape when projecting a UQ theta
        # offset. Applies to the perturbation only -- a nominal fit that already dips is
        # left alone (see _apply_theta_offset).
        self._unimodal_proj_on = bool(options.get('fit_unimodal_proj', True))
        self._peak_x = np.unique(np.concatenate([
            np.linspace(self.x0, 0.96, 16, endpoint=False),
            np.linspace(0.96, 1.0 - 1e-6, 48),
        ]))
        # Theta stashed by the most recent _resolve, keyed by profile; the torch
        # overlay reads s from here to keep A live in the BCs (see below).
        self._current_theta: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Solver-space bounds / seeds
    # ------------------------------------------------------------------
    def _theta_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([self._S_BOUNDS[0], self._C_BOUNDS[0], self._W1_BOUNDS[0], self._RHO_BOUNDS[0]], dtype=float)
        hi = np.array([self._S_BOUNDS[1], self._C_BOUNDS[1], self._W1_BOUNDS[1], self._RHO_BOUNDS[1]], dtype=float)
        return lo, hi

    def _rho_c_to_r(self, rho: float, c: float) -> Tuple[float, float, float]:
        """Map solver rho (+ c) to the raw width-asymmetry r, with its partials.

        r = rho * _R_SAFETY * (1-x0)/(c-x0). Returns (r, dr/drho, dr/dc); the c
        partial is needed by the fit Jacobian because c enters r as well as the
        pedestal shape. The (c-x0) floor guards c -> x0 (bounded away at 0.9).
        """
        cx = max(c - self.x0, 1e-6)
        g = self._R_SAFETY * (1.0 - self.x0) / cx
        r = rho * g
        return r, g, -r / cx

    def _seed_c(self, R: float) -> float:
        p = self.c_seed_p
        frac_mid = (self.c_seed_mid - self.c_seed_lo) / (self.c_seed_hi - self.c_seed_lo)
        a = 1.0 / frac_mid - 1.0   # so c(R=1) = c_mid for any p
        Rp = max(float(R), 0.0) ** p
        frac = Rp / (Rp + a)
        return self.c_seed_lo + (self.c_seed_hi - self.c_seed_lo) * frac

    def _default_theta0(self, aLy_k: np.ndarray, aLy_bc: float) -> np.ndarray:
        last = float(aLy_k[-1]) if aLy_k.size else 1.0
        R = aLy_bc / last if abs(last) > 1e-12 else 1.0
        c0 = self._seed_c(R)
        rho0 = 0.5   # mid of [0,1): moderate foot asymmetry, off both bounds
        return np.array([self.s0, c0, self.w1_0, rho0], dtype=float)

    # ------------------------------------------------------------------
    # theta = [s, c, w1, r]  ->  base-class physical tuple (A,D0,delta,m,c,b)
    # ------------------------------------------------------------------
    def _theta_to_phys(self, theta, y_bc: float, aLy_bc: float):
        """Map solver theta -> reconstruction tuple ``(g, D0, delta, m, c, y_bc)``.

        NOTE: slots 0 and 5 are *not* the base class's free amplitude ``A`` and
        offset ``b``. This subclass never materializes those: for the "c>=1 tail"
        shapes the fit legitimately picks, ``A = s*y1*aLy1/(sech^2(u1)*u1')``
        blows up (u1 large -> sech^2(u1)->0 -> A ~ 1e10), and reconstructing
        ``y = A*(1-tanh) + b`` with ``b = y1 - A*(1-tanh(u1)) ~ -A`` then loses the
        LCFS gradient to catastrophic cancellation (aLy(1) collapses instead of
        equalling aLy_bc). Instead we carry the two *finite* physical quantities

            g = s * y1 * aLy1        (tanh's share of the LCFS gradient, O(1))
            m = (1 - s) * y1 * aLy1  (linear background's share, >= 0)

        and reconstruct through the cancellation-free shape ``Phi`` in the
        overridden ``_y_mtanh``/``_dydx_mtanh``/``_d2ydx2_mtanh``. The slots line
        up positionally with the base tuple so the inherited
        ``_evaluate_once``/``get_y``/``get_aLy`` machinery needs no change (those
        reconstruction methods read slot 0 as ``g`` and slot 5 as ``y_bc``).
        """
        s, c, w1, rho = (float(v) for v in np.asarray(theta, dtype=float).reshape(-1)[:4])
        r, _, _ = self._rho_c_to_r(rho, c)   # rho (feasible fraction) -> raw r
        # base-class width w(x) = D0*exp(delta*(x-c)) with D0 = w(c) = w1;
        # matching -r*(x-c)/(1-x0) requires delta = -r/(1-x0).
        delta = -r / (1.0 - self.x0)
        D0 = w1
        g = s * y_bc * aLy_bc
        m = (1.0 - s) * y_bc * aLy_bc   # >= 0 by construction (s in (0,1))
        return (g, D0, delta, m, c, y_bc)

    # ------------------------------------------------------------------
    # Cancellation-free reconstruction (overrides the base A,b tuple form).
    #
    # The tanh contribution is carried as g*Phi(x), where Phi is the step
    # normalized to unit LCFS gradient. Every exponentially-small tanh-tail
    # factor enters only as the RATIO cosh(u1)/cosh(u) (in log-space) or the
    # DIFFERENCE sinh(u1-u) -- never as A*(tiny) or A - A -- so y(1)=y_bc and
    # aLy(1)=aLy_bc hold to machine precision for every (c, w1), including the
    # c>=1 narrow-width regime that overflowed the old explicit-A form.
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh(z: np.ndarray) -> np.ndarray:
        a = np.abs(np.asarray(z, dtype=float))
        return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)

    def _shape_terms(self, x, D0, delta, c):
        """Return (Phi, dPhi/dx, d2Phi/dx2) for the unit-LCFS-gradient tanh step."""
        x = np.asarray(x, dtype=float)
        w = D0 * np.exp(delta * (x - c))
        u = (x - c) / w
        up = (1.0 - delta * (x - c)) / w
        ud = -delta * (2.0 - delta * (x - c)) / w
        w1_ = D0 * np.exp(delta * (1.0 - c))            # w(1)
        u1 = (1.0 - c) / w1_
        u1p = (1.0 - delta * (1.0 - c)) / w1_
        if abs(u1p) < self._DENOM_FLOOR:                # genuine turning-point guard only
            u1p = self._DENOM_FLOOR if u1p >= 0 else -self._DENOM_FLOOR
        r_ch = np.exp(self._logcosh(u1) - self._logcosh(u))   # cosh(u1)/cosh(u)
        tanh_u = np.tanh(u)
        Phi = np.sinh(u1 - u) / u1p * r_ch
        dPhi = -(up / u1p) * r_ch ** 2
        d2Phi = -(r_ch ** 2) * (ud - 2.0 * tanh_u * up ** 2) / u1p
        return Phi, dPhi, d2Phi

    def _y_mtanh(self, x, g, D0, delta, m, c, y_bc):
        Phi, _, _ = self._shape_terms(x, D0, delta, c)
        return y_bc + g * Phi - m * (np.asarray(x, dtype=float) - 1.0)

    def _dydx_mtanh(self, x, g, D0, delta, m, c):
        _, dPhi, _ = self._shape_terms(x, D0, delta, c)
        return g * dPhi - m

    def _d2ydx2_mtanh(self, x, g, D0, delta, c):
        _, _, d2Phi = self._shape_terms(x, D0, delta, c)
        return g * d2Phi

    # ------------------------------------------------------------------
    # Corrector as a CONSTANT LINEAR operator (replaces the per-call scipy pchip).
    #
    # Delta(x) is the PIECEWISE-LINEAR interpolant through [0, knots, 1] with values
    # [0, resid, 0]; for fixed knots that map is linear in ``resid``:
    #     Delta(x) = B(x) @ resid,   Delta'(x) = Bp(x) @ resid
    # with B, Bp constant per eval grid (built once, cached).  Three payoffs:
    #   * speed -- no scipy pchip construction per resolve (it was ~55% of the hot
    #     path); the reconstruction is now constant-matrix matmuls.
    #   * robustness -- piecewise-linear CANNOT overshoot between knots (a natural
    #     cubic does, and under large knot perturbations that broke the monotone
    #     reconstruction), so the corrector never manufactures a non-monotone foot.
    #   * autograd -- being linear in ``resid = target - model`` (target = the aLy
    #     knot DVs), Delta is differentiable w.r.t. every knot DV, so the corrected
    #     y/aLy carry the DV gradient (the torch path reuses the same B, Bp).
    # (Trade-off vs pchip: aLy is C0 with kinks at the knots rather than C1; y stays
    #  C1. The knots are sparse and the residual is small on a good fit, so the tail
    #  the solver integrates is unaffected.)
    # ------------------------------------------------------------------
    def _corrector_ops(self, x):
        x = np.asarray(x, dtype=float)
        knots = np.asarray(self.knots, dtype=float)
        n = knots.size
        cache = self.__dict__.setdefault("_corr_cache", {})
        ck = (n, x.size, float(x[0]), float(x[-1]))
        hit = cache.get(ck)
        if hit is not None and hit[0].shape == (x.size, n):
            return hit
        xs = np.concatenate(([0.0], knots, [1.0]))          # abscissae, resid pinned 0 at 0,1
        seg = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, xs.size - 2)
        t = (x - xs[seg]) / (xs[seg + 1] - xs[seg])          # local coord in each segment
        B = np.zeros((x.size, n)); Bp = np.zeros((x.size, n))
        for i in range(n):
            j = i + 1                                        # knot i sits at xs[j]
            left = seg == (j - 1); right = seg == j          # segments touching this knot
            B[left, i] = t[left]; B[right, i] = 1.0 - t[right]
            Bp[left, i] = 1.0 / (xs[j] - xs[j - 1])
            Bp[right, i] = -1.0 / (xs[j + 1] - xs[j])
        cache[ck] = (B, Bp)
        return B, Bp

    # ------------------------------------------------------------------
    # Single-profile corrected reconstruction (y, aLy, curvature).
    #
    # One source of truth for the interior residual corrector so the standalone
    # getters and the batched update()/_evaluate_once path report the *same*
    # profile. The mtanh base is analytic; the corrector Delta(x)=corr(x) forces
    # exact knot pass-through (the analytic form is NOT guaranteed to hit the
    # requested aLy knots -- on non-mtanh interior shapes the constrained fit
    # lands up to ~10-18% off, which is exactly what Delta cancels).
    #
    # Corrected profile y = y_base * exp(int_x^1 Delta dx'), so
    #     aLy = aLy_base + Delta
    #     y'' = e^phase * [ y_base'' - 2*Delta*y_base' + y_base*(Delta^2 - Delta') ]
    # (product rule on y_base * e^phase with phase' = -Delta), all analytic and
    # BC-preserving (Delta vanishes at the axis and the LCFS).
    # ------------------------------------------------------------------
    def _corrected_profile(self, prof, pv, x):
        x = np.asarray(x, dtype=float)
        phys = self._resolve(prof, pv)
        if phys is None:
            return None
        A, D0, delta, m, c, b = phys
        # The cancellation-free mtanh (r_ch = cosh(u1)/cosh(u), sinh(u1-u)) is
        # finite across the physical fit domain, but a large theta perturbation
        # (edge-UQ scan, >~5 sigma) can collapse the foot width so that u1 blows
        # up and the reconstruction overflows to +-inf/NaN. Detect that here and
        # bail to None -- the caller then degrades to the Akima spline fallback
        # (update path) or a penalty vector (getters) rather than propagating
        # NaN into the corrector's pchip (which used to raise deep in scipy).
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y_base = np.clip(self._y_mtanh(x, A, D0, delta, m, c, b), 0.0, None)
            dydx_base = self._dydx_mtanh(x, A, D0, delta, m, c)
            d2_base = self._d2ydx2_mtanh(x, A, D0, delta, c)
        if not (np.all(np.isfinite(y_base)) and np.all(np.isfinite(dydx_base))
                and np.all(np.isfinite(d2_base))):
            self._warn_nonfinite(prof, "mtanh base reconstruction")
            return None
        y_safe = np.where(np.abs(y_base) < self._Y_FLOOR, self._Y_FLOOR, y_base)
        aLy_base = np.clip(-dydx_base / y_safe, 0.0, None)

        resid = self._aLy_correction(prof, pv, A, D0, delta, m, c, b)
        if resid is None:
            return y_base, aLy_base, d2_base

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            # Reported profile uses a SMOOTH, monotone-preserving pchip corrector:
            # piecewise-linear left visible kinks in aLy at the knots, and the natural/
            # clamped cubics overshoot badly under large knot perturbations (~100/200
            # non-monotone vs pchip's ~14). pchip is both smooth (C1) and robust. The
            # torch AUTOGRAD path instead uses the constant LINEAR operator
            # (_corrector_ops) for differentiability, decoupled here via straight-through.
            knots = np.asarray(self.knots, dtype=float)
            xs = np.concatenate(([0.0], knots, [1.0]))
            ds = np.concatenate(([0.0], resid, [0.0]))
            xs, idx = np.unique(xs, return_index=True)
            ds = ds[idx]
            corr = pchip(xs, ds, extrapolate=True)
            delta_aLy = corr(x)
            ddelta_aLy = corr.derivative()(x)
            cum = cumulative_trapezoid(delta_aLy, x, initial=0.0)
            phase = np.clip(cum[-1] - cum, -10.0, 10.0)   # int_x^1 Delta dx'
            efac = np.exp(phase)
            y_corr = np.clip(y_base * efac, 0.0, None)
            aLy_corr = np.clip(aLy_base + delta_aLy, 0.0, None)
            curv = efac * (
                d2_base - 2.0 * delta_aLy * dydx_base
                + y_base * (delta_aLy ** 2 - ddelta_aLy)
            )
        if not (np.all(np.isfinite(y_corr)) and np.all(np.isfinite(aLy_corr))
                and np.all(np.isfinite(curv))):
            self._warn_nonfinite(prof, "corrected reconstruction")
            return None
        return y_corr, aLy_corr, curv

    def _warn_nonfinite(self, prof, where):
        """Trace a non-finite reconstruction once per channel (typically a
        large edge-UQ theta perturbation collapsing the foot width). Not fatal:
        the caller degrades gracefully; this just makes the event visible."""
        seen = getattr(self, "_nonfinite_warned", None)
        if seen is None:
            seen = self._nonfinite_warned = set()
        key = (prof, where)
        if key in seen:
            return
        seen.add(key)
        off = getattr(self, "_theta_offset", None)
        off_p = None if not isinstance(off, dict) else off.get(prof)
        print(f"[SplineMtanhAnalytic] non-finite {where} for '{prof}' "
              f"(theta_offset={None if off_p is None else np.round(np.asarray(off_p, float), 4)}); "
              f"degrading to spline fallback", typeMsg="w")

    def get_y(self, params: Dict[str, Any], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        x_eval = np.asarray(x_eval, dtype=float)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            res = self._corrected_profile(prof, pv, x_eval)
            out[prof] = self._penalty_vec(x_eval.size) if res is None else res[0]
        self.y = out
        return out

    def get_aLy(self, params: Dict[str, Any], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        x_eval = np.asarray(x_eval, dtype=float)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            res = self._corrected_profile(prof, pv, x_eval)
            out[prof] = self._penalty_vec(x_eval.size) if res is None else res[1]
        self.aLy = out
        return out

    def get_curvature(self, params: Dict[str, Any], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        x_eval = np.asarray(x_eval, dtype=float)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            res = self._corrected_profile(prof, pv, x_eval)
            out[prof] = self._penalty_vec(x_eval.size) if res is None else res[2]
        self.curv = out
        return out

    def _evaluate_once(
        self, batch_params: Dict[str, Any], x_1d: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        """Single-batch (y, aLy, curvature) through :meth:`_corrected_profile`.

        Overrides the base so the batched update()/solver path shares the exact
        same corrected reconstruction (and analytic curvature) as the standalone
        get_y/get_aLy/get_curvature. Only the genuinely non-finite case falls
        back to a full Akima spline (as in the base).
        """
        x_1d = np.asarray(x_1d, dtype=float)
        y_out: Dict[str, np.ndarray] = {}
        aLy_out: Dict[str, np.ndarray] = {}
        curv_out: Dict[str, np.ndarray] = {}

        for prof, pv in batch_params.items():
            # Autograd path: a DV passed as a live tensor -> torch reconstruction
            # (y, aLy differentiable in the DVs + BCs). Curvature stays numpy (its
            # gradient is not needed by the flux-match objective).
            if torch.is_tensor(pv) and pv.requires_grad:
                bc_y = self.get_nearest_bc(prof, 1.0)
                bc_a = self.get_nearest_bc(f'aL{prof}', 1.0)
                if bc_y is not None and bc_a is not None:
                    y_bc = torch.as_tensor(float(bc_y['val'])).to(pv)
                    aLy_bc = torch.as_tensor(float(bc_a['val'])).to(pv)
                    rec = self._reconstruct_torch(prof, pv, torch.as_tensor(x_1d).to(pv), y_bc, aLy_bc)
                    if rec is not None:
                        y_out[prof], aLy_out[prof] = rec
                        resc = self._corrected_profile(prof, pv.detach().cpu().numpy(), x_1d)
                        curv_out[prof] = resc[2] if resc is not None else np.zeros_like(x_1d)
                        continue
            res = self._corrected_profile(prof, pv, x_1d)
            if res is None:
                if not hasattr(self, '_spline_fallback') or self._spline_fallback is None:
                    fallback_options = {
                        'knots': self.knots.tolist(),
                        'defined_on': self.defined_on,
                        'spline_type': 'akima',
                        'predicted_profiles': self.predicted_profiles,
                        'include_zero_grad_on_axis': self.include_zero_grad_on_axis,
                        'sigma': self.sigma,
                    }
                    self._spline_fallback = Spline(fallback_options)
                self._spline_fallback.bc_dict = self.bc_dict
                single_params = {prof: pv}
                y_out[prof]    = self._spline_fallback.get_y(single_params, x_1d)[prof]
                aLy_out[prof]  = self._spline_fallback.get_aLy(single_params, x_1d)[prof]
                curv_out[prof] = self._spline_fallback.get_curvature(single_params, x_1d)[prof]
                continue
            y_out[prof], aLy_out[prof], curv_out[prof] = res

        return y_out, aLy_out, curv_out, batch_params

    # ------------------------------------------------------------------
    # Analytic-Jacobian interior fit
    # ------------------------------------------------------------------
    def _resid_jac(self, theta, x_k, aLy_k, y_bc, aLy_bc, scale, peak_cap=None):
        s, c, w1, rho = (float(v) for v in theta)
        r, dr_drho, dr_dc = self._rho_c_to_r(rho, c)
        with np.errstate(over="ignore", invalid="ignore"):
            _, aLy_val, d_s, d_c, d_w1, d_r = _pedestal_eval_all(x_k, s, c, w1, r, self.x0, y_bc, aLy_bc)
        res = (aLy_val - aLy_k) / scale
        # Chain rule: r = rho*g(c), so c moves both the shape (d_c) and r (d_r*dr_dc);
        # rho moves r alone. Columns are [s, c, w1, rho].
        J = np.stack([d_s, d_c + d_r * dr_dc, d_w1, d_r * dr_drho], axis=-1) / scale[:, None]
        if self.lam_r > 0:
            # Regularize the (bounded) foot fraction rho toward 0; keeps the fit unique
            # when #knots<4 and is neutral on peaking once the knots demand it.
            res = np.concatenate([res, [self.lam_r * rho]])
            J = np.concatenate([J, [[0.0, 0.0, 0.0, self.lam_r]]], axis=0)
        if peak_cap is not None:
            # Runaway-tail penalty (only threaded in on the spike-guard re-fit): a
            # one-sided ReLU on the fractional overshoot of aLy above peak_cap at the
            # tail sample points. Rows are 0 (and drop out of J^TJ) wherever the backbone
            # is already under the cap, so the penalty only pulls down an over-peaked foot.
            with np.errstate(over="ignore", invalid="ignore"):
                _, aLy_p, ds_p, dc_p, dw_p, dr_p = _pedestal_eval_all(
                    self._peak_x, s, c, w1, r, self.x0, y_bc, aLy_bc)
            excess = (aLy_p - peak_cap) / peak_cap
            active = excess > 0.0
            pen = self.fit_peak_weight * np.where(active, excess, 0.0)
            Jp = (self.fit_peak_weight / peak_cap) * np.stack(
                [ds_p, dc_p + dr_p * dr_dc, dw_p, dr_p * dr_drho], axis=-1)
            Jp[~active] = 0.0
            res = np.concatenate([res, pen])
            J = np.concatenate([J, np.nan_to_num(Jp)], axis=0)
        return res, J

    def _interior_maxrel_theta(self, theta, x_k, aLy_k, y_bc, aLy_bc) -> float:
        s, c, w1, rho = (float(v) for v in theta)
        r, _, _ = self._rho_c_to_r(rho, c)
        with np.errstate(over="ignore", invalid="ignore"):
            _, aLy_val, *_ = _pedestal_eval_all(x_k, s, c, w1, r, self.x0, y_bc, aLy_bc)
        rel = np.abs(aLy_val - aLy_k) / np.maximum(np.abs(aLy_k), 1e-3)
        return float(np.max(rel)) if rel.size else np.inf

    def _backbone_aLy(self, theta, y_bc, aLy_bc) -> np.ndarray:
        """Backbone aLy on the full-domain guard grid. Cheap (one _pedestal_eval_all
        on ~64 points); the shared primitive behind the peak and dip diagnostics."""
        s, c, w1, rho = (float(v) for v in theta)
        r, _, _ = self._rho_c_to_r(rho, c)
        with np.errstate(over="ignore", invalid="ignore"):
            _, aLy_p, *_ = _pedestal_eval_all(self._peak_x, s, c, w1, r, self.x0, y_bc, aLy_bc)
        return np.nan_to_num(np.asarray(aLy_p, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def _profile_peak(self, theta, y_bc, aLy_bc) -> float:
        """Max backbone aLy over the full domain -- the quantity the runaway guard caps."""
        aLy_p = self._backbone_aLy(theta, y_bc, aLy_bc)
        return float(np.max(aLy_p)) if aLy_p.size else 0.0

    def _has_interior_dip(self, aLy_p) -> bool:
        """True if aLy turns over: it descends and then rises again somewhere on the
        domain (an interior local MINIMUM).  A monotone rise to an LCFS peak, or a
        single sub-separatrix peak, is unimodal and fine -- what this rejects is the
        dip-then-peak shape, which has no pedestal interpretation.  Slopes below 0.1%
        of the peak are treated as flat so grid noise does not register as a turnover."""
        aLy_p = np.asarray(aLy_p, dtype=float)
        if aLy_p.size < 3:
            return False
        pk = float(np.max(np.abs(aLy_p)))
        d = np.diff(aLy_p)
        sgn = np.sign(d[np.abs(d) > 1e-3 * max(pk, 1e-12)])
        if sgn.size < 2:
            return False
        sgn = sgn[np.insert(np.diff(sgn) != 0, 0, True)]      # collapse flat runs
        return bool(np.any((sgn[:-1] < 0) & (sgn[1:] > 0)))

    def _peak_cap(self, aLy_k, aLy_bc) -> float:
        """Cap on the reconstructed aLy peak: kappa * max(aLy at ANY knot, aLy(1)).

        The reference is the largest knot, not the last one: the solver legitimately
        asks for aLy(0.97) >> aLy_lcfs, and referencing only aLy_k[-1]/aLy_bc would
        either penalize that or (when the last knot is small) leave the cap far below
        what the DVs actually demand."""
        ref = abs(float(aLy_bc))
        if np.size(aLy_k):
            ref = max(ref, float(np.max(np.abs(np.asarray(aLy_k, dtype=float)))))
        return self.fit_peak_kappa * ref

    def _lm_solve(self, seed, x_k, aLy_k, y_bc, aLy_bc, scale, lo, hi, peak_cap=None):
        """Bounded Levenberg-Marquardt on the analytic-Jacobian residual. No scipy
        overhead (that was ~7.7 ms/solve): the 4x4 normal-equation solve + a handful
        of ``_resid_jac`` (~0.2 ms) evaluations converge a warm start in 1-2 outer
        iterations -> O(1 ms) fits. Deterministic (fixed iteration/damping caps, no
        randomness). Returns (theta, cost)."""
        theta = np.clip(np.asarray(seed, dtype=float).reshape(-1)[:4], lo, hi)
        r, J = self._resid_jac(theta, x_k, aLy_k, y_bc, aLy_bc, scale, peak_cap)
        r = np.nan_to_num(r); J = np.nan_to_num(J)
        cost = float(r @ r)
        lam = 1e-3
        for _ in range(self.fit_lm_iters):
            prev_cost = cost
            JtJ = J.T @ J
            grad = J.T @ r
            # Already at a (KKT-projected) stationary point -> stop before wasting the
            # damping search. This is the common case on a WARM start (the previous fit
            # is at/near the solution), so an unchanged re-fit costs ~1 residual eval.
            gp = grad.copy()
            at_lo = theta <= lo + 1e-12; at_hi = theta >= hi - 1e-12
            gp[at_lo & (gp > 0)] = 0.0; gp[at_hi & (gp < 0)] = 0.0
            if float(np.max(np.abs(gp))) < self.fit_lm_gtol:
                break
            diag = np.maximum(np.diag(JtJ), 1e-12)
            stepped = False
            for _ in range(self.fit_lm_damping_tries):
                try:
                    dth = np.linalg.solve(JtJ + lam * np.diag(diag), -grad)
                except np.linalg.LinAlgError:
                    break
                thn = np.clip(theta + dth, lo, hi)
                rn, Jn = self._resid_jac(thn, x_k, aLy_k, y_bc, aLy_bc, scale, peak_cap)
                rn = np.nan_to_num(rn)
                cn = float(rn @ rn)
                if cn < cost:                       # accept -> lower damping
                    theta, r, J, cost = thn, rn, np.nan_to_num(Jn), cn
                    lam = max(lam * 0.3, 1e-9)
                    stepped = True
                    break
                lam = min(lam * 4.0, 1e10)          # reject -> raise damping, retry
            # Converged: no accepted step, tiny step, OR the cost has stopped improving
            # meaningfully. The last criterion is what keeps the DEGENERATE foot
            # direction (which shaves the cost by epsilons forever) from burning the full
            # iteration budget -- the corrector pins the knots regardless, so a fit within
            # fit_lm_ctol of the local min is more than enough for the backbone shape.
            if (not stepped or float(np.max(np.abs(dth))) < 1e-7
                    or (prev_cost - cost) <= self.fit_lm_ctol * max(prev_cost, 1e-12)):
                break
        return theta, cost, r, J          # r, J correspond to the returned theta

    def _fit_theta(self, x_k, aLy_k, y_bc, aLy_bc, p0=None) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=float)
        aLy_k = np.asarray(aLy_k, dtype=float)
        scale = np.maximum(aLy_k, 1e-3)
        lo, hi = self._theta_bounds()

        if p0 is not None:
            # WARM path (solver hot loop): the previous fit is a converged solution for
            # nearby DVs, so a hand-rolled LM from it takes 1-2 iterations (~1 ms, no
            # scipy overhead) and stays in that (good) basin.
            best_theta, best_cost, best_r, best_J = self._lm_solve(p0, x_k, aLy_k, y_bc, aLy_bc, scale, lo, hi)
        else:
            # COLD path (first eval, no history): robust scipy TRF over a deterministic
            # c-seed spread, argmin on the interior residual. This is the ~1/solve cold
            # cost (rare -- the solver reuses one parameterizer, so all later evals are
            # warm), and it is a pure function of the inputs (deterministic).
            def fun(th):
                return np.nan_to_num(self._resid_jac(th, x_k, aLy_k, y_bc, aLy_bc, scale)[0],
                                     nan=1e6, posinf=1e6, neginf=-1e6)

            def jac(th):
                return np.nan_to_num(self._resid_jac(th, x_k, aLy_k, y_bc, aLy_bc, scale)[1],
                                     nan=0.0, posinf=0.0, neginf=0.0)
            seeds = [self._default_theta0(aLy_k, aLy_bc)]
            seeds += [np.array([self.s0, c0, self.w1_0, 0.5], dtype=float) for c0 in (0.93, 0.97, 1.02)]
            best_theta, best_rel = None, np.inf
            for seed in seeds:
                try:
                    cand = np.asarray(least_squares(
                        fun, np.clip(seed, lo, hi), jac=jac, bounds=(lo, hi), method='trf',
                        ftol=self.fit_trf_ftol, gtol=self.fit_trf_gtol, xtol=self.fit_trf_xtol,
                        x_scale='jac', max_nfev=self.fit_max_nfev).x, dtype=float)
                except Exception:
                    continue
                rel = self._interior_maxrel_theta(cand, x_k, aLy_k, y_bc, aLy_bc)
                if rel < best_rel:
                    best_theta, best_rel = cand, rel
                if best_rel <= self.fit_max_rel_error:
                    break
            if best_theta is None:
                best_theta = self._default_theta0(aLy_k, aLy_bc)
            best_r, best_J = self._resid_jac(best_theta, x_k, aLy_k, y_bc, aLy_bc, scale)

        # Runaway peak guard (detection-gated -> no cost/perturbation on well-behaved
        # fits). If the fitted backbone overshoots the cap anywhere on [x0,1], do ONE
        # penalized LM re-fit seeded at the current theta and adopt it only if it actually
        # lowers the peak. best_r/best_J are recomputed WITHOUT the penalty rows so the UQ
        # covariance stays the honest data-fit covariance. NB this is accept-if-better, not
        # accept-if-under-cap, so a converged fit can still sit above the cap -- the UQ
        # projection in _apply_theta_offset accounts for that.
        if self._peak_penalty_on and aLy_k.size:
            cap = self._peak_cap(aLy_k, aLy_bc)
            if self._profile_peak(best_theta, y_bc, aLy_bc) > cap:
                th_pen, _, _, _ = self._lm_solve(best_theta, x_k, aLy_k, y_bc, aLy_bc,
                                                 scale, lo, hi, peak_cap=cap)
                if self._profile_peak(th_pen, y_bc, aLy_bc) < self._profile_peak(best_theta, y_bc, aLy_bc):
                    best_theta = th_pen
                    best_r, best_J = self._resid_jac(best_theta, x_k, aLy_k, y_bc, aLy_bc, scale)

        # Fit-parameter covariance for UQ: Cov(theta) = sigma^2 (J^T J)^-1 at the
        # solution, with sigma^2 = SSR / dof.  theta = {s, c, w1, rho}; the analytic
        # Jacobian _resid_jac gives J directly, so this is essentially free and
        # carries all cross-terms of the mtanh-base fit.  Stashed for the caller
        # (_resolve) to key by channel.  pinv guards a rank-deficient J^T J.
        try:
            # Reuse the LM's final residual/Jacobian (they correspond to best_theta) so
            # the UQ covariance costs no extra _resid_jac -- a warm re-fit is then ~1 eval.
            r_fit = np.nan_to_num(np.asarray(best_r, dtype=float))
            J_fit = np.nan_to_num(np.asarray(best_J, dtype=float))
            m, n = J_fit.shape
            dof = max(m - n, 1)
            sigma2 = float(r_fit @ r_fit) / dof
            # Store the raw pieces so the UQ layer can add the LCFS y / aLy BC
            # soft-priors (those BCs are hard-enforced here -> zero Jacobian ->
            # they don't inform (JᵀJ), which is what lets degenerate c/w/r
            # directions blow up the under-resolved foot).  The data-only cov is
            # sigma2*pinv(JᵀJ); the regularized cov is built in fit_uq.
            self._last_fit_JtJ = J_fit.T @ J_fit
            self._last_fit_sigma2 = sigma2
            self._last_fit_cov = sigma2 * np.linalg.pinv(self._last_fit_JtJ)
        except Exception:
            self._last_fit_cov = None
            self._last_fit_JtJ = None
            self._last_fit_sigma2 = None
        return best_theta

    # ------------------------------------------------------------------
    # Resolve (cache + warm start) -> base-class physical tuple
    # ------------------------------------------------------------------
    def _apply_theta_offset(self, prof, theta, y_bc=None, aLy_bc=None, aLy_k=None):
        """UQ hook: add a fit-covariance perturbation to theta before reconstruction.

        ``_theta_offset[prof]`` is set by the edge-UQ scan to propagate the mtanh
        fit uncertainty through the full chain.  Nominal runs leave it unset (no
        effect).  Operates on the scan's deep-copied parameterizer, so it never
        pollutes the live cache.

        Two conditioners apply to the PERTURBED theta.  Both are needed because every
        feasibility guarantee this class advertises is enforced inside the FIT (bounds
        as lo/hi in _lm_solve, peak via the guard in _fit_theta) and the offset is added
        afterwards -- so an unconditioned UQ sample is reconstructed with all of them off:

        * bounds clip -- w1 back inside _W1_BOUNDS (below 0.01 the tanh foot is a
          razor-thin spike and aLy runs to ~10x its neighbours), rho below the u'(x)>0
          feasibility limit (past it the foot turns over, so aLy dips before rising to
          the peak -- the bimodal samples), s inside (0,1) (keeps m >= 0).
        * shape projection -- the offset direction comes from a LINEAR profile Jacobian
          but is applied as a FINITE step through a nonlinear reconstruction, so its
          amplitude is bisected down until the sample is feasible: peak under _peak_cap
          AND no turnover (_has_interior_dip).  The direction is preserved and only its
          magnitude is bounded, so each scan direction stays symmetric about nominal
          rather than being truncated on one side.  NB the clip alone does NOT prevent
          the dip-then-peak shape -- it is reachable with all of theta in bounds (c
          moving within _C_BOUNDS is enough), which is why the dip is a projection
          condition and not just a consequence of the bounds.
        """
        theta = np.asarray(theta, dtype=float)
        offs = getattr(self, "_theta_offset", None)
        off = None if offs is None else offs.get(prof)
        if off is None:
            return theta
        off = np.asarray(off, dtype=float)
        lo, hi = self._theta_bounds()
        th_full = np.clip(theta + off, lo, hi)
        if not self._peak_penalty_on or y_bc is None or aLy_bc is None:
            return th_full

        # alpha=0 must be feasible for the bisection to be well posed.  The fit-time
        # guard is accept-if-better rather than accept-if-under-cap, so the nominal fit
        # can itself exceed the cap (or, rarely, already dip); the feasibility test is
        # therefore relaxed to whatever the NOMINAL profile does.  Rule: "a UQ sample is
        # never worse-peaked or worse-shaped than the cap or the nominal, whichever is
        # looser" -- it conditions the excursion without re-shaping the nominal.
        aLy_nom = self._backbone_aLy(theta, y_bc, aLy_bc)
        cap = max(self._peak_cap(aLy_k, aLy_bc),
                  float(np.max(aLy_nom)) if aLy_nom.size else 0.0)
        dip_ok = self._unimodal_proj_on and not self._has_interior_dip(aLy_nom)

        def feasible(th):
            aLy_p = self._backbone_aLy(th, y_bc, aLy_bc)
            if aLy_p.size and float(np.max(aLy_p)) > cap:
                return False
            return not (dip_ok and self._has_interior_dip(aLy_p))

        if feasible(th_full):
            return th_full
        a_lo, a_hi = 0.0, 1.0
        for _ in range(self.peak_proj_iters):
            a = 0.5 * (a_lo + a_hi)
            if feasible(np.clip(theta + a * off, lo, hi)):
                a_lo = a
            else:
                a_hi = a
        return np.clip(theta + a_lo * off, lo, hi)

    def _resolve(self, prof, prof_params):
        bc_y = self.get_nearest_bc(prof, 1.0)
        bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
        y_bc = float(bc_y['val']) if bc_y is not None else 1.0
        aLy_bc = float(bc_aLy['val']) if bc_aLy is not None else 1.0

        n = len(self.knots)
        if isinstance(prof_params, dict):
            vec = np.array([float(prof_params[name]) for name in self.param_names], dtype=float)
        else:
            vec = np.asarray(prof_params, dtype=float).reshape(-1)
            if vec.size < len(self.param_names):
                raise ValueError(f"Expected at least {len(self.param_names)} params for '{prof}', got {vec.size}")
            vec = vec[:n].astype(float, copy=False)

        # Deterministic: the fit is a pure function of (knots, vec, BCs) -- no resolve
        # cache (its 10%-tolerance shape reuse was path-dependent AND could return a
        # stale backbone). The previous fit is passed only as a warm-start SEED; the LM
        # still converges to the same (unimodal) basin, so the result is unchanged --
        # it just gets there in ~1 iteration for the small DV steps a solver takes.
        theta = self._fit_theta(self.knots, vec, y_bc, aLy_bc,
                                p0=self._last_theta_guess.get(prof))
        phys = self._theta_to_phys(
            self._apply_theta_offset(prof, theta, y_bc=y_bc, aLy_bc=aLy_bc, aLy_k=vec),
            y_bc, aLy_bc)
        if phys is None or not np.all(np.isfinite(np.asarray(phys, dtype=float))):
            return None

        self._current_theta[prof] = theta
        self._last_theta_guess[prof] = theta
        # Per-channel mtanh-base fit covariance over theta={s,c,w1,r} for UQ,
        # plus the raw pieces + BC values needed to add the LCFS y/aLy soft-priors.
        if not hasattr(self, "_fit_cov"):
            self._fit_cov = {}
        if not hasattr(self, "_fit_cov_data"):
            self._fit_cov_data = {}
        self._fit_cov[prof] = getattr(self, "_last_fit_cov", None)
        self._fit_cov_data[prof] = {
            "JtJ": getattr(self, "_last_fit_JtJ", None),
            "sigma2": getattr(self, "_last_fit_sigma2", None),
            "theta": np.asarray(theta, dtype=float),
            "y_bc": float(y_bc), "aLy_bc": float(aLy_bc),
        }
        return phys

    # ------------------------------------------------------------------
    # Torch BC-gradient overlay: A = s*A_max scales with (y_bc*aLy_bc), so it
    # must stay live in the BCs (unlike the base class's free A). Mirrors
    # SplineMtanh._torch_grad_overlay but sources s from the resolved theta.
    # ------------------------------------------------------------------
    def _eval_once_torch_s(self, shape, s, y_bc, aLy_bc, x):
        """Torch reconstruction keeping A/aLy(1) live in the BCs, using the same
        cancellation-free ``g*Phi`` form as the numpy path (no explicit A)."""
        _, D0, delta, c = (float(v) for v in shape)
        w1_ = D0 * math.exp(delta * (1.0 - c))          # w(1)
        u1 = (1.0 - c) / w1_
        u1p = (1.0 - delta * (1.0 - c)) / w1_
        if abs(u1p) < self._DENOM_FLOOR:                # genuine turning-point guard only
            u1p = self._DENOM_FLOOR if u1p >= 0 else -self._DENOM_FLOOR

        y_bc = y_bc.to(x)
        aLy_bc = aLy_bc.to(x)
        g = s * y_bc * aLy_bc               # live in BCs
        m = (1.0 - s) * y_bc * aLy_bc

        w = D0 * torch.exp(delta * (x - c))
        u = (x - c) / w
        up = (1.0 - delta * (x - c)) / w
        # cosh(u1)/cosh(u) in log-space (u1 scalar, u tensor); ratio and the
        # sinh(u1-u) difference stay O(1) so no tanh-tail underflow/cancellation.
        logcosh_u1 = abs(u1) + math.log1p(math.exp(-2.0 * abs(u1))) - math.log(2.0)
        logcosh_u = u.abs() + torch.log1p(torch.exp(-2.0 * u.abs())) - math.log(2.0)
        r_ch = torch.exp(logcosh_u1 - logcosh_u)
        Phi = torch.sinh(u1 - u) / u1p * r_ch
        dPhi = -(up / u1p) * r_ch ** 2

        y = y_bc + g * Phi - m * (x - 1.0)
        dydx = g * dPhi - m

        y = torch.clamp(y, min=0.0)
        y_safe = torch.where(y.abs() < self._Y_FLOOR, torch.full_like(y, self._Y_FLOOR), y)
        aLy = torch.clamp(-dydx / y_safe, min=0.0)
        return y, aLy

    def _reconstruct_torch(self, prof, pv, x_t, y_bc, aLy_bc):
        """Fully corrected (y, aLy) as torch tensors, differentiable w.r.t. the aLy
        knot DVs ``pv`` and the LCFS BCs. The fit theta (backbone SHAPE) is DETACHED --
        the live gradient flows through ``g,m`` (the BCs) and the linear corrector
        ``Delta = B @ (DV - model)``. This straight-through gradient is EXACT at the
        knots (``d aLy(knot_i)/dDV_i = 1``) and analytic in the BCs; off the knots the
        DV gradient omits the backbone's re-fit response (a fully-exact one is ill-posed
        here -- the fit is locally degenerate at the steep foot, so the implicit-function
        pinv blows up; straight-through is stable and order-correct). Returns None on a
        non-finite fit (caller keeps the numpy value)."""
        n = len(self.knots)
        pv_np = (pv.detach().cpu().numpy() if torch.is_tensor(pv)
                 else np.asarray([float(pv[nm]) for nm in self.param_names] if isinstance(pv, dict)
                                 else pv, dtype=float)).reshape(-1)[:n]
        phys = self._resolve(prof, pv_np)
        theta = self._current_theta.get(prof)
        if phys is None or theta is None:
            return None
        A, D0, delta, m, c, b = phys
        s = float(theta[0])
        y_bc = y_bc.to(x_t); aLy_bc = aLy_bc.to(x_t)
        # backbone at the eval grid and at the knots (torch, live in BCs; shape detached)
        y_base, aLy_base = self._eval_once_torch_s((A, D0, delta, c), s, y_bc, aLy_bc, x_t)
        if not self.use_corrector:
            # Bare backbone: no interior-knot pinning, so the DVs enter only through the
            # detached fit -> live gradient is BC-only (matches the numpy value).
            y = torch.clamp(y_base, min=0.0)
            aLy = torch.clamp(aLy_base, min=0.0)
            if not (torch.all(torch.isfinite(y)) and torch.all(torch.isfinite(aLy))):
                return None
            return y, aLy
        kt = torch.as_tensor(np.asarray(self.knots, dtype=float)).to(x_t)
        _, model_k = self._eval_once_torch_s((A, D0, delta, c), s, y_bc, aLy_bc, kt)
        # linear corrector Delta = B @ resid (resid live in DVs); B constant (cached)
        tgt = pv.to(x_t).reshape(-1)[:n] if torch.is_tensor(pv) else torch.as_tensor(pv_np).to(x_t)
        resid = tgt - model_k
        B, _ = self._corrector_ops(x_t.detach().cpu().numpy().reshape(-1))
        Delta = torch.as_tensor(B).to(x_t) @ resid
        xr = x_t.reshape(-1)
        cum = torch.cat([torch.zeros(1, dtype=xr.dtype, device=xr.device),
                         torch.cumulative_trapezoid(Delta, xr)])
        phase = torch.clamp(cum[-1] - cum, -10.0, 10.0)      # int_x^1 Delta dx'
        y = torch.clamp(y_base * torch.exp(phase), min=0.0)
        aLy = torch.clamp(aLy_base + Delta, min=0.0)
        if not (torch.all(torch.isfinite(y)) and torch.all(torch.isfinite(aLy))):
            return None
        return y, aLy

    def _torch_grad_overlay(self, prof, pv, bc_dict, batch_idx, x_t, y_np, aLy_np):
        """Straight-through overlay: value = the faithful numpy reconstruction, gradient
        = the torch reconstruction (:meth:`_reconstruct_torch`), which carries BOTH the
        aLy-knot-DV gradient and the LCFS-BC gradient. Fires whenever the DVs or the BCs
        carry a live graph."""
        def _bc_at(val, idx):
            flat = val.reshape(-1)
            return flat[idx] if flat.numel() > 1 else flat[0]

        pv_live = torch.is_tensor(pv) and pv.requires_grad
        aLy_key, y_key = f"aL{prof}", prof
        aLy_entry = bc_dict.get(aLy_key) if isinstance(bc_dict, dict) else None
        if isinstance(aLy_entry, dict) and torch.is_tensor(aLy_entry.get("val")):
            aLy_bc = _bc_at(aLy_entry["val"], batch_idx)
        else:
            bc_a = self.get_nearest_bc(aLy_key, 1.0)
            aLy_bc = torch.as_tensor(float(bc_a["val"])).to(x_t) if bc_a is not None else None
        bc_live = torch.is_tensor(aLy_bc) and aLy_bc.requires_grad
        if aLy_bc is None or not (pv_live or bc_live):
            return None   # nothing to differentiate -> keep the numpy value

        y_entry = bc_dict.get(y_key) if isinstance(bc_dict, dict) else None
        if isinstance(y_entry, dict) and torch.is_tensor(y_entry.get("val")):
            y_bc = _bc_at(y_entry["val"], batch_idx)
        else:
            bc_y = self.get_nearest_bc(y_key, 1.0)
            if bc_y is None:
                return None
            y_bc = torch.as_tensor(float(bc_y["val"])).to(x_t)

        rec = self._reconstruct_torch(prof, pv, x_t, y_bc, aLy_bc)
        if rec is None:
            return None
        y_t, aLy_t = rec
        y_ref = torch.as_tensor(np.asarray(y_np, dtype=float)).to(x_t)
        aLy_ref = torch.as_tensor(np.asarray(aLy_np, dtype=float)).to(x_t)
        y_out = y_t + (y_ref - y_t).detach()
        aLy_out = aLy_t + (aLy_ref - aLy_t).detach()
        return y_out, aLy_out

    def _aLy_correction(self, prof, prof_params, A, D0, delta, m, c, b):
        """Residual corrector is *always* active for this model: it forces exact
        pass-through of the interior knots and simply collapses to ~0 when the
        analytic mtanh already matches them. (The base class short-circuits to
        ``None`` -- the pure-analytic fast path -- once the fit is within
        ``fit_max_rel_error``; here we never take that shortcut so the knot
        interpolation is exact on every resolve.)

        Delta(x) is pinned to 0 at the axis (x=0) and the LCFS (x=1) so both
        boundary conditions are preserved by the superposition.
        """
        n = len(self.knots)
        if n == 0 or not self.use_corrector:
            # use_corrector=False -> return None so _corrected_profile reports the bare
            # mtanh backbone (interior knots matched only through the fit theta).
            return None
        if isinstance(prof_params, dict):
            target = np.array([float(prof_params[name]) for name in self.param_names], dtype=float)
        else:
            target = np.asarray(prof_params, dtype=float).reshape(-1)[:n]

        knots = np.asarray(self.knots, dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y_k = self._y_mtanh(knots, A, D0, delta, m, c, b)
            dydx_k = self._dydx_mtanh(knots, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y_k) < self._Y_FLOOR, self._Y_FLOOR, y_k)
            model = np.clip(-dydx_k / y_safe, 0.0, None)
        resid = target - model
        # A large theta perturbation can overflow the mtanh at the knots; a
        # non-finite residual would poison the corrector. Bail so the caller
        # (_corrected_profile) degrades to the spline fallback instead.
        if not (np.all(np.isfinite(model)) and np.all(np.isfinite(resid))):
            return None
        # Return the raw knot residual; _corrected_profile maps it to Delta(x) via
        # the constant linear basis (_corrector_ops): Delta = B @ resid.
        return resid

    # ------------------------------------------------------------------
    # Merged base machinery (formerly SplineMtanh): profile parameterization,
    # the penalty vector, and the batched update() the solver drives.
    # ------------------------------------------------------------------
    @staticmethod
    def _penalty_vec(n: int) -> np.ndarray:
        return np.full(int(max(1, n)), SplineMtanhAnalytic._PENALTY, dtype=float)

    def parameterize(
        self, state, bc_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        self.build_bcs(bc_dict)
        params: Dict[str, Dict[str, float]] = {}
        params_std: Dict[str, Dict[str, float]] = {}

        x_data = np.asarray(getattr(state, 'roa')).flatten()
        n = len(self.knots)

        for prof in self.predicted_profiles:
            # DVs are a/Ly at the interior knots -> sample the input profile's actual
            # a/Ly there (robust and exact for any monotone profile; the mtanh+corrector
            # reconstruction in _resolve renders these knot values back within tolerance).
            aLy_name = f'aL{prof}'
            if hasattr(state, aLy_name):
                aLy_prof = np.asarray(getattr(state, aLy_name)).flatten()
            else:
                y_raw = np.asarray(getattr(state, prof)).flatten()
                y_safe = np.where(np.abs(y_raw) < self._Y_FLOOR, self._Y_FLOOR, y_raw)
                aLy_prof = -np.gradient(y_raw, x_data) / y_safe

            if aLy_prof.shape[0] != x_data.shape[0]:
                y_raw = np.asarray(getattr(state, prof)).flatten()
                y_safe = np.where(np.abs(y_raw) < self._Y_FLOOR, self._Y_FLOOR, y_raw)
                aLy_prof = -np.gradient(y_raw, x_data) / y_safe

            aLy_knots = np.clip(np.interp(self.knots, x_data, aLy_prof), 0.0, None)
            pdict = {f'aLy{i}': float(aLy_knots[i]) for i in range(n)}
            params[prof] = pdict
            params_std[prof] = {k: abs(v) * self.sigma for k, v in pdict.items()}

        self.params = params
        self.params_std = params_std
        return params, params_std

    def update(self, params: Dict[str, np.ndarray], bc_dict: Dict[str, Any], x_eval: np.ndarray):
        """Batched (y, aLy, curvature): one resolve per profile per batch item,
        with the torch BC-gradient overlay when the batch carries a live BC graph."""
        self.bc_tensors = {}
        n_knots = len(self.knots)

        def _to_numpy(value: Any) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def _slice_params_for_batch(
            all_params: Dict[str, Any], batch_idx: int, batch_size: int
        ) -> Dict[str, Any]:
            sliced: Dict[str, Any] = {}
            for prof, prof_params in all_params.items():
                if isinstance(prof_params, dict):
                    prof_sliced: Dict[str, Any] = {}
                    for name, value in prof_params.items():
                        arr = _to_numpy(value)
                        if arr.ndim == 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = float(arr[batch_idx])
                        elif arr.ndim > 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = arr[batch_idx]
                        else:
                            prof_sliced[name] = value
                    sliced[prof] = prof_sliced
                else:
                    arr = _to_numpy(prof_params)
                    if arr.ndim > 1 and arr.shape[0] == batch_size:
                        sliced[prof] = arr[batch_idx]
                    else:
                        sliced[prof] = prof_params
            return sliced

        x_arr = _to_numpy(x_eval)
        if x_arr.ndim <= 1:
            self.build_bcs(bc_dict)
            y, aLy, curv, _ = self._evaluate_once(params, x_arr)
            self.y = y
            self.aLy = aLy
            self.curv = curv
            return y, aLy, curv

        batch_size = x_arr.shape[0]
        use_batched_bcs = self._is_batched_bc_input(bc_dict, batch_size)
        if use_batched_bcs:
            self.bc_tensors = bc_dict
        else:
            self.build_bcs(bc_dict)

        x_is_tensor = torch.is_tensor(x_eval) and x_eval.dim() >= 2

        # Grad path fires whenever the eval grid is torch AND either the BCs are batched
        # tensors OR a DV is passed as a live tensor -- then _torch_grad_overlay carries
        # the DV and BC gradients (value stays the faithful numpy reconstruction).
        def _dv_is_live(pp) -> bool:
            if isinstance(pp, dict):
                return any(torch.is_tensor(v) and v.requires_grad for v in pp.values())
            return torch.is_tensor(pp) and pp.requires_grad

        def _live_dv_slice(prof: str, i: int):
            pp = params.get(prof)
            if isinstance(pp, dict):
                return {nm: (v[i] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == batch_size else v)
                        for nm, v in pp.items()}
            if torch.is_tensor(pp) and pp.dim() > 1 and pp.shape[0] == batch_size:
                return pp[i]
            return pp

        want_grad = x_is_tensor and (use_batched_bcs or any(_dv_is_live(params.get(pr)) for pr in params))

        y_batches: Dict[str, List[Any]] = {}
        aLy_batches: Dict[str, List[Any]] = {}
        curv_batches: Dict[str, List[np.ndarray]] = {}
        grad_profiles: set = set()

        for i in range(batch_size):
            if use_batched_bcs:
                self.build_bcs(self._slice_batched_bc_dict(bc_dict, i, batch_size))
            batch_params = _slice_params_for_batch(params, i, batch_size)
            y_i, aLy_i, curv_i, _ = self._evaluate_once(batch_params, x_arr[i])

            for prof, vals in curv_i.items():
                curv_batches.setdefault(prof, []).append(np.asarray(vals, dtype=float))

            for prof in y_i:
                y_val, aLy_val = y_i[prof], aLy_i[prof]
                overlay = None
                if want_grad:
                    overlay = self._torch_grad_overlay(
                        prof, _live_dv_slice(prof, i), bc_dict, i, x_eval[i],
                        y_val, aLy_val,
                    )
                if overlay is not None:
                    grad_profiles.add(prof)
                    y_batches.setdefault(prof, []).append(overlay[0])
                    aLy_batches.setdefault(prof, []).append(overlay[1])
                else:
                    y_batches.setdefault(prof, []).append(np.asarray(y_val, dtype=float))
                    aLy_batches.setdefault(prof, []).append(np.asarray(aLy_val, dtype=float))

        def _stack(batches, prof):
            vals = batches[prof]
            if prof in grad_profiles:
                ref = next(v for v in vals if torch.is_tensor(v))
                vals = [v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v, dtype=float)).to(ref) for v in vals]
                return torch.stack(vals, dim=0)
            return np.stack(vals, axis=0)

        y_out = {prof: _stack(y_batches, prof) for prof in y_batches}
        aLy_out = {prof: _stack(aLy_batches, prof) for prof in aLy_batches}
        curv_out = {prof: np.stack(vals, axis=0) for prof, vals in curv_batches.items()}

        self.y = y_out
        self.aLy = aLy_out
        self.curv = curv_out
        return y_out, aLy_out, curv_out


# -------------------------
# Factory and registry
# -------------------------


# Backward-compat alias: SplineMtanh was merged into SplineMtanhAnalytic (the analytic
# fit is the single mtanh-spline parameterizer). Existing configs/imports that name
# "SplineMtanh" / "spline_mtanh" resolve to the merged class.
SplineMtanh = SplineMtanhAnalytic

PARAMETER_MODELS = {
    'spline': Spline,
    'mtanh': Mtanh,
    'spline_mtanh': SplineMtanhAnalytic,
    'spline_mtanh_analytic': SplineMtanhAnalytic,
}


def create_parameter_model(config: Dict[str, Any]) -> ParameterBase:
    """Create a parameter model instance from config.

    Expected config format:
    {"type": "spline"|"mtanh"|"spline_mtanh"|"spline_mtanh_analytic", "kwargs": { ... model options ... }}
    """
    model_type = (config or {}).get('type', 'spline')
    kwargs = (config or {}).get('kwargs', {})
    cls = PARAMETER_MODELS.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown parameter model type: {model_type}")
    return cls(kwargs)


BCEntry = Dict[str, float]  # {'val': float, 'loc': float}

def _normalize_single_bc(val: Union[tuple, list, dict]) -> BCEntry:
    """Accept (value, loc) tuple/list or {'value':..., 'location':...}"""
    if isinstance(val, dict):
        if "val" in val:
            v = float(val["val"])
        else:
            v = float(val.get("value", 0.0))
        if "loc" in val:
            loc = float(val["loc"])
        else:
            loc = float(val.get("location", 1.0))
    elif isinstance(val, (tuple, list)) and len(val) == 2:
        v, loc = val
        v, loc = float(v), float(loc)
    else:
        raise ValueError("BC must be (value,location) or dict{'value','location'}")
    return {'val': v, 'loc': loc}