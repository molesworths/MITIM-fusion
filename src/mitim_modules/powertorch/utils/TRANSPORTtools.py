import json
from matplotlib.pylab import f
import torch
import numpy as np
from functools import partial
import copy
import shutil
from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def write_json(self, file_name = 'fluxes_turb.json', suffix= 'turb'):
    '''
    For tracking and reproducibility (e.g. external runs), we want to write a json file
    containing the simulation results. JSON should look like:
    
    {
        'fluxes_mean': 
            {
                'QeGB': ...
                'QiGB': ...
                'GeGB': ...
                'GZGB': ...
                'MtGB': ...
                'QieGB': ...
            },
        'fluxes_stds': 
            {
                'QeGB': ...
                'QiGB': ...
                'GeGB': ...
                'GZGB': ...
                'MtGB': ...
                'QieGB': ...
            },
        'additional_info': {
                'rho': rho.tolist(),
            }
    }
    '''
    
    write_json_from_variables = self._write_json_from_variables_turb if suffix == 'turb' else self._write_json_from_variables_neoc
    
    if self.folder.exists() and write_json_from_variables:
        
        with open(self.folder / file_name, 'w') as f:

            fluxes_mean = {}
            fluxes_stds = {}

            for var in ['QeGB', 'QiGB', 'GeGB', 'GZGB', 'MtGB']:
                fluxes_mean[var] = self.__dict__[f"{var}_{suffix}"].tolist()
                fluxes_stds[var] = self.__dict__[f"{var}_{suffix}_stds"].tolist()

            try:
                var = 'QieGB'
                fluxes_mean[var] = self.__dict__[f"{var}_{suffix}"].tolist()
                fluxes_stds[var] = self.__dict__[f"{var}_{suffix}_stds"].tolist()
            except KeyError:
                # NEO file may not have it
                pass

            json_dict = {
                'fluxes_mean': fluxes_mean,
                'fluxes_stds': fluxes_stds,
                'additional_info': {
                    'rho': self.powerstate.plasma["rho"][0, 1:].cpu().numpy().tolist(),
                    'roa': self.powerstate.plasma["roa"][0, 1:].cpu().numpy().tolist(),
                    'Qgb': self.powerstate.plasma["Qgb"][0, 1:].cpu().numpy().tolist(),
                    'aLte': self.powerstate.plasma["aLte"][0, 1:].cpu().numpy().tolist(),
                    'aLti': self.powerstate.plasma["aLti"][0, 1:].cpu().numpy().tolist(),
                    'aLne': self.powerstate.plasma["aLne"][0, 1:].cpu().numpy().tolist(),
                }
            }

            json.dump(json_dict, f, indent=4)

        print(f"\t* Written JSON with {suffix} information to {self.folder / file_name}")
        
    else:
        
        print(f"\t* Folder {self.folder} does not exist, cannot write {file_name}", typeMsg='w')

class power_transport:

    def __init__(self, powerstate, name = "test", folder = "~/scratch/", evaluation_number = 0):

        self.name = name
        self.folder = IOtools.expandPath(folder)
        self.evaluation_number = evaluation_number
        self.powerstate = powerstate

        self.transport_evaluator_options  = self.powerstate.transport_options["options"]
        self.cold_start                   = self.powerstate.transport_options["cold_start"]

        # Model results is None by default, but can be assigned in evaluate
        self.model_results = None
        
        # By default, write the json files after evaluating the variables (will be changed in gyrokinetic "prep" run mode)
        self._write_json_from_variables_turb = True
        self._write_json_from_variables_neoc = True

        # ----------------------------------------------------------------------------------------
        # labels for plotting
        # ----------------------------------------------------------------------------------------

        self.powerstate.labelsFluxes = {
            "te": "$Q_e$ ($MW/m^2$)",
            "ti": "$Q_i$ ($MW/m^2$)",
            "ne": "$Q_{conv}$ ($MW/m^2$)",
            "nZ": "$Q_{conv}$ $\\cdot f_{Z,0}$ ($MW/m^2$)",
            "w0": "$M_T$ ($J/m^2$)",
        }

    def evaluate(self):

        # Copy the input.gacode files to the output folder
        self._profiles_to_store()

        '''
        ******************************************************************************************************
        Evaluate neoclassical and turbulent transport (*in GB units*). 
        These functions use a hook to write the .json files to communicate the results to powerstate.plasma
        ******************************************************************************************************
        '''
        
        # Initialize them as zeros
        for var in ['QeGB','QiGB','GeGB','GZGB','MtGB','QieGB']:
            for suffix in ['turb', 'neoc']:
                for suffix0 in ['', '_stds']:
                    self.__dict__[f"{var}_{suffix}{suffix0}"] = torch.zeros(self.powerstate.plasma['rho'].shape[-1]-1)
        
        neoclassical = self.evaluate_neoclassical()
        turbulence = self.evaluate_turbulence()

        # Archive the real evaluation's decks + fluxes (opt-in, never fatal).
        # Done here, before _postprocess, while the GB flux/std arrays and the
        # per-rho deck objects (turbulence/neoclassical.inputs_files) coexist.
        self._archive_evaluation(turbulence, neoclassical)

        '''
        ******************************************************************************************************
        From the json to powerstate.plasma and GB to real units transformation
        ******************************************************************************************************
        '''
        self._populate_from_json(file_name = 'fluxes_turb.json', suffix= 'turb')
        self._populate_from_json(file_name = 'fluxes_neoc.json', suffix= 'neoc')

        '''
        ******************************************************************************************************
        Post-process the data: add turb and neoc, tensorize and transformations
        ******************************************************************************************************
        '''
        self._postprocess()

    def _archive_evaluation(self, turbulence, neoclassical):
        """Write the real evaluation to the EvaluationArchive.

        Enabled by default; the archive persists ACROSS runs (that is the point --
        future solves reuse prior-run neighbours), so the default path is a stable
        user-level location, NOT the per-run folder:
            transport_options["archive"]["path"]  ->  $MITIM_EVAL_ARCHIVE  ->  ~/.mitim/evaluation_archive
        Disable per run with transport_options["archive"] = {"enabled": False}.
        Wrapped so an archiving failure can never break a transport evaluation.
        """
        import os
        archive_opts = self.powerstate.transport_options.get("archive", {}) \
            if isinstance(self.powerstate.transport_options, dict) else {}
        archive_opts = archive_opts or {}
        if not archive_opts.get("enabled", True):
            return
        try:
            from pathlib import Path
            from mitim_modules.powertorch.utils.EVALarchive import EvaluationArchive
            path = archive_opts.get("path") or os.environ.get("MITIM_EVAL_ARCHIVE") \
                or (Path.home() / ".mitim" / "evaluation_archive")
            arch = EvaluationArchive(path)
            records = arch.records_from_transport(self, turbulence, neoclassical)
            if records:
                arch.append(records)
                print(f"\t* [EvaluationArchive] wrote {len(records)} record(s) to {IOtools.clipstr(str(arch.path))}")
        except Exception as e:
            print(f"\t* [EvaluationArchive] skipped (non-fatal): {e}", typeMsg="w")
        
    def _postprocess(self):
        '''
        Curate information for the powerstate (e.g. add models, add batch dimension, rho=0.0, and tensorize)
        Before calling this function, the powerstate.plasma should have the following variables:
            'QeMWm2_tr_X', 'QiMWm2_tr_X', 'Ge1E20m2_tr_X', 'GZ1E20m2_tr_X', 'MtJm2_tr_X', 'QieMWm3_tr_X'
        where X = 'turb' or 'neoc'
        and also the corresponding _stds versions
        '''

        variables = ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']

        for variable in variables:
            for suffix in ['_tr_turb', '_tr_turb_stds', '_tr_neoc', '_tr_neoc_stds']:

                # Make them tensors and add a batch dimension
                self.powerstate.plasma[f"{variable}{suffix}"] = torch.Tensor(self.powerstate.plasma[f"{variable}{suffix}"]).to(self.powerstate.dfT).unsqueeze(0)
 
                # Pad with zeros at rho=0.0
                self.powerstate.plasma[f"{variable}{suffix}"] = torch.cat((
                    torch.zeros((1, 1)),
                    self.powerstate.plasma[f"{variable}{suffix}"],
                ), dim=1)

        # -----------------------------------------------------------
        # Sum the turbulent and neoclassical contributions
        # -----------------------------------------------------------
        
        variables = ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2']
        
        for variable in variables:
            self.powerstate.plasma[f"{variable}_tr"] = self.powerstate.plasma[f"{variable}_tr_turb"] + self.powerstate.plasma[f"{variable}_tr_neoc"]

        # ---------------------------------------------------------------------------------
        # Convective fluxes (& Re-scale the GZ flux by the original impurity concentration)
        # ---------------------------------------------------------------------------------
        
        mapper_convective = {
            'Ce': 'Ge1E20m2',
            'CZ': 'GZ1E20m2',
        }
        
        for key in mapper_convective.keys():
            for tt in ['','_turb', '_turb_stds', '_neoc', '_neoc_stds']:
                
                mult = 1/self.powerstate.fImp_orig if key == 'CZ' else 1.0
                
                self.powerstate.plasma[f"{key}_tr{tt}"] = PLASMAtools.convective_flux(
                    self.powerstate.plasma["te"],
                    self.powerstate.plasma[f"{mapper_convective[key]}_tr{tt}"]
                ) * mult
           
    def produce_profiles(self):
        # Only add self._produce_profiles() if it's needed (e.g. full TGLF), otherwise this is somewhat expensive
        # (e.g. for flux matching of analytical models)
        pass

    def _produce_profiles(self,derive_quantities=True):

        self.applyCorrections = self.powerstate.transport_options["applyCorrections"]

        # Write this updated profiles class (with parameterized profiles and target powers)
        self.file_profs = self.folder / "input.gacode"

        powerstate_detached = self.powerstate.copy_state()

        self.powerstate.profiles = powerstate_detached.from_powerstate(
            write_input_gacode=self.file_profs,
            postprocess_input_gacode=self.applyCorrections,
            rederive_profiles = derive_quantities,        # Derive quantities so that it's ready for analysis and plotting later
            insert_highres_powers = derive_quantities,    # Insert powers so that Q, Pfus and all that it's consistent when read later
        )

        self.powerstate.profiles_transport = copy.deepcopy(self.powerstate.profiles)

        self._modify_profiles()

    def _modify_profiles(self):
        '''
        Modify the profiles (e.g. lumping) before running the transport model 
        '''

        # After producing the profiles, copy for future modifications
        self.file_profs_unmod = self.file_profs.parent / f"{self.file_profs.name}_unmodified"
        shutil.copy2(self.file_profs, self.file_profs_unmod)

        profiles_postprocessing_fun = self.powerstate.transport_options["profiles_postprocessing_fun"]

        if profiles_postprocessing_fun is not None:
            print(f"\t- Modifying input.gacode to run transport calculations based on {profiles_postprocessing_fun}",typeMsg="i")
            self.powerstate.profiles_transport = profiles_postprocessing_fun(self.file_profs)

        # Position of impurity ion may have changed
        p_old = PROFILEStools.gacode_state(self.file_profs_unmod)
        p_new = PROFILEStools.gacode_state(self.file_profs)

        impurity_of_interest = p_old.Species[self.powerstate.impurityPosition]

        try:
            impurityPosition_new = p_new.Species.index(impurity_of_interest)

        except ValueError:
            print(f"\t- Impurity {impurity_of_interest} not found in new profiles, keeping position {self.powerstate.impurityPosition}",typeMsg="w")
            impurityPosition_new = self.powerstate.impurityPosition

        if impurityPosition_new != self.powerstate.impurityPosition:
            print(f"\t- Impurity position has changed from {self.powerstate.impurityPosition} to {impurityPosition_new}",typeMsg="i")
            self.powerstate.impurityPosition_transport = p_new.Species.index(impurity_of_interest)

    def _profiles_to_store(self):

        if "folder" in self.powerstate.transport_options:
            whereFolder = IOtools.expandPath(self.powerstate.transport_options["folder"] / "Outputs" / "portals_profiles")
            if not whereFolder.exists():
                IOtools.askNewFolder(whereFolder)

            fil = whereFolder / f"input.gacode.{self.evaluation_number}"
            shutil.copy2(self.file_profs, fil)
            shutil.copy2(self.file_profs_unmod, fil.parent / f"{fil.name}_unmodified")
            print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
        else:
            print("\t- Could not move files", typeMsg="w")
                
    def _populate_from_json(self, file_name = 'fluxes_turb.json', suffix= 'turb'):
        '''
        Populate the powerstate.plasma with the results from the json file
        '''
        
        mapper = {
            'QeGB': ['Qgb', 'QeMWm2'],
            'QiGB': ['Qgb', 'QiMWm2'],
            'GeGB': ['Ggb', 'Ge1E20m2'],
            'GZGB': ['Ggb', 'GZ1E20m2'],
            'MtGB': ['Pgb', 'MtJm2'],
            'QieGB': ['Sgb', 'QieMWm3']
        }
        
        '''
        **********************************************************************************************
        If no population file exists, I only convert from GB to real units and return
        **********************************************************************************************
        '''
        if not (self.folder / file_name).exists():
            print(f"\t* File {self.folder / file_name} does not exist, cannot populate powerstate.plasma", typeMsg='w')
            print(f"\t- Tranforming from GB to real units:")
            
            def _np(a):
                return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
            for var in mapper:
                gb = _np(self.powerstate.plasma[f"{mapper[var][0]}"][0, 1:])
                self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"] = _np(self.__dict__[f"{var}_{suffix}"]) * gb
                self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}_stds"] = _np(self.__dict__[f"{var}_{suffix}_stds"]) * gb

            return
        
        '''
        **********************************************************************************************
        Populate the powerstate.plasma from the json file
        **********************************************************************************************
        '''
        print(f"\t* Populating powerstate.plasma with JSON data from {self.folder / file_name}")

        with open(self.folder / file_name, 'r') as f:
            json_dict = json.load(f)
        
        # See if the file has GB or real units
        units_GB, units_real = False, False
        if 'QeGB' in json_dict['fluxes_mean']:
            units_GB = True
        if 'QeMWm2' in json_dict['fluxes_mean']:
            units_real = True

        units = 'both' if (units_GB and units_real) else 'GB' if units_GB else 'real' if units_real else 'none'

        if units == 'real':
            
            print("\t\t- File has fluxes in real units... populating powerstate directly")

            for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
                self.powerstate.plasma[f"{var}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var])
                self.powerstate.plasma[f"{var}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var])

        elif units == 'GB' or units == 'both':

            dum = {}
            for var in mapper:
                gb = self.powerstate.plasma[f"{mapper[var][0]}"][0,1:].cpu().numpy()
                dum[f"{mapper[var][1]}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var]) * gb
                dum[f"{mapper[var][1]}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var]) * gb

            if units == 'GB':
                
                print("\t\t- File has fluxes in GB units... using GB units from powerstate to convert to real units")

                for var in mapper:
                    self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"] = dum[f"{mapper[var][1]}_tr_{suffix}"]
                    self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}_stds"] = dum[f"{mapper[var][1]}_tr_{suffix}_stds"]

            elif units == 'both':
                
                print("\t\t- File has fluxes in both GB and real units... using real units and checking consistency")

                for var in mapper:
                    if not np.allclose(self.powerstate.plasma[f"{mapper[var][1]}_tr_{suffix}"], dum[f"{mapper[var][1]}_tr_{suffix}"]):
                        print(f"\t\t\t- Inconsistent values found for {mapper[var][1]}_tr_{suffix}")

                for var in ['QeMWm2', 'QiMWm2', 'Ge1E20m2', 'GZ1E20m2', 'MtJm2', 'QieMWm3']:
                    self.powerstate.plasma[f"{var}_tr_{suffix}"] = np.array(json_dict['fluxes_mean'][var])
                    self.powerstate.plasma[f"{var}_tr_{suffix}_stds"] = np.array(json_dict['fluxes_stds'][var])

        else:
            raise ValueError("[MITIM] Unknown units in JSON file")

    # ----------------------------------------------------------------------------------------------------
    # EVALUATE (custom part)
    # ----------------------------------------------------------------------------------------------------
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self):
        '''
        This needs to populate the following np.arrays in self., with dimensions of rho:
            - QeGB_turb
            - QiGB_turb
            - GeGB_turb
            - GZGB_turb
            - MtGB_turb
            - QieGB_turb (turbulence exchange)
        and their respective standard deviations, e.g. QeGB_turb_stds
        '''

        print(">> No turbulent fluxes to evaluate", typeMsg="w")
    
    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))    
    def evaluate_neoclassical(self):
        '''
        This needs to populate the following np.arrays in self.:
            - QeGB_neoc
            - QiGB_neoc
            - GeGB_neoc
            - GZGB_neoc
            - MtGB_neoc
            - QieGB_neoc (zero)
        and their respective standard deviations, e.g. QeGB_neoc_stds
        '''

        print(">> No neoclassical fluxes to evaluate", typeMsg="w")
        
        
# *******************************************************************************************
# Combinations
# *******************************************************************************************

from mitim_modules.powertorch.physics_models.transport_tglf import tglf_model
from mitim_modules.powertorch.physics_models.transport_neo import neo_model
from mitim_modules.powertorch.physics_models.transport_cgyro import cgyro_model
from mitim_modules.powertorch.physics_models.transport_qlgyro import qlgyro_model
from mitim_modules.powertorch.physics_models.transport_gx import gx_model

# *******************************************************************************************
# Location-specific model selection
# *******************************************************************************************
'''
turbulence_model / neoclassical_model accept, besides the usual single string
(same code at every radius), a *per-radius* specification tied to the predicted
radial locations (predicted_roa / predicted_rho, in that order):

    turbulence_model: "tglf"                                # all radii (legacy)
    turbulence_model: ["tglf", "tglf", "cgyro", "cgyro"]    # one entry per predicted radius
    turbulence_model: {0.85: "tglf", 0.95: "cgyro"}         # keyed by r/a (or rho)
    turbulence_model: {0: "tglf", 3: "cgyro"}               # keyed by radial index
    turbulence_model: {"default": "tglf", 0.95: "cgyro"}    # partial dict + fallback

Each distinct code is run ONCE over the subset of radii assigned to it (the codes
are radius-vectorized, so this is the cheapest grouping), on a radially-subset view
of the powerstate, and the resulting GB fluxes/stds are scattered back into the
full-length arrays that power_transport expects. When a single code covers every
radius the legacy path is used verbatim (same folders, same behavior).
'''

_TURBULENCE_MODELS = {
    'tglf': tglf_model.evaluate_turbulence,
    'cgyro': cgyro_model.evaluate_turbulence,
    'qlgyro': qlgyro_model.evaluate_turbulence,
    'gx': gx_model.evaluate_turbulence,
}

_NEOCLASSICAL_MODELS = {
    'neo': neo_model.evaluate_neoclassical,
}

_FLUX_VARS_GB = ['QeGB', 'QiGB', 'GeGB', 'GZGB', 'MtGB', 'QieGB']


def models_per_radius(spec, n_radii, roa=None, rho=None, kind='transport'):
    '''
    Expand a model specification (str, per-radius sequence, or dict keyed by
    radial index / r-a / rho) into a list of model names, one per radius.
    '''

    if spec is None:
        raise Exception(f"[MITIM] No {kind} model specified")

    if isinstance(spec, str):
        return [spec] * n_radii

    if isinstance(spec, dict):
        default = spec.get("default", None)
        models = [default] * n_radii

        for key, value in spec.items():
            if key == "default":
                continue

            if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
                indices = [int(key)]
                if not (0 <= indices[0] < n_radii):
                    raise Exception(f"[MITIM] {kind} model index {key} out of range (0-{n_radii-1})")
            else:
                # Float key: match against the predicted r/a first, then rho
                indices = []
                for coord in (roa, rho):
                    if coord is None:
                        continue
                    indices = [i for i in range(n_radii) if np.isclose(coord[i], float(key), rtol=0.0, atol=1e-6)]
                    if len(indices) > 0:
                        break
                if len(indices) == 0:
                    raise Exception(
                        f"[MITIM] {kind} model key {key} does not match any predicted radius "
                        f"(r/a = {None if roa is None else list(np.round(roa,6))}, "
                        f"rho = {None if rho is None else list(np.round(rho,6))})")

            for i in indices:
                models[i] = value

        if any(m is None for m in models):
            missing = [i for i, m in enumerate(models) if m is None]
            raise Exception(
                f"[MITIM] {kind} model not defined at radial position(s) {missing}. "
                "Provide them explicitly or add a 'default' key")

        return models

    # Sequence (list, tuple, array)
    models = list(spec)
    if len(models) != n_radii:
        raise Exception(f"[MITIM] {kind} model list has {len(models)} entries but there are {n_radii} predicted radii")

    return models


def unique_models(spec):
    '''
    Set of model names in a (possibly per-radius) model specification. Useful for
    downstream checks that require a specific code at all radii.
    '''

    if spec is None:
        return set()
    if isinstance(spec, str):
        return {spec}
    if isinstance(spec, dict):
        return {v for v in spec.values() if v is not None}
    return {m for m in spec if m is not None}


def _subset_plasma_radially(plasma, indices, n_radii):
    '''
    Radially-subset view of a transport-facing plasma dict. Tensors whose radial
    dimension (dim 1) is n_radii+1 keep their axis-padding entry (index 0); those
    of length n_radii are subset directly. Everything else passes through.
    '''

    idx_pad = [0] + [i + 1 for i in indices]
    idx = list(indices)

    out = {}
    for key, val in plasma.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[1] in (n_radii, n_radii + 1):
            selection = idx_pad if val.shape[1] == n_radii + 1 else idx
            out[key] = val.index_select(1, torch.as_tensor(selection, device=val.device))
        elif isinstance(val, np.ndarray) and val.ndim >= 2 and val.shape[1] in (n_radii, n_radii + 1):
            selection = idx_pad if val.shape[1] == n_radii + 1 else idx
            out[key] = np.take(val, selection, axis=1)
        else:
            out[key] = val

    return out


class multi_model_results:
    '''
    Aggregate of the per-code model objects produced by a location-specific run.
    Exposes a merged ``inputs_files`` (keyed by rho, as each code does) so that
    downstream consumers that look up decks by radius (e.g. EvaluationArchive)
    see the full radial set.
    '''

    def __init__(self, results_by_model, models_per_radius):
        self.results_by_model = results_by_model
        self.models_per_radius = models_per_radius

        self.inputs_files = {}
        for model_object in results_by_model.values():
            inputs_files = getattr(model_object, "inputs_files", None)
            if isinstance(inputs_files, dict):
                self.inputs_files.update(inputs_files)


class portals_transport_model(power_transport, tglf_model, neo_model, cgyro_model, qlgyro_model, gx_model):

    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

        # Defaults (a string applies to all radii; see models_per_radius for per-location specs)
        self.turbulence_model = 'tglf'
        self.neoclassical_model = 'neo'

    def produce_profiles(self):
        self._produce_profiles()

    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_turb.json', suffix= 'turb'))
    def evaluate_turbulence(self):
        return self._evaluate_by_location(self.turbulence_model, _TURBULENCE_MODELS, 'turb', 'turbulence')

    @IOtools.hook_method(after=partial(write_json, file_name = 'fluxes_neoc.json', suffix= 'neoc'))
    def evaluate_neoclassical(self):
        return self._evaluate_by_location(self.neoclassical_model, _NEOCLASSICAL_MODELS, 'neoc', 'neoclassical')

    # ----------------------------------------------------------------------------------------------------
    # Location-specific dispatch
    # ----------------------------------------------------------------------------------------------------

    def _evaluate_by_location(self, spec, registry, suffix, kind):

        n_radii = self.powerstate.plasma["rho"].shape[-1] - 1

        def _coord(key):
            if key not in self.powerstate.plasma:
                return None
            return self.powerstate.plasma[key][0, 1:].detach().cpu().numpy()

        models = models_per_radius(spec, n_radii, roa=_coord("roa"), rho=_coord("rho"), kind=kind)

        for model in models:
            if model not in registry:
                raise Exception(f"Unknown {kind} model {model}")

        models_unique = list(dict.fromkeys(models))

        # Single code everywhere: legacy path, untouched (same folders and naming)
        if len(models_unique) == 1:
            return registry[models_unique[0]](self)

        print(f"\t- Location-specific {kind} models: " +
              ", ".join([f"#{i} ({models[i]})" for i in range(n_radii)]), typeMsg="i")

        results_by_model = {}
        model_results = {}
        for model in models_unique:

            indices = [i for i in range(n_radii) if models[i] == model]

            sub = self._radial_subset_evaluator(indices, model, suffix, n_radii)
            results_by_model[model] = registry[model](sub)

            self._gather_from_subset(sub, indices, suffix)

            if getattr(sub, "model_results", None) is not None:
                model_results[model] = sub.model_results

        if len(model_results) > 0:
            self.model_results = model_results

        return multi_model_results(results_by_model, models)

    def _radial_subset_evaluator(self, indices, model, suffix, n_radii):
        '''
        Shallow clone of this evaluator that sees only ``indices`` of the radial grid,
        with its own folder so that the codes' subfolders never collide.
        '''

        sub = copy.copy(self)

        sub.powerstate = copy.copy(self.powerstate)
        sub.powerstate.plasma = _subset_plasma_radially(self.powerstate.plasma, indices, n_radii)

        # Codes mutate their shared options dict (QLGYRO and CGYRO force the tglf block's
        # use_scan_trick_for_stds=None for their internal base-TGLF run). Give each group its
        # own copy, so that one code cannot silently change the settings another group runs
        # with depending on the order the groups happen to be evaluated in.
        try:
            sub.transport_evaluator_options = copy.deepcopy(self.transport_evaluator_options)
        except Exception as e:
            print(f"\t- Could not isolate the {model} transport options ({e}), sharing them", typeMsg="w")

        sub.folder = self.folder / model
        sub.folder.mkdir(parents=True, exist_ok=True)
        sub.name = f"{self.name}_{model}"

        # Flux containers at the subset length (the codes overwrite them, but keep shapes consistent)
        for var in _FLUX_VARS_GB:
            for suffix0 in ['', '_stds']:
                sub.__dict__[f"{var}_{suffix}{suffix0}"] = np.zeros(len(indices))

        return sub

    def _gather_from_subset(self, sub, indices, suffix):
        '''
        Scatter the subset GB fluxes/stds back into this evaluator's full-length arrays.
        '''

        for var in _FLUX_VARS_GB:
            for suffix0 in ['', '_stds']:
                key = f"{var}_{suffix}{suffix0}"

                values = sub.__dict__.get(key, None)
                if values is None:
                    continue
                values = np.asarray(_to_numpy(values), dtype=float).reshape(-1)

                if values.shape[0] != len(indices):
                    raise Exception(
                        f"[MITIM] {sub.name} returned {values.shape[0]} value(s) for '{key}' "
                        f"but was run at {len(indices)} radial location(s)")

                full = np.asarray(_to_numpy(self.__dict__[key]), dtype=float).reshape(-1)
                full[indices] = values
                self.__dict__[key] = full

        # A code may ask that the json is provided externally instead of written from its
        # variables (gyrokinetic run_type='prep'). Then the json IS the source of truth and
        # is read back by _populate_from_json from the PARENT folder, whereas this sub-model
        # waits for one in its own folder covering only its own radii -- so the fluxes at the
        # other radii would be silently lost. Not supported per-location.
        for flag in ['_write_json_from_variables_turb', '_write_json_from_variables_neoc']:
            if not getattr(sub, flag, True):
                raise NotImplementedError(
                    "[MITIM] Externally-provided fluxes (gyrokinetic run_type='prep') are not "
                    "supported with location-specific transport models, because the json is "
                    "read back for the full radial grid. Run that code at all radii instead")


def _to_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
