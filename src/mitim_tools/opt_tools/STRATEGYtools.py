import sys
import re
import copy
import datetime
import array
import traceback
from typing import IO
import torch
from pathlib import Path
from collections import OrderedDict
from IPython import embed
import dill as pickle_dill
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools, GRAPHICStools, GUItools, LOGtools
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools.opt_tools import OPTtools, STEPtools
from mitim_tools.opt_tools.optimizers import multivariate_tools
from mitim_tools.opt_tools.utils import (
    BOgraphics,
    SBOcorrections,
    TESTtools,
    EVALUATORtools,
    SAMPLINGtools,
)
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__

"""
Example usage (see tutorials for actual examples and parameter definitions):

	# Define function to optimize

		class optimization_object(opt_evaluator):

			def __init__(self,folder,namelist=None,function_parameters={}):

				super().__init__(folder,namelist=namelist)

				self.function_parameters = function_parameters

			def run(self,paramsfile,resultsfile):

				# Read stuff
				FolderEvaluation,numEval,dictDVs,dictOFs = self.read(paramsfile,resultsfile)

				# Operations
				...

				# Write stuff
				self.write(dictOFs,resultsfile)

			def analyze_results(self,plotYN=True,fn = None):

				# Things to do when looking at final results [OPTIONAL]

	# Run Workflow

	MITIM_BO = STRATEGYtools.MITIM_BO(optimization_object)
	MITIM_BO.run()

Notes:
	- A "nan" in the evaluator output (e.g. written in optimization_data) means that it has not been evaluated, so this is prone to be tried again.
		This is especially useful when I only want to read from a previous optimization workflow the x values, but I want to evaluate.
	- An "inf" in the evaluator output (e.g. written in optimization_data) means that the evaluation failed and won't be re-tried again. That individual
		will just not be considered during surrogate fitting.
"""


# Parent optimization function
class opt_evaluator:
    def __init__(
        self,
        folder,
        namelist=None,
        default_namelist_function=None,
        tensor_options = {
            "dtype": torch.double,
            "device": torch.device("cpu"),
        }
    ):
        """
        Namelist file can be provided and will be copied to the folder
        """

        self.tensor_options = tensor_options

        print("- Parent opt_evaluator function initialized")

        self.folder = folder

        print(f"\t- Folder: {self.folder}")

        if self.folder is not None:

            self.folder = IOtools.expandPath(self.folder)
            if not self.folder.exists():
                IOtools.askNewFolder(self.folder)
            if not (self.folder / "Outputs").exists():
                IOtools.askNewFolder(self.folder / "Outputs")

        if namelist is not None:
            print(f"\t- Optimizaiton namelist provided: {namelist}", typeMsg="i")

            self.optimization_options = IOtools.read_mitim_yaml(namelist)

        elif default_namelist_function is not None:
            print("\t- Optimizaiton namelist not provided, using MITIM default for this optimization sub-module", typeMsg="i")

            namelist = __mitimroot__ / "templates" / "namelist.optimization.yaml"
            self.optimization_options = IOtools.read_mitim_yaml(namelist)

            self.optimization_options = default_namelist_function(self.optimization_options)

        else:
            print("\t- No optimizaiton namelist provided (likely b/c for reading/plotting purposes)",typeMsg="i")
            self.optimization_options = None

        self.surrogate_parameters = {
            "parameters_combined": {},
            "surrogate_transformation_variables_alltimes": None,
            "surrogate_transformation_variables_lasttime": None,
            "transformationInputs": STEPtools.identity,  # Transformation of inputs
            "transformationOutputs": STEPtools.identityOutputs,  # Transformation of outputs
        }

        # Determine type of tensors to work with
        torch.set_default_dtype(self.tensor_options["dtype"])  # In case I forgot to specify a type explicitly, use as default (https://github.com/pytorch/botorch/discussions/1444)
        self.dfT = torch.randn( (2, 2), **tensor_options)

        # Name of calibrated objectives (e.g. QiRes1 to represent the objective from Qi1-QiT1)
        self.name_objectives = None

        # Name of transformed functions (e.g. Qi1_GB to represent the transformation of Qi1)
        self.name_transformed_ofs = None

        # Variables in the class not to save (e.g. COMSOL model)
        self.doNotSaveVariables = []

    def read(self, paramsfile, resultsfile):
        # Read stuff
        FolderEvaluation,numEval,inputFilePath,outputFilePath = IOtools.obtainGeneralParams(paramsfile, resultsfile)
        
        MITIMparams = IOtools.generateDictionaries(inputFilePath)
        dictDVs = MITIMparams["dictDVs"]
        dictOFs = MITIMparams["dictOFs"]

        # Do not store as part of the class not to confuse parallel evals
        return FolderEvaluation, numEval, dictDVs, dictOFs

    def write(self, dictOFs, resultsfile):
        IOtools.writeOFs(resultsfile, dictOFs)

    """
	**********************************************************************************************************************************************************
	Methods that need to be re-defined by children classes for specific applications
	**********************************************************************************************************************************************************
	"""

    def run(self, paramsfile, resultsfile):
        # Read stuff
        FolderEvaluation, numEval, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        # Operations (please, modify as needed when this class is used as parent)
        pass

        # Write stuff
        self.write(dictOFs, resultsfile)

    def scalarized_objective(self, Y):
        """
        * Receives Y as (batch1...N,dimY)
        * Must produce OF (batch1...N,dimYof), CAL (batch1...N,dimYof) and residual (batch1...N)
        * Notes:
                - Residual must be ready for maximization. It's used by the residual tracker (best) and some optimization algorithms
                - The reason why OF and CAL must be provided is because of visualization purposes of matching conditions and for optimization algorithms such as ROOT
                - Works with tensors
                - Here is where I control weights, relatives, etc
        """
        pass

    # **********************************************************************************************************************************************************

    def read_optimization_results(
        self,
        plotFN=None,
        folderRemote=None,
        analysis_level=0,
        pointsEvaluateEachGPdimension=50,
        rangePlot=None,
    ):
        with np.errstate(all="ignore"):
            LOGtools.ignoreWarnings()
            (
                self.fn,
                self.res,
                self.mitim_model,
                self.data,
            ) = BOgraphics.retrieveResults(
                self.folder,
                analysis_level=analysis_level,
                doNotShow=True,
                plotFN=plotFN,
                folderRemote=folderRemote,
                pointsEvaluateEachGPdimension=pointsEvaluateEachGPdimension,
                rangePlot=rangePlot,
            )

        # Make folders local
        try:
            self.mitim_model.folderOutputs = Path(str(self.mitim_model.folderOutputs).replace(
                str(self.mitim_model.folderExecution), str(self.folder)
            ))
            self.mitim_model.optimization_extra = self.mitim_model.optimization_object.optimization_extra = (
                Path(str(self.mitim_model.optimization_extra).replace(
                    str(self.mitim_model.folderExecution), str(self.folder)
                ))
            )
            self.mitim_model.folderExecution = self.mitim_model.optimization_object.folder = (
                self.folder
            )
        except:
            pass
            

    def analyze_optimization_results(self):
        print("- Analyzing MITIM BO results")

        # ----------------------------------------------------------------------------------------------------------------
        # Interpret stuff
        # ----------------------------------------------------------------------------------------------------------------

        if "res" not in self.__dict__.keys():
            self.read_optimization_results()
        variations_best = self.res.best_absolute_full["x"]
        variations_original = self.res.evaluations[0]["x"]

        print(
            f"\t- Best case in MITIM was achieved at evaluation #{self.res.best_absolute_index}:"
        )
        for ikey in variations_best:
            print(f"\t\t* {ikey} = {variations_best[ikey]}")

        try:
            self_complete = self.mitim_model.optimization_object
        except:
            self_complete = None
            print("\t- Problem retrieving function", typeMsg="w")

        return variations_original, variations_best, self_complete

    def plot_optimization_results(
        self,
        analysis_level=0,
        folderRemote=None,
        retrieval_level=None,
        plotYN=True,
        pointsEvaluateEachGPdimension=50,
        rangesPlot=None,
        save_folder=None,
        tabs_colors=0,
    ):
        time1 = datetime.datetime.now()

        if analysis_level < 0:
            print("\t- Only read optimization_results.out")
        if analysis_level == 0:
            print("\t- Only plot optimization_results.out")
        if analysis_level == 1:
            print("\t- Read optimization_results.out and pickle")
        if analysis_level == 2:
            print("\t- Perform full analysis")
        if analysis_level > 2:
            print(
                f"\t- Perform extra analysis for this sub-module (analysis level {analysis_level})"
            )

        if plotYN and (analysis_level >= 0):
            if "fn" not in self.__dict__:
                self.fn = GUItools.FigureNotebook("MITIM Optimization Results")
            
        self.read_optimization_results(
            plotFN=self.fn if (plotYN and (analysis_level >= 0)) else None,
            folderRemote=folderRemote,
            analysis_level= retrieval_level if (retrieval_level is not None) else analysis_level,
            pointsEvaluateEachGPdimension=pointsEvaluateEachGPdimension,
            rangePlot=rangesPlot,
        )

        self_complete = None
        if analysis_level > 1:
            """
            If the analyze_results exists, I'm in a child class, so just proceed to analyze.
            Otherwise, let's grab the method from the pickled
            """
            if hasattr(self, "analyze_results"):
                self_complete = self.analyze_results(
                    plotYN=plotYN, fn=self.fn, analysis_level=analysis_level
                )

            else:
                # What function is it?
                class_name = str(self.mitim_model.optimization_object).split()[0].split(".")[-1]
                print(
                    f'\t- Retrieving "analyze_results" method from class "{class_name}"',
                    typeMsg="i",
                )

                if class_name == "freegsu":
                    from mitim_modules.freegsu.FREEGSUmain import analyze_results
                elif class_name == "vitals":
                    from mitim_modules.vitals.VITALSmain import analyze_results
                elif class_name == "portals":
                    from mitim_modules.portals.PORTALSmain import analyze_results
                else:
                    analyze_results = None

                if analyze_results is not None:
                    self_complete = analyze_results(
                        self, plotYN=plotYN, fn=self.fn, analysis_level=analysis_level
                    )
                else:
                    print(
                        '\t- No "analyze_results" method found for this function class',
                        typeMsg="w",
                    )

        if plotYN and (analysis_level >= 0):
            print(f"\n- Plotting took {IOtools.getTimeDifference(time1)}")

            if save_folder is not None:
                self.fn.save(save_folder)

        return self_complete


# Main BO class that performs optimization
class MITIM_BO:
    def __init__(
        self,
        optimization_object,
        cold_start=False,
        storeClass=True,
        onlyInitialize=False,
        seed=0,
        askQuestions=True,
    ):
        """
        Inputs:
                - optimization_object   :  Function that is executed,
                        with .optimization_options in it (Dictionary with optimization parameters (must be obtained using namelist and read_mitim_yaml))
                        and .folder (Where the function runs)
                        and surrogate_parameters: Parameters to pass to surrogate (e.g. for transformed function), It can be different from function_parameters because of making evaluations fast.
                - cold_start 	 :  If False, try to find the values from Outputs/optimization_data.csv
                - storeClass 	 :  If True, write a class pickle for well-behaved cold_starting
                - askQuestions 	 :  To avoid that a SLURM job gets stop becuase something is asked, set to False
        """

        self.optimization_object = optimization_object
        self.cold_start = cold_start
        self.storeClass = storeClass
        self.askQuestions = askQuestions
        self.seed = seed
        self.avoidPoints = []
        
        if self.optimization_object.name_objectives is None:
            self.optimization_object.name_objectives = "y"

        # Folders and Logger
        self.folderExecution = IOtools.expandPath(self.optimization_object.folder) if (self.optimization_object.folder is not None) else Path("")

        self.folderOutputs = self.folderExecution / "Outputs"

        if (not self.cold_start) and askQuestions:
            
            # Check if Outputs folder is empty (if it's empty, do not ask the user, just continue)
            if self.folderOutputs.exists() and (len(list(self.folderOutputs.iterdir())) > 0):
                if not print(f"\t* Because {cold_start = }, MITIM will try to read existing results from folder",typeMsg="q"):
                    raise Exception("[MITIM] - User requested to stop")

        if optimization_object.optimization_options is not None:
            if not self.folderOutputs.exists():
                IOtools.askNewFolder(self.folderOutputs, force=True)

            """
			Prepare class where I will store some extra data
			---
			Do not carry out this dictionary through the workflow, just read and write
			"""

            self.optimization_extra = self.folderOutputs / "optimization_extra.pkl"

            # Read if exists
            exists = False
            if self.optimization_extra.exists():
                try:
                    dictStore = IOtools.unpickle_mitim(self.optimization_extra)
                    exists = True
                except (ModuleNotFoundError,EOFError):
                    exists = False
                    print('Problem loading "optimization_extra.pkl"',typeMsg="w")
            
            # nans if not
            if not exists:
                dictStore = {}
                for i in range(200):
                    dictStore[i] = np.nan

            # Write
            with open(self.optimization_extra, "wb") as handle:
                pickle_dill.dump(dictStore, handle, protocol=4)

            # Write the class into the optimization_object
            optimization_object.optimization_extra = self.optimization_extra

        # Function to execute
        self.surrogate_parameters = self.optimization_object.surrogate_parameters
        self.optimization_options = self.optimization_object.optimization_options

        if self.optimization_options is not None:

            # Check if the optimization options are in the namelist
            optimization_options_default = IOtools.read_mitim_yaml(__mitimroot__ / "templates" / "namelist.optimization.yaml")
            potential_flags = IOtools.deep_grab_flags_dict(optimization_options_default)
            IOtools.check_flags_mitim_namelist(
                self.optimization_options, potential_flags,
                avoid = ["stopping_criteria_parameters"], # Because they are specific to the stopping criteria
                askQuestions=askQuestions
                )

            # Write the optimization parameters stored in the object, into a file
            if self.optimization_object.folder is not None:
                IOtools.write_mitim_yaml(self.optimization_options, self.optimization_object.folder / "optimization.namelist.yaml")
                print(f" --> Optimization namelist written to {self.optimization_object.folder / 'optimization.namelist.yaml'}")
            
        # -------------------------------------------------------------------------------------------------

        if not onlyInitialize:
            
            """
			------------------------------------------------------------------------------
			Grab variables
			------------------------------------------------------------------------------
			"""
   
            self.timings_file = self.folderOutputs / "timing.jsonl"

            # Logger
            sys.stdout = LOGtools.Logger(logFile=self.folderOutputs / "optimization_log.txt", writeAlsoTerminal=True)

            print("\n-----------------------------------------------------------------------------------------")
            print("\t\t\t BO class module")
            print("-----------------------------------------------------------------------------------------\n")

            # Print machine resources
            IOtools.print_machine_info()

            # Meta
            self.numIterations = self.optimization_options["convergence_options"]["maximum_iterations"]
            self.strategy_options = self.optimization_options["strategy_options"]
            self.parallel_evaluations = self.optimization_options["evaluation_options"]["parallel_evaluations"]
            self.dfT = self.optimization_object.dfT

            """
			Notes about the "avoidPoints" variables
			---------------------------------------
				The avoidPoints_failed list is updated in the following instances:
					- When evaluating initial batch, result has failed for this simulation
					- When updating the set, result has failed for this simulation
				The avoidPoints_outside list is updated in the following instances:
					- When reducing the trust region, DV fall outside of new bounds
				After each iteration, avoidPoints is re-constructed with the failed points and the outside points.
				This logic is because when expanding the TR, points may fall again into the TR, so I don't
				want to keep the same track iteration after iteration
			"""
            self.avoidPoints_failed, self.avoidPoints_outside = [], []

            # Initialize the metrics as a dictionary with fixed size of number of iterations that I will be updating
            self.hard_finish = False
            self.numEval = 0
            self.keys_metrics = [
                "BOratio",
                "xBest_track",
                "yBest_track",
                "yVarBest_track",
                "BOmetric",
                "TRoperation",
                "BoundsStorage",
                "iteration",
                "BOmetric_it",
            ]
            self.BOmetrics = {"overall": {}}
            for ikey in self.keys_metrics:
                self.BOmetrics[ikey] = {}
                for i in range(self.optimization_options["convergence_options"]["maximum_iterations"] + 1):
                    self.BOmetrics[ikey][i] = np.nan

            """
			------------------------------------------------------------------------------
			Prepare Desgin variables (DVs) with minimum and maximum values
			------------------------------------------------------------------------------
			"""

            self.bounds, self.boundsInitialization = OrderedDict(), []
            for cont, i in enumerate(self.optimization_options["problem_options"]["dvs"]):
                self.bounds[i] = np.array([self.optimization_options["problem_options"]["dvs_min"][cont], self.optimization_options["problem_options"]["dvs_max"][cont]])
                self.boundsInitialization.append(np.array([self.optimization_options["problem_options"]["dvs_min"][cont], self.optimization_options["problem_options"]["dvs_max"][cont]]))

            self.boundsInitialization = np.transpose(self.boundsInitialization)

            # Bounds may change during the workflow (corrections, TURBO)
            self.bounds_orig = copy.deepcopy(self.bounds)

            """
			----------------------------------------------------------------------------------------------------------------------------
			Prepare Objective functions (OFs) with minimum and maximum values
			----------------------------------------------------------------------------------------------------------------------------
			"""

            # Objective functions (OFs)
            self.outputs = self.surrogate_parameters["outputs"] = self.optimization_options["problem_options"]["ofs"]

            # How many points each iteration will produce?
            self.best_points_sequence = self.optimization_options["acquisition_options"]["points_per_step"]
            self.best_points = int(np.sum(self.best_points_sequence))

            """
			------------------------------------------------------------------------------
			Prepare Initialization
			------------------------------------------------------------------------------
			"""

            if (
                (self.optimization_options["initialization_options"]["type_initialization"] == 1)
                and ((self.folderExecution / "Execution" / "Evaluation.1").exists())
                and (self.cold_start)
            ):
                print("\t--> Random initialization has been requested",typeMsg="q" if self.askQuestions else "qa")

            self.type_initialization = self.optimization_options["initialization_options"]["type_initialization"]
            self.initial_training = self.optimization_options["initialization_options"]["initial_training"]

            """
			------------------------------------------------------------------------------
			Initialize Output files
			------------------------------------------------------------------------------
			"""

            if (self.type_initialization == 3) and (self.cold_start):
                print("\t* Initialization based on Tabular, yet cold_start has been requested. I am NOT removing the previous optimization_data",typeMsg="w")
                if self.askQuestions:
                    flagger = print("\t\t* Are you sure this was your intention?", typeMsg="q")
                    if not flagger:
                        embed()
                forceNewTabulars = False
            else:
                forceNewTabulars = self.cold_start

            inputs = [i for i in self.bounds]

            self.scalarized_objective = self.optimization_object.scalarized_objective

            self.optimization_data = BOgraphics.optimization_data(
                inputs,
                self.outputs,
                file=self.folderOutputs / "optimization_data.csv",
                forceNew=forceNewTabulars,
            )

            # If the file turned out to be empty, I will force it to be new
            if forceNewTabulars and (len(self.optimization_data.data) == 0):
                print("\t* Tabular file is empty, forcing new, to avoid radii/channel specifications from dummy sims",typeMsg="w")
                self.optimization_data = BOgraphics.optimization_data(
                    inputs,
                    self.outputs,
                    file=self.folderOutputs / "optimization_data.csv",
                    forceNew=True,
                )

            res_file = self.folderOutputs / "optimization_results.out"

            """
			------------------------------------------------------------------------------
			Parameters that will be needed at each step (unchanged)
			------------------------------------------------------------------------------
			"""

            self.stepSettings = {
                "optimization_options": self.optimization_options,
                "dfT": self.dfT,
                "bounds_orig": self.bounds_orig,
                "best_points_sequence": self.best_points_sequence,
                "folderOutputs": self.folderOutputs,
                "fileOutputs": res_file,
                "name_objectives": self.optimization_object.name_objectives,
                "name_transformed_ofs": self.optimization_object.name_transformed_ofs,
                "outputs": self.outputs,
            }

            self.optimization_results = BOgraphics.optimization_results(file=res_file)

            self.optimization_results.initialize(self)

    def run(self):
        """
        Notes:
                - self.train_X,self.train_Y are still provided in absolute units, not normalized
                - train_Ystd is in standard deviations (square root of the variance), not normalized and not relative
        """

        timeBeginning = datetime.datetime.now()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Initialization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.initializeOptimization()

        self.currentIteration = -1

        # Has the problem reached convergence in the training?
        converged,_ = self.optimization_options['convergence_options']['stopping_criteria'](self, parameters = self.optimization_options['convergence_options']['stopping_criteria_parameters'])
        if converged:
            print("- Optimization has converged in training!",typeMsg="i")
            self.numIterations = 0

        # If no iterations are requested, just run the training step
        if self.numIterations == 0:
            print("- No BO iterations requested, workflow will stop after running a training step (to enable reading later)",typeMsg="i")
            self.numIterations = 1
            self.hard_finish = True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Iterative workflow
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.strategy_options_use = self.strategy_options

        self.steps, self.resultsSet = [], []
        for self.currentIteration in range(self.numIterations+1):
            timeBeginningThis = datetime.datetime.now()

            print("\n------------------------------------------------------------")
            print(f'\tMITIM Step {self.currentIteration} ({timeBeginningThis.strftime("%Y-%m-%d %H:%M:%S")})')
            print("------------------------------------------------------------")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~ Update training population with next points
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if self.currentIteration > 0:
                print(f"--> Proceeding to updating set (which currently has {len(self.train_X)} points)")

                # *NOTE*: self.x_next has been updated earlier, either from a cold_start or from the full workflow
                yN, yNstd = self.updateSet(self.strategy_options_use)

                # Stored in previous step
                self.steps[-1].y_next = yN
                self.steps[-1].ystd_next = yNstd

                # Determine here when to stop the loop
                if self.currentIteration > self.numIterations - 1:
                    print("- Last iteration has been reached",typeMsg="i")
                    self.hard_finish = True

            # After evaluating metrics inside updateSet, I may have requested a hard finish
            if self.hard_finish:
                print("- Hard finish has been requested", typeMsg="i")

                # Removing those spaces in the metrics that were not filled up
                for ikey in self.keys_metrics:
                    for i in range(self.currentIteration + 1, self.optimization_options["convergence_options"]["maximum_iterations"] + 1):
                        del self.BOmetrics[ikey][i]
                # ------------------------------------------------------------------------------------------

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~ Perform BO step
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Does the tabular file include at least as many rows as requested to be run in this step?
            pointsTabular = len(self.optimization_data.data)  # Number of points that the Tabular contains
            pointsExpected = len(self.train_X) + self.best_points
            if not self.cold_start:
                if not pointsTabular >= pointsExpected:
                    print(f"--> CSV file does not contain information for all points ({pointsTabular}/{pointsExpected}), disabling cold_starting-from-previous from this point on",typeMsg="w", )
                    self.cold_start = True
                else:
                    print(f"--> CSV file contains at least as many points as expected at this stage ({pointsTabular}/{pointsExpected})",typeMsg="i",)

            # In the case of starting from previous, do not run BO process.
            if not self.cold_start:
                """
                Philosophy of cold_starting is:
                        - cold_starting requires that settings are the same (e.g. best_points).
                        - cold_starting requires Tabular data, since I will be grabbing from it.
                        - The pkl file in cold_starting is only used to store the "step" data with opt info, but not
                                required to continue. But here I enforce it anyway. I want to know what I have done.
                        - Later I pass the x_next from the pickle, so I could just trust the pickle if I wanted.
                """

                # Read step from pkl
                current_step = self.read()

                if current_step is None:
                    print("\t* Because reading pkl step had problems, disabling cold_starting-from-previous from this point on",typeMsg="w")
                    print("\t* Are you aware of the consequences of continuing?",typeMsg="q")

                    self.cold_start = True

            if not self.cold_start:
                # Read next from Tabular
                self.x_next, _, _ = self.optimization_data.extract_points(points=np.arange(len(self.train_X), len(self.train_X) + self.best_points))
                self.x_next = torch.from_numpy(self.x_next).to(self.dfT)

                # Re-write x_next from the pkl... reason for this is that if optimization is heuristic, I may prefer what was in Tabular
                if current_step is not None:
                    current_step.x_next = self.x_next

                # If there is any Nan, assume that I cannot cold_start this step
                if IOtools.isAnyNan(self.x_next.cpu()):
                    print("\t* Because x_next points have NaNs, disabling cold_starting-from-previous from this point on",typeMsg="w")
                    self.cold_start = True

                # Step is valid, append to this current one
                if not self.cold_start:
                    self.steps.append(current_step)

                # When cold_starting, make sure that the strategy options are preserved (like correction, bounds and TURBO)
                self.strategy_options_use = current_step.strategy_options_use

                print("\t* Step successfully restarted from pkl file", typeMsg="i")

            # Standard (i.e. start from beginning, not read values)
            if self.cold_start:
                # For standard use, use the actual strategy_options launched
                self.strategy_options_use = self.strategy_options

                # Remove from tabular next points in case they were there. Since I'm not cold_starting, I don't care about what has come next
                self.optimization_data.removePointsAfter(len(self.train_X) - 1)

                """
				---------------------------------------------------------------------------------------
				BOstep is in charge to fit models and optimize objective function
							(inputs and returns are unnormalized)
				---------------------------------------------------------------------------------------
				"""

                self._step()


            # Pass the information about next step
            self.x_next = self.steps[-1].x_next

            # ~~~~~~~~ Store class now with the next points found (after optimization)
            if self.storeClass and self.cold_start:
                self.save()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.hard_finish:
                break

        self.save()

        print(f"- Complete MITIM workflow took {IOtools.getTimeDifference(timeBeginning)} ~~")
        print("********************************************************\n")

    def prepare_for_save_MITIMBO(self, copyClass):
        """
        Downselect what elements to store
        """

        # -------------------------------------------------------------------------------------------------
        # To avoid circularity when cold_starting, do not store the class in the optimization_results sub-class
        # -------------------------------------------------------------------------------------------------

        del copyClass.optimization_results.MITIM_BO

        # -------------------------------------------------------------------------------------------------
        # Saving state files with functions is very expensive (deprecated maybe when I had lambdas?) [TODO: Remove]
        # -------------------------------------------------------------------------------------------------

        del copyClass.scalarized_objective

        for i in range(len(self.steps)):
            if "functions" in copyClass.steps[i].__dict__:
                del copyClass.steps[i].functions
            if "evaluators" in copyClass.steps[i].__dict__:
                del copyClass.steps[i].evaluators

        # -------------------------------------------------------------------------------------------------
        # Add time stamp
        # -------------------------------------------------------------------------------------------------

        copyClass.timeStamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return copyClass

    def prepare_for_read_MITIMBO(self, copyClass):
        """
        Repair the downselection
        """

        copyClass.optimization_results.MITIM_BO = copy.deepcopy(self)

        copyClass.scalarized_objective = copyClass.optimization_object.scalarized_objective

        for i in range(len(copyClass.steps)):
            copyClass.steps[i].defineFunctions(copyClass.scalarized_objective)

        return copyClass

    def save(self, name="optimization_object.pkl"):
        print("* Proceeding to save new MITIM state pickle file")
        stateFile = self.folderOutputs / f"{name}"
        stateFile_tmp = self.folderOutputs / f"{name}_tmp"

        # Do not store certain variables (that cannot even be copied, that's why I do it here)
        saver = {}
        for ikey in self.optimization_object.doNotSaveVariables:
            saver[ikey] = self.optimization_object.__dict__[ikey]
            del self.optimization_object.__dict__[ikey]
        # -----------------------------------------------------------------------------------

        copyClass = self.prepare_for_save_MITIMBO(copy.deepcopy(self))

        with open(stateFile_tmp, "wb") as handle:
            try:
                pickle_dill.dump(copyClass, handle, protocol=4)
            except:
                print(f"\t* Problem saving {name}, trying without the optimization_object, but that will lead to limiting applications. I recommend you populate self.optimization_object.doNotSaveVariables = ['variable1', 'variable2'] with the variables you think cannot be pickled", typeMsg="w")
                del copyClass.optimization_object
                pickle_dill.dump(copyClass, handle, protocol=4)

        # Get variables back ----------------------------------------------------------------
        for ikey in saver:
            self.optimization_object.__dict__[ikey] = saver[ikey]
        # -----------------------------------------------------------------------------------

        stateFile_tmp.replace(stateFile)  # This way I reduce the risk of getting a mid-creation file

        print(f"\t- MITIM state file {IOtools.clipstr(stateFile)} generated, containing the MITIM_BO class")

    def read(self, name="optimization_object.pkl", iteration=None, file=None, provideFullClass=False):
        iteration = iteration or self.currentIteration

        print("- Reading pickle file with optimization_object class")
        stateFile = file if (file is not None) else self.folderOutputs / f"{name}"

        try:
            # If I don't create an Individual attribute I cannot unpickle GA information
            try:
                import deap
                deap.creator.create("Individual", array.array)
            except:
                pass

            aux = IOtools.unpickle_mitim(stateFile)

            aux = self.prepare_for_read_MITIMBO(aux)

            step = aux.steps[iteration]
            print(f"\t* Read {IOtools.clipstr(stateFile)} state file, grabbed step #{iteration}",typeMsg="i")
        
        except FileNotFoundError:
            print(f"\t- State file {IOtools.clipstr(stateFile)} not found", typeMsg="w")
            step, aux = None, None
        
        except IndexError:
            print(f"\t- State file {IOtools.clipstr(stateFile)} does not have all iterations required to continue from it", typeMsg="w")
            step = None

        return aux if provideFullClass else step

    # Convenient helper methods to track timings of components

    @mitim_timer(lambda self: f'Eval @ {self.currentIteration}', log_file=lambda self: self.timings_file)
    def _evaluate(self):
        
        y_next, ystd_next, self.numEval = EVALUATORtools.fun(
            self.optimization_object,
            self.x_next,
            self.folderExecution,
            self.bounds,
            self.outputs,
            self.optimization_data,
            parallel=self.parallel_evaluations,
            cold_start=self.cold_start,
            numEval=self.numEval,
        )
        
        return y_next, ystd_next
    
    @mitim_timer(lambda self: f'Surr @ {self.currentIteration}', log_file=lambda self: self.timings_file)
    def _step(self):
        
        train_Ystd = self.train_Ystd if (self.optimization_options["evaluation_options"]["train_Ystd"] is None) else self.optimization_options["evaluation_options"]["train_Ystd"]
        
        current_step = STEPtools.OPTstep(
            self.train_X,
            self.train_Y,
            train_Ystd,
            bounds=self.bounds,
            stepSettings=self.stepSettings,
            currentIteration=self.currentIteration,
            strategy_options=self.strategy_options_use,
            BOmetrics=self.BOmetrics,
            surrogate_parameters=self.surrogate_parameters,
        )

        # Incorporate strategy_options for later retrieving
        current_step.strategy_options_use = copy.deepcopy(self.strategy_options_use)

        self.steps.append(current_step)

        # Avoid points
        avoidPoints = np.append(self.avoidPoints_failed, self.avoidPoints_outside)
        self.avoidPoints = np.unique([int(j) for j in avoidPoints])

        # ***** Fit
        self.steps[-1].fit_step(avoidPoints=self.avoidPoints)

        # ***** Define evaluators
        self.steps[-1].defineFunctions(self.scalarized_objective)

        # Store class with the model fitted and evaluators defined
        if self.storeClass:
            self.save()

        # ***** Optimize
        if not self.hard_finish:
            self.steps[-1].optimize(
                position_best_so_far=self.BOmetrics["overall"]["indBest"],
                seed=self.seed,
            )
        else:
            self.steps[-1].x_next = None
        
    # ---------------------------------------------------------------------------------


    def updateSet(
        self, strategy_options_use, isThisCorrected=False, ForceNotApplyCorrections=False
    ):
        # ~~~~~~~~~~~~~~~~~~
        # What's the expected value of the next points?
        # ~~~~~~~~~~~~~~~~~~

        y, u, l, _ = self.steps[-1].GP["combined_model"].predict(self.x_next)
        self.y_next_pred = y.detach()
        self.y_next_pred_u = u.detach()
        self.y_next_pred_l = l.detach()

        # ~~~~~~~~~~~~~~~~~~
        # What's the actual value of the next points? Insert them in the database
        # ~~~~~~~~~~~~~~~~~~

        # Update the train_X
        self.train_X = np.append(self.train_X, self.x_next.cpu(), axis=0)

        # Update optimization_data with nans
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())

        # Update optimization_results only as "predicted"
        if not isThisCorrected:
            self.optimization_results.addPoints(
                includePoints=[
                    len(self.train_X) - len(self.x_next),
                    len(self.train_X) - len(self.x_next) + 1,
                ],
                executed=False,
                predicted=True,
                Best=True,
            )
            self.optimization_results.addPoints(
                includePoints=[len(self.train_X) - len(self.x_next), len(self.train_X)],
                executed=False,
                predicted=True,
                Name=f"Evaluating points from iteration {self.currentIteration}, comprised of {len(self.x_next)} points",
            )

        # --- Evaluation
        time1 = datetime.datetime.now()
        y_next, ystd_next = self._evaluate()
        txt_time = IOtools.getTimeDifference(time1)
        print(f"\t- Complete model update took {txt_time}")
        # ------------------

        # Update the train_Y
        self.train_Y = np.append(self.train_Y, y_next, axis=0)
        self.train_Ystd = np.append(self.train_Ystd, ystd_next, axis=0)

        # --- If problem in evaluation don't use this point -------------------------------------------------------------------
        for i in range(self.train_Y.shape[0]):
            boole = (np.isinf(self.train_Y[i]).any()) and (
                i not in self.avoidPoints_failed
            )
            if boole:
                self.avoidPoints_failed.append(i)
        if len(self.avoidPoints_failed) > 0:
            print(
                f"\t- Points {self.avoidPoints_failed} are avoided b/c at least one of the OFs could not be computed"
            )
        # ---------------------------------------------------------------------------------------------------------------------

        # Update Tabular data with the actual evaluations
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())

        # Update optimization_results with the actual evaluations
        if not isThisCorrected:
            txt = f"Evaluating points from iteration {self.currentIteration}, comprised of {len(self.x_next)} points"
            predicted, forceWrite, addheader = True, True, True
        else:
            txt = f"Evaluating further points after trust region operation... batch comprised of {len(self.x_next)} points"
            predicted, forceWrite, addheader = False, False, False
        self.optimization_results.addPoints(
            includePoints=[len(self.train_X) - len(self.x_next), len(self.train_X)],
            executed=True,
            predicted=predicted,
            Name=txt,
            forceWrite=forceWrite,
            addheader=addheader,
            timingString=txt_time,
        )

        """
		~~~~~~~~~~~~~~~~~~
		If the optimization step has allowed out-of-bounds points, I should here upgrade my original bounds if the point chosen was out.
		This is not really an option, it must always happen. Otherwise it doesn't make sense to allow extrapolations
		~~~~~~~~~~~~~~~~~~
		"""
        print("\n~~~~~~~~~~~~~~~ Entering bounds upgrade module ~~~~~~~~~~~~~~~~~~~")
        print("(if extrapolations were allowed during optimization)")
        self.bounds = SBOcorrections.upgradeBounds(self.bounds, self.train_X, self.avoidPoints_outside)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        # ~~~~~~~~~~~~~~~~~~
        # Possible corrections to modeled & optimization region
        # ~~~~~~~~~~~~~~~~~~

        if not isThisCorrected:
            SBOcorrections.updateMetrics(
                self,
                evaluatedPoints=self.x_next.shape[0],
                position=self.currentIteration,
            )

            changesMade = 0
            # Apply TURBO
            if strategy_options_use["TURBO_options"]["apply"]:
                changesMade = SBOcorrections.TURBOupdate(
                    self,
                    strategy_options_use,
                    position=self.currentIteration,
                    seed=self.seed,
                )
            # Apply some corrections
            if strategy_options_use["applyCorrections"] and not ForceNotApplyCorrections:
                changesMade = SBOcorrections.correctionsSet(self, strategy_options_use)

            if changesMade > 0:
                print(
                    f"\t~ {changesMade} correction strategies implemented, requesting a new evaluation of metrics in this new region"
                )

                # Get the metrics because I may have changed the index of which one of the trained is the best
                SBOcorrections.updateMetrics(
                    self,
                    IsThisAFreshIteration=False,
                    evaluatedPoints=self.x_next.shape[0],
                    position=self.currentIteration,
                )

        # ~~~~~~~~~~~~~~~~~~
        # Stopping criteria
        # ~~~~~~~~~~~~~~~~~~

        converged,_ = self.optimization_options['convergence_options']['stopping_criteria'](self, parameters = self.optimization_options['convergence_options']['stopping_criteria_parameters'])

        if converged:
            self.hard_finish = self.hard_finish or True
            print("- * Optimization considered converged *", typeMsg="w")

        return y_next, ystd_next

    @mitim_timer(lambda self: f'Init', log_file=lambda self: self.timings_file)
    def initializeOptimization(self):
        print("\n")
        print("------------------------------------------------------------")
        print(" Problem initialization")
        print("------------------------------------------------------------\n")

        # Print Optimization Settings

        print("\n==============================================================================")
        print(f"  {IOtools.getStringFromTime()}, Starting MITIM Optimization")
        print("==============================================================================")

        print(f"* Folder: {self.folderExecution}")
        print("* Optimization Settings:")
        for i in self.optimization_options:
            strs = f"\t\t{i:25}:"
            if i in ["strategy_options", "surrogate_options"]:
                print(strs)
                for j in self.optimization_options[i]:
                    print(f"\t\t\t{j:25}: {self.optimization_options[i][j]}")
            else:
                print(f"{strs} {self.optimization_options[i]}")

        print("* Main Function Parameters:")
        par = self.optimization_object.__dict__
        for i in par:
            if i not in ["optimization_options"]:
                strs = f"\t\t{i:25}:"
                if "file_in_lines_" not in i:
                    print(f"{strs} {par[i]}")
                else:
                    print(f"{strs} NOT PRINTED")

        # --------------------------------------------------------------------------------------------------

        self.Originalinitial_training = copy.deepcopy(self.initial_training)

        # -----------------------------------------------------------------
        # Force certain optimizations depending on existence of folders
        # -----------------------------------------------------------------

        if (not self.cold_start) and (self.optimization_data is not None):
            self.type_initialization = 3
            print("--> Since restart from a previous MITIM has been requested, forcing initialization type to 3 (read from optimization_data)",typeMsg="i",)

        if self.type_initialization == 3:
            print("--> Initialization by reading tabular data...")

            try:
                tabExists = len(self.optimization_data.data) >= self.initial_training
                print(f"\t- optimization_data file has {len(self.optimization_data.data)} elements, and initial_training were {self.initial_training}")
            except:
                tabExists = False
                print("\n\nCould not read Tabular, because:", typeMsg="w")
                print(traceback.format_exc())

            if not tabExists:
                print("--> type_initialization 3 requires optimization_data but something failed. Assigning type_initialization=1 and cold_starting from scratch",typeMsg="i",)
                if self.askQuestions:
                    flagger = print("Are you sure?", typeMsg="q")
                    if not flagger:
                        embed()

                self.type_initialization = 1
                self.cold_start = True

        # -----------------------------------------------------------------
        # Initialization
        # -----------------------------------------------------------------

        readCasesFromTabular = (not self.cold_start) or self.optimization_options["initialization_options"]["read_initial_training_from_csv"]  # Read when starting from previous or forced it

        # cold_started run from previous. Grab DVs of initial set
        if readCasesFromTabular:
            try:
                self.train_X, self.train_Y, self.train_Ystd = self.optimization_data.extract_points(points=np.arange(self.initial_training))

                # It could be the case that those points in Tabular are outside the bounds that I want to apply to this optimization, remove outside points?
                
                if self.optimization_options["initialization_options"]["ensure_within_bounds"]:
                    for i in range(self.train_X.shape[0]):
                        insideBounds = TESTtools.checkSolutionIsWithinBounds(
                            torch.from_numpy(self.train_X[i, :]).to(self.dfT),
                            torch.from_numpy(np.array(list(self.bounds.values())).T),
                        )
                        if not insideBounds.item():
                            self.avoidPoints_outside.append(i)

            except:
                flagger = print("Error reading Tabular. Do you want to continue without cold_start and do standard initialization instead?",typeMsg="q",)

                self.type_initialization = 1
                self.cold_start = True
                readCasesFromTabular = False

            if readCasesFromTabular and IOtools.isAnyNan(self.train_X):
                flagger = print(" --> cold_start requires non-nan DVs, doing normal initialization",typeMsg="q",)
                if not flagger:
                    embed()

                self.type_initialization = 1
                self.cold_start = True
                readCasesFromTabular = False

        # Standard - RUN

        if not readCasesFromTabular:
            if self.type_initialization == 1 and self.optimization_options["problem_options"]["dvs_base"] is not None:
                self.initial_training = self.initial_training - 1
                print(f"--> Baseline point has been requested with LHS initialization, reducing requested initial random set to {self.initial_training}",typeMsg="i",)

            """
			Initialization
			--------------
			"""

            if self.optimization_options["initialization_options"]["initialization_fun"] is None:
                if self.type_initialization == 1:
                    if self.initial_training == 0:
                        self.train_X = np.atleast_2d(
                            [i for i in self.optimization_options["problem_options"]["dvs_base"]]
                        )
                    else:
                        self.train_X = SAMPLINGtools.LHS(
                            self.initial_training,
                            self.boundsInitialization,
                            seed=self.seed,
                        )
                        self.train_X = self.train_X.cpu().numpy().astype("float")

                        # if (self.optimization_options['problem_options']['dvs_base'] is not None):
                        # 	self.train_X = np.append(np.atleast_2d([i for i in self.optimization_options['problem_options']['dvs_base']]),self.train_X,axis=0)

                elif self.type_initialization == 2:
                    raise Exception("Option not implemented yet")
                elif self.type_initialization == 3:
                    self.train_X = SAMPLINGtools.readInitializationFile(
                        self.folderExecution / "Outputs" / "optimization_data.csv",
                        self.initial_training,
                        self.stepSettings["optimization_options"]["problem_options"]["dvs"],
                    )
                elif self.type_initialization == 4:
                    self.train_X = IOtools.readExecutionParams(
                        self.folderExecution, nums=[0, self.initial_training - 1]
                    )

                if (
                    (self.type_initialization == 1)
                    and (self.optimization_options["problem_options"]["dvs_base"] is not None)
                    and (self.initial_training > 0)
                ):
                    self.train_X = np.append(
                        np.atleast_2d([i for i in self.optimization_options["problem_options"]["dvs_base"]]),
                        self.train_X,
                        axis=0,
                    )

            else:
                print("- Initialization function has been selected", typeMsg="i")
                self.train_X = self.optimization_options["initialization_options"]["initialization_fun"](self)
                readCasesFromTabular = True

            # Initialize train_Y as nan until evaluated
            self.train_Y = (
                np.ones((self.Originalinitial_training, len(self.outputs))) * np.nan
            )
            self.train_Ystd = (
                np.ones((self.Originalinitial_training, len(self.outputs))) * np.nan
            )

        # -----------------------------------------------------------------
        # Write prior to evaluation
        # -----------------------------------------------------------------

        # Write initialization in Tabular
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())

        # Write optimization_results
        self.optimization_results.addPoints(
            includePoints=[0, self.Originalinitial_training],
            executed=False,
            predicted=False,
            Name=f"Initial trust region, comprised of {self.Originalinitial_training} points",
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Evaluate initial training set
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        time1 = datetime.datetime.now()

        self.train_Y, self.train_Ystd, self.numEval = EVALUATORtools.fun(
            self.optimization_object,
            self.train_X,
            self.folderExecution,
            self.bounds,
            self.outputs,
            self.optimization_data,
            parallel=self.parallel_evaluations,
            cold_start=(not readCasesFromTabular),
            numEval=self.numEval,
        )

        txt_time = IOtools.getTimeDifference(time1)
        print(f"\t- Complete model initial training took: {txt_time}\n")
        # ------------------

        # --- If nan for important outputs don't use this point
        for i in range(self.train_Y.shape[0]):
            boole = (np.isinf(self.train_Y[i]).any()) and (
                i not in self.avoidPoints_failed
            )
            if boole:
                self.avoidPoints_failed.append(i)
        if len(self.avoidPoints_failed) > 0:
            print(
                f"\t- Points {self.avoidPoints_failed} are avoided b/c at least one of the OFs could not be computed"
            )
        # ------------------
        _,_,objective = self.optimization_object.scalarized_objective(torch.from_numpy(self.train_Y))
        self.optimization_data.update_points(self.train_X, Y=self.train_Y, Ystd=self.train_Ystd, objective=objective.cpu().numpy())
        self.optimization_results.addPoints(
            includePoints=[0, self.Originalinitial_training],
            executed=True,
            predicted=False,
            timingString=txt_time,
            Name=f"Initial trust region, comprised of {self.Originalinitial_training} points",
        )

        # Get some metrics about this iteration
        SBOcorrections.updateMetrics(
            self, evaluatedPoints=self.Originalinitial_training, position=0
        )

        # Make sure is 2D
        self.train_X = np.atleast_2d(self.train_X)

        """
		Some initialization strategies may create points outside of the original bounds, but I may want to include them!
		"""
        if self.optimization_options["initialization_options"]["expand_bounds"]:
            for i, ikey in enumerate(self.bounds):
                self.bounds[ikey][0] = np.min(
                    [self.bounds[ikey][0], self.train_X.min(axis=0)[i]]
                )
                self.bounds[ikey][1] = np.max(
                    [self.bounds[ikey][1], self.train_X.max(axis=0)[i]]
                )

    def plot(
        self,
        fn=None,
        plotoptimization_results=True,
        doNotShow=False,
        number_of_models_per_tab=5,
        stds=2,
        pointsEvaluateEachGPdimension=50,
        rangePlot_force=None,
    ):
        print(
            "\n ***************************************************************************"
        )
        print(f"* MITIM plotting module - Generic ({stds}sigma models)")
        print(
            "***************************************************************************\n"
        )

        GPs = self.steps

        if doNotShow:
            plt.ioff()

        if fn is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            geometry = (
                "1200x1000" if len(GPs[0].GP["individual_models"]) == 1 else "1700x1000"
            )
            fn = FigureNotebook("MITIM BO Strategy", geometry=geometry)

        """
		****************************************************************
		Model Stuff
		****************************************************************
		"""

        number_of_models_per_tab = np.min([number_of_models_per_tab, len(self.outputs)])
        Tabs_needed = int(np.ceil(len(self.outputs) / number_of_models_per_tab))

        if rangePlot_force is not None:
            rangePlot = rangePlot_force[:len(GPs)]
        else:
            if len(GPs) == 1:
                rangePlot = [0]
            elif len(GPs) == 2:
                rangePlot = range(len(GPs))
            else:
                rangePlot = [len(GPs) - 2, len(GPs) - 1]

        for ck, k in enumerate(rangePlot):
            print(
                f"- Plotting MITIM step #{k} information ({ck+1}/{len(rangePlot)})... {len(self.outputs)} GP models need to be plotted ({pointsEvaluateEachGPdimension} points/dim)..."
            )

            tab_color = ck + 5

            figsFund, figs, figsFundTrain = [], [], []
            for i in range(Tabs_needed):
                figsFund.append(
                    fn.add_figure(
                        label=f"#{k} Fundamental Surr. ({i+1}/{Tabs_needed})",
                        tab_color=tab_color,
                    )
                )
            for i in range(Tabs_needed):
                figsFundTrain.append(
                    fn.add_figure(
                        label=f"#{k} Fundamental Train ({i+1}/{Tabs_needed})",
                        tab_color=tab_color,
                    )
                )
            for i in range(Tabs_needed):
                figs.append(
                    fn.add_figure(
                        label=f"#{k} Surrogate ({i+1}/{Tabs_needed})",
                        tab_color=tab_color,
                    )
                )

            grid = plt.GridSpec(
                nrows=5, ncols=number_of_models_per_tab, hspace=0.5, wspace=0.3
            )
            gridTrain = plt.GridSpec(
                nrows=3, ncols=number_of_models_per_tab, hspace=0.3, wspace=0.3
            )

            for i in range(len(self.outputs)):
                GP = GPs[k].GP["individual_models"][i]

                figIndex = i // number_of_models_per_tab
                figIndex_inner = i % number_of_models_per_tab

                fig = figs[figIndex]
                figFund = figsFund[figIndex]
                figFundTrain = figsFundTrain[figIndex]

                x_next = None
                if "x_next" in GPs[k].__dict__.keys():
                    x_next = GPs[k].x_next
                x_best = self.BOmetrics["xBest_track"][k].unsqueeze(0)

                y_best = self.BOmetrics["yBest_track"][k]
                yVar_best = self.BOmetrics["yVarBest_track"][k]

                # --------------------------------------------------------------------
                # Plotting of models
                # --------------------------------------------------------------------

                # Fundamental Surrogates

                plotFundamental = True

                dimX = GP.gpmodel.ard_num_dims if plotFundamental else len(GP.bounds)
                if dimX == 1:
                    ax0 = figFund.add_subplot(grid[0:4, figIndex_inner])
                    ax1 = None
                    axL = figFund.add_subplot(grid[4, figIndex_inner])
                elif dimX == 2:
                    ax0 = figFund.add_subplot(grid[:2, figIndex_inner])
                    ax1 = figFund.add_subplot(grid[2:4, figIndex_inner], sharex=ax0)
                    axL = figFund.add_subplot(grid[4, figIndex_inner])
                else:
                    ax0 = figFund.add_subplot(grid[:2, figIndex_inner])
                    ax1 = figFund.add_subplot(grid[2:4, figIndex_inner])
                    axL = figFund.add_subplot(grid[4:, figIndex_inner])

                if y_best is not None:
                    y_best1, yVar_best1 = y_best[i], yVar_best[i]
                else:
                    y_best1, yVar_best1 = None, None
                GP.plot(
                    axs=[ax0, ax1, axL],
                    x_next=x_next,
                    x_best=x_best,
                    y_best=y_best1,
                    yVar_best=yVar_best1,
                    plotFundamental=plotFundamental,
                    stds=stds,
                    pointsEvaluate=pointsEvaluateEachGPdimension,
                )

                # Fundamental Surrogates - Training

                relative_to = -1

                ax0 = figFundTrain.add_subplot(gridTrain[0, figIndex_inner])
                ax1 = figFundTrain.add_subplot(gridTrain[1, figIndex_inner], sharex=ax0)
                ax2 = figFundTrain.add_subplot(gridTrain[2, figIndex_inner], sharex=ax0)

                GP.plotTraining(
                    axs=[ax0, ax1, ax2],
                    relative_to=relative_to,
                    figIndex_inner=figIndex_inner,
                    stds=stds,
                )

                # Optimization-relevant variables

                plotFundamental = False

                dimX = GP.dimX if plotFundamental else len(GP.bounds)
                if dimX == 1:
                    ax0 = fig.add_subplot(grid[0:4, figIndex_inner])
                    ax1 = None
                    axL = fig.add_subplot(grid[4, figIndex_inner])
                elif dimX == 2:
                    ax0 = fig.add_subplot(grid[:2, figIndex_inner])
                    ax1 = fig.add_subplot(grid[2:4, figIndex_inner], sharex=ax0)
                    axL = fig.add_subplot(grid[4, figIndex_inner])
                else:
                    ax0 = fig.add_subplot(grid[:2, figIndex_inner])
                    ax1 = fig.add_subplot(grid[2:4, figIndex_inner])
                    axL = fig.add_subplot(grid[4:, figIndex_inner])

                GP.plot(
                    axs=[ax0, ax1, axL],
                    x_next=x_next,
                    x_best=x_best,
                    y_best=y_best1,
                    yVar_best=yVar_best1,
                    plotFundamental=plotFundamental,
                    stds=stds,
                    pointsEvaluate=pointsEvaluateEachGPdimension,
                )

            # Plot model specifics from last model
            self.plotModelStatus(boStep=k, fn=fn, stds=stds, tab_color=tab_color)

        print("- Finished plotting of step models")

        """
		****************************************************************
		Optimization Stuff
		****************************************************************
		"""

        tab_color = ck + 5 + 1

        # ---- Trust region ----------------------------------------------------------
        figTR = fn.add_figure(label="Trust Region", tab_color=tab_color)
        try:
            SBOcorrections.plotTrustRegionInformation(self, fig=figTR)
        except:
            print("\t- Problem plotting trust region", typeMsg="w")

        # ---- optimization_results ---------------------------------------------------
        if plotoptimization_results:
            # Most current state of the optimization_results.out
            self.optimization_results.read()
            self.optimization_results.plot(
                fn=fn, doNotShow=True, log=self.timings_file, tab_color=tab_color
            )

        """
		****************************************************************
		Acquisition
		****************************************************************
		"""
        try:    
            self.plotAcquisitionOptimizationSummary(fn=fn)
        except: 
            print('\t- Problem plotting acquisition optimization summary', typeMsg='w')

        return fn


    def plotAcquisitionOptimizationSummary(self, fn=None, step_from=0, step_to=-1):

        if step_to == -1:
            step_to = len(self.steps)

        step_to = np.min([step_to, len(self.steps)])

        step_num = np.arange(step_from, step_to)

        fig = fn.add_figure(label='Acquisition Convergence')

        axs = GRAPHICStools.producePlotsGrid(len(step_num), fig=fig, hspace=0.6, wspace=0.3)
        colors = GRAPHICStools.listColors()

        for step in step_num:

            ax = axs[step]

            if 'InfoOptimization' not in self.steps[step].__dict__: break

            # Grab info from optimization
            infoOPT = self.steps[step].InfoOptimization
            acq = self.steps[step].evaluators['acq_function']

            acq_trained = np.zeros(self.steps[step].train_X.shape[0])
            for ix in range(self.steps[step].train_X.shape[0]):
                acq_trained[ix] = acq(torch.Tensor(self.steps[step].train_X[ix,:]).unsqueeze(0)).item()

            # Plot trained acquisition
            ax.axhline(y=acq_trained.max(), c='k', ls='--', lw=1.0, label='max of trained')

            # Plot acquisition evolution 
            for i in range(len(infoOPT)-1): #no cleanup stage
                y_acq = infoOPT[i]['info']['acq_evaluated'].cpu().numpy()
                
                if len(y_acq.shape)>1:
                    for j in range(y_acq.shape[1]):
                        ax.plot(y_acq[:,j],'-o', c=colors[i], markersize=0.5, lw = 0.3, label=f'{infoOPT[i]["method"]} (candidate #{j})')
                else:
                    ax.plot(y_acq,'-o', c=colors[i], markersize=1, lw = 0.5, label=f'{infoOPT[i]["method"]}')
                
                # Plot max of guesses
                if len(y_acq)>0:
                    ax.axhline(y=y_acq.max(axis=1)[0], c=colors[i], ls='--', lw=1.0, label=f'{infoOPT[i]["method"]} (max of guesses)')

            ax.set_title(f'BO Step #{step}')
            ax.set_ylabel('$f_{acq}$ (to max)')
            ax.set_xlabel('Evaluations')
            if step == step_num[0]:
                ax.legend(loc='best', fontsize=6)

            GRAPHICStools.addDenseAxis(ax)


    def plotModelStatus(
        self, fn=None, boStep=-1, plotsPerFigure=20, stds=2, tab_color=None
    ):
        step = self.steps[boStep]

        GP = step.GP["combined_model"]

        # ---- Jacobian -------------------------------------------------------
        fig = fn.add_figure(label=f"#{boStep}: Jacobian", tab_color=tab_color)
        maxPoints = 1  # 4
        xExplore = []
        if "x_next" in step.__dict__.keys() and step.x_next is not None:
            for i in range(np.min([step.x_next.shape[0], maxPoints])):
                xExplore.append(step.x_next[i].cpu().numpy())
        else:
            xExplore.append(step.train_X[0])

        axs = GRAPHICStools.producePlotsGrid(
            len(xExplore), fig=fig, hspace=0.3, wspace=0.3
        )

        for i in range(len(xExplore)):
            GP.localBehavior(
                torch.from_numpy(xExplore[i]),
                prefix=f"Next #{i+1}\n",
                outputs=self.outputs,
                ax=axs[i],
            )

        # ---- Training quality -------------------------------------------------------
        x_next = step.x_next if "x_next" in step.__dict__.keys() else None
        y_next, ystd_next = (
            (step.y_next, step.ystd_next)
            if "y_next" in step.__dict__.keys()
            else (None, None)
        )

        numGPs = GP.train_Y.shape[1]
        numfigs = int(np.ceil(numGPs / plotsPerFigure))

        figsQuality = []
        for i in range(numfigs):
            figsQuality.append(
                fn.add_figure(
                    label=f"#{boStep}: Quality {i+1}/{numfigs}", tab_color=tab_color
                )
            )

        axs = GP.testTraining(
            plotYN=True,
            figs=figsQuality,
            x_next=x_next,
            y_next=y_next,
            ystd_next=ystd_next,
            plotsPerFigure=plotsPerFigure,
            ylabels=self.outputs,
            stds=stds,
        )

        if axs is not None:
            for i in range(len(self.outputs)):
                # axs[i].set_title(self.outputs[i])
                GRAPHICStools.addDenseAxis(axs[i])

        # ---- Optimization Performance ---------------------------------------
        if "InfoOptimization" not in step.__dict__.keys():
            return

        figOPT1 = fn.add_figure(label=f"#{boStep}: Optim. Perfom.", tab_color=tab_color)
        figOPT2 = fn.add_figure(label=f"#{boStep}: Optim. Ranges", tab_color=tab_color)
        self.plotSurrogateOptimization(fig1=figOPT1, fig2=figOPT2, boStep=boStep)
        # ---------------------------------------------------------------------

    def plotSurrogateOptimization(self, fig1=None, fig2=None, boStep=-1):
        # ----------------------------------------------------------------------
        # Select information
        # ----------------------------------------------------------------------

        step = self.steps[boStep]
        info, boundsRaw = step.InfoOptimization, step.bounds

        bounds = torch.Tensor([boundsRaw[b] for b in boundsRaw])
        boundsThis = info[0]["bounds"].cpu().numpy().transpose(1, 0) if "bounds" in info[0] else None

        # ----------------------------------------------------------------------
        # Prep figures
        # ----------------------------------------------------------------------

        colors = GRAPHICStools.listColors()

        if fig1 is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            fn = FigureNotebook("PRF BO Strategy", geometry="1700x1000")
            fig2 = fn.add_figure(label=f"#{boStep}: optimization_options Ranges")
            fig1 = fn.add_figure(label=f"#{boStep}: optimization_options Perfom.")

        grid = plt.GridSpec(nrows=2, ncols=2, hspace=0.2, wspace=0.2)
        ax0_r = fig2.add_subplot(grid[:, 0])
        ax1_r = fig2.add_subplot(grid[0, 1])
        ax2_r = fig2.add_subplot(grid[1, 1])

        # Get dimensions and prepare figures
        num_x = step.InfoOptimization[0]["info"]["x_start"].shape[1]
        num_y = step.InfoOptimization[0]["info"]["yFun_start"].shape[1]

        num_axes_x, num_axes_y, num_axes_res = (
            int(np.ceil(num_x / 2)),
            int(np.ceil(num_y / 2)),
            1,
        )

        if num_axes_x == 0:
            num_axes_x += 1
        if num_axes_y == 0:
            num_axes_y += 1

        num_plots = num_axes_x + num_axes_y + num_axes_res
        axs = GRAPHICStools.producePlotsGrid(num_plots, fig=fig1, hspace=0.4, wspace=0.4)

        axsDVs = axs[:num_axes_x]
        axsOFs = axs[num_axes_x:-1]
        axR = [axs[-1]]

        axislabels = [i for i in boundsRaw]

        # ----------------------------------------------------------------------
        # Plot DVs and OFs - Training
        # ----------------------------------------------------------------------

        it_start = 0

        iinfo = info[0]["info"]
        it_start, xypair = OPTtools.plotInfo(
            iinfo,
            label="Training",
            plotStart=True,
            xypair=[],
            axTraj=ax0_r,
            axDVs_r=ax1_r,
            axOFs_r=ax2_r,
            axDVs=axsDVs,
            axOFs=axsOFs,
            axR=axR,
            bounds=bounds,
            boundsThis=boundsThis,
            axislabels_x=axislabels,
            axislabels_y=self.optimization_object.name_objectives,
            color="k",
            ms=10,
            alpha=0.5,
            it_start=it_start,
        )

        # Loop over posterior steps
        for ipost in range(len(info) - 1):
            iinfo = info[ipost]["info"]
            try:
                it_start, xypair = OPTtools.plotInfo(
                    iinfo,
                    label=info[ipost]["method"],
                    plotStart=False,
                    xypair=xypair,
                    axTraj=ax0_r,
                    axDVs_r=ax1_r,
                    axOFs_r=ax2_r,
                    axDVs=axsDVs,
                    axOFs=axsOFs,
                    axR=axR,
                    axislabels_x=axislabels,
                    axislabels_y=self.optimization_object.name_objectives,
                    color=colors[ipost],
                    ms=8 - ipost * 1.5,
                    alpha=0.5,
                    it_start=it_start,
                )
            except KeyError as e:
                print(f"\t- Problem plotting {info[ipost]['method']}: ",e, typeMsg="w")

        xypair = np.array(xypair)

        axsDVs[0].legend(prop={"size": 5})
        ax1_r.set_ylabel("DV values")
        GRAPHICStools.addDenseAxis(ax1_r)
        GRAPHICStools.autoscale_y(ax1_r)

        ax2_r.set_ylabel("Acquisition values")
        GRAPHICStools.addDenseAxis(ax2_r)
        GRAPHICStools.autoscale_y(ax2_r)

        ax0_r.plot(xypair[:, 0], xypair[:, 1], "-s", markersize=5, lw=2.0, c="k")

        iinfo = info[-1]["info"]
        for i, y in enumerate(iinfo["y_res"]):
            ax0_r.axhline(
                y=y,
                c=colors[ipost + 1],
                ls="--",
                lw=2,
                label=info[-1]["method"] if i == 0 else "",
            )
        iinfo = info[0]["info"]
        ax0_r.axhline(y=iinfo["y_res_start"][0], c="k", ls="--", lw=2)

        ax0_r.set_xlabel("Optimization iterations")
        ax0_r.set_ylabel("$f_{acq}$")
        GRAPHICStools.addDenseAxis(ax0_r)
        ax0_r.legend(loc="best", prop={"size": 8})
        ax0_r.set_title("Evolution of acquisition in optimization stages")

        for i in range(len(axs)):
            GRAPHICStools.addDenseAxis(axs[i])


# ===========================================================================
# Surrogate-assisted Weighted Levenberg-Marquardt solver
# ===========================================================================

class MITIM_wLM(MITIM_BO):
    """
    Surrogate-Assisted Weighted Levenberg-Marquardt (wLM) transport solver.

    Replaces MITIM_Anderson: Anderson mixing does not accelerate near the
    inflection point of the stiff transport submodel, where the fixed-point
    map x -> g(x) is locally divergent. Weighted LM with uncertainty-adaptive
    damping handles this naturally -- large lambda (driven by high GP
    uncertainty / small residuals) gives damped, relaxation-like steps; small
    lambda gives Gauss-Newton convergence once the surrogate is trustworthy.
    See surrogate_assisted_solver.md for the full design rationale.

    Outer loop (identical structure to MITIM_BO.run -- fully inherited):
        1. initializeOptimization() seeds the GPs with initial_training
           real-model evaluations (this also plays the role of the design
           doc's Phase 1 "basin entry": the doc's own conclusion is that
           Phase 2's adaptive damping subsumes relaxation-like behaviour, so
           no separate relaxation driver is added here).
        2. Per outer iteration:
           a. _step()      -- fit GPs, run weighted LM on surrogate -> x_next
           b. updateSet()  -- evaluate the expensive code at x_next
           c. check stopping criterion
        3. Repeat until convergence or maximum_iterations.

    Inner solve (Phase 2 of the design doc), per surrogate iteration:
        1. mu_F, Sigma_F, J <- _residual_and_jacobian(x)
           (mu_F = cal - of, i.e. the same residual "source" MITIM_Anderson
           extracted; Sigma_F is exactly diagonal because `combined_model` is
           a ModifiedModelListGP of independent single-output GPs -- no
           cross-residual covariance terms; J is the analytic Jacobian
           d(mu_F)/d(x_int) via mitim_jacobian, the in-house vmap+autograd
           implementation that already differentiates through this exact
           GP-posterior path in production, see scipy_root/solver='lm')
        2. Solve the weighted normal equations, refined by a box- and
           soft-monotonicity-constrained QP (see _solve_weighted_lm_step)
        3. Update x <- x + delta, clipped to bounds and trust regions
        4. Adapt lambda ~ tr(Sigma_F)/||mu_F||^2 and the monotonicity
           penalty rho
        5. Check convergence via the average weighted-gradient magnitude
           |J^T Sigma_F^-1 mu_F|

    Phase 3 (x_lcfs feasibility) runs once at convergence: see
    _assess_lcfs_feasibility.
    """

    _wlm_defaults = {
        # --- weighted-LM inner loop -------------------------------------
        "max_inner_iter":     20,     # max LM iterations per outer step
        "lambda_init":        1.0,    # initial damping lambda
        "lambda_min":         1e-6,   # Gauss-Newton floor
        "lambda_max":         1e3,    # damping ceiling (also scales rho)
        "lambda_scale":       1.0,    # proportionality constant in lambda ~ tr(Sigma_F)/||mu_F||^2
        "grad_tol":           1e-4,   # convergence: average |J^T W mu_F|
        "stds":               2,      # GP CI half-width in units of sigma (mirrors _anderson_defaults)
        "sigma_rel_tol":      1.0,    # stop if ||sigma_y||/||y|| exceeds this (mirrors _anderson_defaults)
        "max_rel_step":       0.1,    # per-step trust region: max move as fraction of DV-bounds width
        "max_total_rel_step": 0.2,    # outer trust region: max cumulative move from x_start
        "nn_dist_tol":        2.0,    # stop if nearest training point exceeds this many DV-std deviations

        # --- constrained QP step (box bounds + soft monotonicity) -------
        "rho_base":         1.0,   # rho_base in rho_i = rho_base * (x_i/x_ref) * (lambda_max/lambda)
        "x_ref":            1.0,   # reference DV magnitude for the penalty scaling (problem-dependent)
        "active_dual_tol":  1e-8,  # |dual| above which a monotonicity row is reported "active"

        # --- Phase 3: x_lcfs feasibility (switchable; no-op unless Sigma_lcfs is set) ---
        "lcfs_feasibility_enabled": True,
        "Sigma_lcfs":          {},     # {channel: sigma}, channel = plain physical name ("te"/"ti"/"ne") regardless of which bc_dict family is perturbed (see y_or_aly). sigma is normally a scalar (applied to whichever single family y_or_aly selects); in "both" mode it may instead be a 2-sequence/dict (sigma_y, sigma_aly) / {"y":.., "aly":..} giving independent per-family values -- needed because the "y" and "aly" bc_dict families generally live on different scales (e.g. keV vs a dimensionless gradient-scale-length ratio), so one absolute number cannot mean both at once. A bare scalar in "both" mode is applied identically to each family, which is only dimensionally sound when sigma_is_rel=True (a relative fraction is scale-free).
        "sigma_is_rel":        False,  # if True, each resolved sigma is a *fraction* of that family's own nominal BC value: sigma_abs = sigma_rel * |bc_dict[key][0]| -- this is what makes a single scalar Sigma_lcfs[channel] dimensionally consistent across both families in "both" mode
        "y_or_aly":            "y",    # which bc_dict family to perturb per channel: "y" -> plain value keys ("te"/"ti"/"ne"), "aly" -> gradient keys ("aLte"/"aLti"/"aLne"), "both" -> probe both families independently per channel (2x the candidates: each gets its own bracket and re-optimization)
        "lcfs_n_sigma":        2.0,    # truncation half-width (in units of sigma) for the Phase-3 joint MC draws: each x_lcfs is drawn from a truncated normal N(nominal, sigma_abs^2) clipped to nominal +/- n_sigma*sigma_abs (keeps draws inside the modeled prior and limits GP extrapolation through the bc->feature path)
        "lcfs_n_samples":      64,    # number of joint Monte Carlo x_lcfs draws in Phase 3 (all channels/families perturbed together per draw, then x_int re-optimized on the GP). Surrogate-only, so this can be large; only the best draw is further refined.
        "lcfs_reopt_max_iter": 5,      # cap on re-optimization iterations per MC draw (warm-started from x_int_star -- small perturbation, should converge fast)
        "lcfs_refine_n_evals": 0,      # number of real-model evaluations with inner wLM refinement at the best-found x_lcfs. If 0, only perform one confirmation check; if > 0, iteratively refine x_int at the best x_lcfs with this many real evals (like phase 2 but anchored to the converged LCFS BC).
        "lcfs_reopt_refine_max_iter": 3,  # cap on inner wLM iterations per refinement evaluation (refinement warm-starts are tighter than MC exploration, so fewer iterations suffice)
        "lcfs_n_jitter_evals": 2,      # real-model evaluations at x_int_star with x_lcfs drawn uniformly from the full bracket, run before the surrogate Phase 3 re-optimizations; enriches the training set at convergence (where the GP is already well-trained on x_int), rather than at initialization where x_lcfs variation looks like noise
    }

    def __init__(self, optimization_object, wlm_options=None, **kwargs):
        super().__init__(optimization_object, **kwargs)

        self.wlm_options = dict(self._wlm_defaults)
        if wlm_options:
            self.wlm_options.update(wlm_options)

        # wLM always proposes exactly one point per outer step (mirrors MITIM_Anderson)
        if self.best_points != 1:
            print(
                f"\t* MITIM_wLM: overriding best_points "
                f"({self.best_points} -> 1) -- weighted LM proposes one point per step",
                typeMsg="w",
            )
            self.best_points = 1
            self.best_points_sequence = [1]
            self.stepSettings["best_points_sequence"] = self.best_points_sequence

        # Per-channel monotonicity pairs derived from the DV naming convention
        # f"aL{channel}_{position}" (see PORTALSinit) -- grouped per channel so
        # that mixed-channel DV vectors are not chained into one global order.
        self._mono_pairs = self._build_monotonicity_pairs()

        self._qp_warm_start = None
        self._active_mono_duals = {}
        self.lcfs_feasibility_report = None

    # ------------------------------------------------------------------
    # run(): fully inherited outer loop, plus a single Phase-3 assessment
    # at the converged point (not every outer step -- each candidate now
    # requires a full capped, warm-started re-optimization, so probing it
    # at every iteration would multiply the cost of the whole run).
    # ------------------------------------------------------------------

    def run(self):
        super().run()

        if not self.steps or "combined_model" not in self.steps[-1].GP:
            return

        combined_gp = self.steps[-1].GP["combined_model"]
        ind_best = self.BOmetrics["overall"]["indBest"]
        x_star = torch.tensor(self.train_X[ind_best]).to(self.dfT)

        # Optional pre-step: real evaluations at varied x_lcfs + GP retraining.
        # Runs before Phase 3 so _assess_lcfs_feasibility uses a GP that has
        # seen the x_lcfs bracket (at convergence, where the GP is already well
        # trained on x_int; doing this at initialization degrades early BO).
        n_jitter = max(0, int(self.wlm_options.get("lcfs_n_jitter_evals", 0)))
        if (n_jitter > 0
                and self.wlm_options["lcfs_feasibility_enabled"]
                and self.wlm_options["Sigma_lcfs"]):
            combined_gp = self._run_lcfs_jitter_and_retrain(x_star, n_jitter)

        self.lcfs_feasibility_report = self._assess_lcfs_feasibility(x_star, combined_gp)

    # ------------------------------------------------------------------
    # Monotonicity grouping: aL{channel}_{position}, ordered per channel
    # ------------------------------------------------------------------

    def _build_monotonicity_pairs(self):
        """
        Build consecutive-index pairs (i_lo, i_hi) within each physical
        channel (ordered by radial position) for the soft-monotonicity
        constraint  x_int[i_hi] - x_int[i_lo] + s >= 0.

        Convention assumption: monotonicity is enforced per profile channel
        across its own radial sequence (e.g. aLte_1 < aLte_2 < ...), not as
        one global chain across mixed-channel DOFs -- derived directly from
        self.bounds (DV order) rather than assumed.
        """
        dv_names = list(self.bounds.keys())
        pattern = re.compile(r"^aL(?P<channel>.+)_(?P<position>\d+)$")

        channel_groups = {}
        for idx, name in enumerate(dv_names):
            m = pattern.match(str(name))
            if not m:
                continue
            channel_groups.setdefault(m.group("channel"), []).append(
                (int(m.group("position")), idx)
            )

        pairs = []
        for channel, entries in channel_groups.items():
            entries.sort(key=lambda e: e[0])
            for k in range(len(entries) - 1):
                pairs.append((entries[k][1], entries[k + 1][1]))

        return pairs

    # ------------------------------------------------------------------
    # mu_F, Sigma_F (diagonal), J = d(mu_F)/d(x_int)
    # ------------------------------------------------------------------

    def _residual_and_jacobian(self, x, combined_gp):
        """
        Physics: mu_F = cal - of = (Q_turb + Q_neo) - Q_target, the same
        residual vector MITIM_Anderson._compute_am_residual calls `source`.

        `combined_gp` is a ModifiedModelListGP of independent single-output
        GPs -- but one per *raw surrogate output* (the n_outputs entries of
        self.outputs / problem_options['ofs'], e.g. Qe_tr_turb_1, Qe_tr_neoc_1,
        Qe_tar_1, ... across channels and radii), not one per final residual
        element. mu_F itself (n_residuals = predicted channels x radii) is an
        *analytic affine combination* of those raw outputs -- scalarized_
        objective/calculate_residuals builds e.g. of = Qx_tr_turb + Qx_tr_neoc
        and cal = Qx_tar (+/- the linear turbulent-exchange integral), so
        n_residuals < n_outputs in general (e.g. 12 vs 40: this mismatch is
        exactly what raised "tensor a (n_outputs) must match tensor b
        (n_residuals)" when Sigma_F was read off the raw GP posterior
        directly -- that lives in the wrong, n_outputs-dimensional space).

        Sigma_F has to instead be propagated through the same affine map
        G = d(mu_F)/d(y_raw) that produces mu_F from the raw posterior mean.
        Because that map only sums/differences (scalar-weighted) independent
        raw-output entries, Var(mu_F) = G Sigma_y G^T; we keep the
        diagonal-only approximation already baked into the weighted-LM
        design (so no dense Sigma_F inversion is needed), i.e.
        Sigma_F_diag[i] = sum_j G[i,j]^2 Sigma_y_diag[j]. This discards the
        (typically small) cross-residual covariance induced by raw outputs
        shared between residual elements -- e.g. the turbulent-exchange term
        entering both the Qe and Qi residuals with opposite sign.

        J is the analytic Jacobian d(mu_F)/d(x_int) computed in one
        vmap+autograd pass via multivariate_tools.mitim_jacobian -- the
        in-house equivalent of the design doc's torch.func.jacrev (chosen
        deliberately over jacrev: it is already proven to differentiate
        through this exact GP-posterior path in production, backing
        scipy_root's solver='lm' acquisition path, and avoids the known
        fragility of torch.func transforms over GPyTorch lazy/Cholesky
        internals -- net effect on the math is identical). G is obtained the
        same way, applied to the of/cal map alone with the raw posterior mean
        as input -- since that map is affine (no curvature), evaluating its
        Jacobian at the current mean is exact, not a local linearisation.
        """
        stds = self.wlm_options["stds"]

        def residual_fn(xi):
            y_mean, _, _, _ = combined_gp.predict(xi.unsqueeze(0))
            of, cal, _ = self.scalarized_objective(y_mean)
            return (cal - of).squeeze(0)

        x_in = x.detach().clone()
        mu_F, J = multivariate_tools.mitim_jacobian(residual_fn, x_in, vectorize=True)

        with torch.no_grad():
            y_mean, y_upper, y_lower, _ = combined_gp.predict(x_in.unsqueeze(0))
            sigma_y = (y_upper - y_lower).squeeze(0).abs() / (2.0 * stds)
            Sigma_y_diag = (sigma_y ** 2).clamp(min=1e-12)

        def of_minus_cal(y_raw):
            of, cal, _ = self.scalarized_objective(y_raw.unsqueeze(0))
            return (cal - of).squeeze(0)

        y_raw_in = y_mean.squeeze(0).detach().clone()
        _, G = multivariate_tools.mitim_jacobian(of_minus_cal, y_raw_in, vectorize=True)

        Sigma_F_diag = ((G.detach() ** 2) @ Sigma_y_diag).clamp(min=1e-12)

        return mu_F.detach(), Sigma_F_diag, J.detach()

    # ------------------------------------------------------------------
    # Constrained QP step: box bounds + soft monotonicity (active-set, OSQP)
    # ------------------------------------------------------------------

    def _solve_weighted_lm_step(self, x, mu_F, Sigma_F_diag, J, lam, lb_np, ub_np):
        """
        Solve the weighted-LM step delta via the constrained QP from the
        design doc's "Constraints -- Active Set QP" section:

            min_{delta,s}  ||J delta + mu_F||^2_{Sigma_F^-1}
                           + lambda ||delta||^2 + rho^T s

            s.t.  lb <= x + delta <= ub                                  (box, hard)
                  x[i_hi]-x[i_lo] + delta[i_hi]-delta[i_lo] + s_k >= 0   (soft monotonicity, per channel)
                  s_k >= 0

        Sigma_F is diagonal (see _residual_and_jacobian), so
        Sigma_F^-1 = diag(1/Sigma_F_diag) and the normal-equation matrix
        A = J^T W J + lambda I is dense but tiny (n_DV x n_DV, ~10x10).
        Expanding the quadratic form gives the OSQP standard form
        min 0.5 z^T P z + q^T z with z = [delta; s]:
            P = blockdiag(2A, 0),   q = [2 J^T W mu_F ; rho]

        Note on "active set": OSQP is an ADMM/interior-point solver that
        solves the full inequality-constrained QP directly in a single pass
        -- it does not need (and would gain nothing from) the explicit
        "activate / release on negative multiplier / re-solve" loop that
        classical active-set QP methods require; that bookkeeping is an
        artifact of the *solution method*, not the problem statement, and
        re-implementing it on top of an ADMM solver would be redundant
        scope. What the design doc actually wants out of that machinery --
        "Lagrange multipliers on active monotonicity constraints" as a
        diagnostic of physically binding profile-shape constraints -- is
        exactly OSQP's dual solution `res.y` on the monotonicity rows,
        which is reported via self._active_mono_duals. Warm-starting is
        carried across LM iterations via self._qp_warm_start, matching the
        doc's "carry the active set from the previous iteration" intent.
        """
        import scipy.sparse as sp
        import osqp

        n = x.shape[0]
        pairs = self._mono_pairs
        m = len(pairs)

        W_diag = (1.0 / Sigma_F_diag).cpu().numpy()
        J_np = J.cpu().numpy()
        mu_np = mu_F.cpu().numpy()
        x_np = x.cpu().numpy()

        JTW = J_np.T * W_diag[None, :]              # (n_DV, n_res)
        A_mat = JTW @ J_np + lam * np.eye(n)         # J^T W J + lambda I
        b_vec = JTW @ mu_np                           # J^T W mu_F

        rho_base = self.wlm_options["rho_base"]
        x_ref = self.wlm_options["x_ref"]
        lambda_max = self.wlm_options["lambda_max"]

        if m > 0:
            P = sp.block_diag([2.0 * A_mat, np.zeros((m, m))], format="csc")
            rho = np.array([
                rho_base * (x_np[i_lo] / x_ref) * (lambda_max / max(lam, 1e-300))
                for (i_lo, _i_hi) in pairs
            ])
            q = np.concatenate([2.0 * b_vec, rho])
        else:
            P = sp.csc_matrix(2.0 * A_mat)
            q = 2.0 * b_vec

        rows, l_list, u_list = [], [], []

        # Box bounds (hard): lb - x <= delta <= ub - x
        rows.append(sp.hstack([sp.eye(n), sp.csc_matrix((n, m))]) if m > 0 else sp.eye(n))
        l_list.append(lb_np - x_np)
        u_list.append(ub_np - x_np)

        if m > 0:
            # Soft monotonicity: (x[i_hi]-x[i_lo]) + (delta[i_hi]-delta[i_lo]) + s_k >= 0
            A_mono = sp.lil_matrix((m, n + m))
            l_mono = np.empty(m)
            for k, (i_lo, i_hi) in enumerate(pairs):
                A_mono[k, i_hi] = 1.0
                A_mono[k, i_lo] = -1.0
                A_mono[k, n + k] = 1.0
                l_mono[k] = -(x_np[i_hi] - x_np[i_lo])
            rows.append(A_mono.tocsc())
            l_list.append(l_mono)
            u_list.append(np.full(m, np.inf))

            # Slack non-negativity: s_k >= 0
            rows.append(sp.hstack([sp.csc_matrix((m, n)), sp.eye(m)]))
            l_list.append(np.zeros(m))
            u_list.append(np.full(m, np.inf))

        A_constr = sp.vstack(rows, format="csc")
        l_vec = np.concatenate(l_list)
        u_vec = np.concatenate(u_list)

        prob = osqp.OSQP()
        prob.setup(P, q, A_constr, l_vec, u_vec, verbose=False, warm_start=True, polish=True)

        if (self._qp_warm_start is not None) and (self._qp_warm_start[0].shape[0] == n + m):
            prob.warm_start(x=self._qp_warm_start[0], y=self._qp_warm_start[1])

        res = prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(
                f"\t\t* QP step solve failed (status: '{res.info.status}'); "
                f"falling back to the damped Gauss-Newton step (no constraint refinement)",
                typeMsg="w",
            )
            delta_np = np.linalg.solve(A_mat + 1e-8 * np.eye(n), -b_vec)
            self._qp_warm_start = None
        else:
            delta_np = res.x[:n]
            self._qp_warm_start = (res.x, res.y)
            if m > 0:
                self._active_mono_duals = {
                    pairs[k]: float(res.y[n + k])
                    for k in range(m)
                    if abs(res.y[n + k]) > self.wlm_options["active_dual_tol"]
                }

        return torch.tensor(delta_np, dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # Shared context (bounds / training-data tensors) and the inner
    # weighted-LM solve loop -- factored out of _step so that Phase 3
    # can re-run the *exact same* solver (warm-started from x_int_star)
    # to find the optimal x_int at a perturbed x_lcfs (see
    # _assess_lcfs_feasibility): "would re-optimizing x_int for a nearby
    # x_lcfs converge to something better?" requires an inner solve, not
    # a fixed-point evaluation.
    # ------------------------------------------------------------------

    def _lm_context(self):
        """Bounds/training-data tensors shared by every _run_inner_lm call
        (both the main _step solve and Phase 3's re-optimization sub-solves)."""
        bounds_arr = np.array(list(self.bounds.values()))   # (n_DV, 2)
        lb = torch.tensor(bounds_arr[:, 0]).to(self.dfT)
        ub = torch.tensor(bounds_arr[:, 1]).to(self.dfT)
        bounds_width = (ub - lb).clamp(min=1e-10)

        # Nearest-neighbour distance in standardised DV space (same GP-support
        # proxy as MITIM_Anderson._step -- catches surrogate extrapolation that
        # posterior variance alone may not flag).
        train_X_t = torch.tensor(self.train_X, dtype=self.dfT.dtype, device=self.dfT.device)
        train_X_std = train_X_t.std(dim=0).clamp(min=1e-10)

        return {
            "lb": lb, "ub": ub,
            "lb_np": bounds_arr[:, 0], "ub_np": bounds_arr[:, 1],
            "bounds_width": bounds_width,
            "train_X_t": train_X_t, "train_X_std": train_X_std,
        }

    def _run_inner_lm(self, x_init, combined_gp, ctx, max_iter=None, label="wLM", verbose=True):
        """
        Run the weighted-LM inner loop (residual/Jacobian -> constrained QP
        step -> adaptive damping, see _residual_and_jacobian /
        _solve_weighted_lm_step) to convergence on the already-fitted
        `combined_gp`, starting from x_init. x_init also anchors the outer
        trust region (mirrors _step: the loop never wanders far from where
        it started).

        Two callers:
          - _step: starts from the best evaluated point, full max_inner_iter
            budget -- this *is* Phase 2 of the design doc.
          - _assess_lcfs_feasibility (Phase 3): starts from x_int_star,
            warm-started, with a small max_iter cap -- finds x_int*(x_lcfs)
            for a perturbed boundary condition. Since the perturbation is
            small and x_int_star is already near-optimal for the nominal
            x_lcfs, this should converge in just a few iterations.

        Returns (x_converged, info) with
            info = {"n_iter", "stop_reason", "grad_mag"}
        """
        lb, ub = ctx["lb"], ctx["ub"]
        lb_np, ub_np = ctx["lb_np"], ctx["ub_np"]
        bounds_width = ctx["bounds_width"]
        train_X_t, train_X_std = ctx["train_X_t"], ctx["train_X_std"]

        x_k = x_init.detach().clone().to(self.dfT)
        x_start = x_k.clone()

        lam = float(self.wlm_options["lambda_init"])
        lambda_min = self.wlm_options["lambda_min"]
        lambda_max = self.wlm_options["lambda_max"]
        lambda_scale = self.wlm_options["lambda_scale"]
        grad_tol = self.wlm_options["grad_tol"]
        total_limit = self.wlm_options["max_total_rel_step"]
        step_limit = self.wlm_options["max_rel_step"]
        max_iter = self.wlm_options["max_inner_iter"] if max_iter is None else max_iter

        self._qp_warm_start = None
        self._active_mono_duals = {}

        stop_reason, grad_mag, inner_it = "max_iter", float("nan"), -1

        if verbose:
            print(
                f"\n\t--- {label} inner loop  (max_iter={max_iter},  "
                f"lambda0={lam:.2e},  mono_pairs={len(self._mono_pairs)}) ---"
            )

        for inner_it in range(max_iter):

            mu_F, Sigma_F_diag, J = self._residual_and_jacobian(x_k, combined_gp)
            W_diag = 1.0 / Sigma_F_diag

            # J^T Sigma_F^-1 mu_F: the weighted gradient -- this is exactly the
            # RHS of the normal equations (zero at a stationary point), and
            # therefore the natural "average gradient magnitude of x_int
            # elements" the design doc names as the primary inner-convergence
            # criterion.
            grad = J.T @ (W_diag * mu_F)
            grad_mag = grad.abs().mean().item()

            # GP output uncertainty relative to prediction magnitude (same
            # diagnostic MITIM_Anderson._step uses to detect surrogate
            # unreliability).
            with torch.no_grad():
                y_mean_k, y_upper_k, y_lower_k, _ = combined_gp.predict(x_k.unsqueeze(0))
            sigma_y_k = (y_upper_k - y_lower_k).squeeze(0).abs() / (2.0 * self.wlm_options["stds"])
            y_norm = y_mean_k.squeeze(0).norm().item() + 1e-12
            sigma_rel = sigma_y_k.norm().item() / y_norm

            total_rel = ((x_k - x_start).abs() / bounds_width).max().item()
            nn_dist = ((x_k.unsqueeze(0) - train_X_t) / train_X_std).norm(dim=1).min().item()

            if verbose:
                print(
                    f"\t\t[{label} {inner_it:3d}]  lambda={lam:.3e}  "
                    f"|J^T W mu_F|={grad_mag:.3e}  sigma/|y|={sigma_rel:.3e}  "
                    f"disp={total_rel:.2f}w  nn={nn_dist:.2f}sigma"
                )

            # Convergence: weighted-gradient magnitude negligible (stationary
            # point of the weighted normal equations on the surrogate)
            if grad_mag < grad_tol:
                stop_reason = "converged"
                if verbose:
                    print(f"\t\t-> {label} converged (gradient magnitude threshold)", typeMsg="i")
                break

            # Stop: GP output uncertainty too high to trust the surrogate
            if sigma_rel > self.wlm_options["sigma_rel_tol"]:
                stop_reason = "surrogate_uncertainty"
                if verbose:
                    print(
                        f"\t\t-> {label} stopped: GP output uncertainty too high "
                        f"({sigma_rel:.3e} > {self.wlm_options['sigma_rel_tol']})",
                        typeMsg="w",
                    )
                break

            # Stop: too far from any training point -- GP is extrapolating
            if nn_dist > self.wlm_options["nn_dist_tol"]:
                stop_reason = "outside_support"
                if verbose:
                    print(
                        f"\t\t-> {label} stopped: outside GP support "
                        f"(nn_dist={nn_dist:.2f}sigma > {self.wlm_options['nn_dist_tol']}sigma)",
                        typeMsg="w",
                    )
                break

            # Stop: outer trust-region budget exhausted
            if total_rel >= total_limit - 1e-6:
                stop_reason = "trust_region"
                if verbose:
                    print(
                        f"\t\t-> {label} stopped: cumulative displacement {total_rel:.2f}w "
                        f"reached outer limit {total_limit}w",
                        typeMsg="i",
                    )
                break

            # ---- Constrained weighted-LM step (normal equations + QP refine)
            delta = self._solve_weighted_lm_step(x_k, mu_F, Sigma_F_diag, J, lam, lb_np, ub_np)

            # Per-step trust region (mirrors MITIM_Anderson._step): no single
            # DV moves more than step_limit x bounds width
            per_step_rel = (delta.abs() / bounds_width).max()
            if per_step_rel > step_limit:
                delta = delta * (step_limit / per_step_rel)

            x_candidate = x_k + delta

            # Outer trust region: cumulative drift from x_start <= total_limit
            dx_total = x_candidate - x_start
            outer_rel = (dx_total.abs() / bounds_width).max()
            if outer_rel > total_limit:
                x_candidate = x_start + dx_total * (total_limit / outer_rel)

            x_k = torch.clamp(x_candidate, lb, ub).to(self.dfT)

            # ---- Adaptive damping: lambda ~ tr(Sigma_F)/||mu_F||^2 ----------
            # High GP uncertainty (large tr(Sigma_F)) or near-stationary
            # residuals (small ||mu_F||) push lambda up -> damped,
            # relaxation-like steps; growing GP confidence and larger
            # residuals push lambda down -> Gauss-Newton convergence. This is
            # precisely what makes Phase 2 "subsume relaxation-like behaviour
            # automatically" per the design doc's summary of decisions, with
            # no separate relaxation phase required.
            mu_F_normsq = (mu_F @ mu_F).item()
            lam = lambda_scale * Sigma_F_diag.sum().item() / max(mu_F_normsq, 1e-300)
            lam = float(np.clip(lam, lambda_min, lambda_max))

        else:
            stop_reason = "max_iter"
            if verbose:
                print(f"\t\t-> {label} stopped: reached max_iter={max_iter}", typeMsg="w")

        if verbose and self._active_mono_duals:
            print(f"\t\t* {label}: active monotonicity constraints (Lagrange multipliers):")
            for (i_lo, i_hi), dual in self._active_mono_duals.items():
                print(f"\t\t\t{list(self.bounds.keys())[i_lo]} -> {list(self.bounds.keys())[i_hi]}:  mu = {dual:+.3e}")

        return x_k, {"n_iter": inner_it + 1, "stop_reason": stop_reason, "grad_mag": grad_mag}

    # ------------------------------------------------------------------
    # Override _step: weighted LM on the surrogate replaces acquisition
    # ------------------------------------------------------------------

    @mitim_timer(lambda self: f'wLM @ {self.currentIteration}', log_file=lambda self: self.timings_file)
    def _step(self):
        """
        Fit GP surrogates, then run weighted Levenberg-Marquardt (with a
        constrained QP refinement step) on the surrogate to propose x_next.
        Drop-in replacement for MITIM_BO._step(); the outer run() loop is
        fully inherited and unchanged.
        """
        train_Ystd = (
            self.train_Ystd
            if self.optimization_options["evaluation_options"]["train_Ystd"] is None
            else self.optimization_options["evaluation_options"]["train_Ystd"]
        )

        # ---- Fit surrogate (identical preamble to MITIM_BO._step / MITIM_Anderson._step)
        current_step = STEPtools.OPTstep(
            self.train_X,
            self.train_Y,
            train_Ystd,
            bounds=self.bounds,
            stepSettings=self.stepSettings,
            currentIteration=self.currentIteration,
            strategy_options=self.strategy_options_use,
            BOmetrics=self.BOmetrics,
            surrogate_parameters=self.surrogate_parameters,
        )
        current_step.strategy_options_use = copy.deepcopy(self.strategy_options_use)
        self.steps.append(current_step)

        avoidPoints = np.append(self.avoidPoints_failed, self.avoidPoints_outside)
        self.avoidPoints = np.unique([int(j) for j in avoidPoints])
        self.steps[-1].fit_step(avoidPoints=self.avoidPoints)
        self.steps[-1].defineFunctions(self.scalarized_objective)

        if self.storeClass:
            self.save()

        if self.hard_finish:
            self.steps[-1].x_next = None
            return

        combined_gp = self.steps[-1].GP["combined_model"]
        ctx = self._lm_context()

        # ---- Start the LM solve from the best evaluated point so far -------
        ind_best = self.BOmetrics["overall"]["indBest"]
        x_init = torch.tensor(self.train_X[ind_best]).to(self.dfT)

        x_k, info = self._run_inner_lm(x_init, combined_gp, ctx, label="wLM")

        # Store proposed point; shape (1, n_DV) matches MITIM_BO convention
        self.steps[-1].x_next = x_k.unsqueeze(0).to(self.dfT)
        print(
            f"\t- wLM proposed x_next ({info['n_iter']} iters, {info['stop_reason']}): "
            f"{x_k.cpu().numpy()}"
        )

    # ------------------------------------------------------------------
    # Real-model confirmation (the surrogate is only ever a proxy)
    # ------------------------------------------------------------------

    def _run_lcfs_jitter_and_retrain(self, x_int_star, n_jitter):
        """
        Run n_jitter real transport evaluations with x_lcfs drawn uniformly
        from the Phase-3 bracket, append the results to the training set, and
        refit the GP.  Returns the updated combined_model for use by
        _assess_lcfs_feasibility.

        x_int is lightly jittered around x_int_star for each draw so the GP
        sees n_jitter distinct input points rather than n_jitter identical
        copies of x_int_star (identical inputs with varied outputs appear as
        noise to the GP and raise inferred likelihood variance instead of
        reducing posterior uncertainty).

        This is done in MITIM_wLM.run() -- after BO convergence, before Phase
        3 -- so the GP is already well-trained on x_int space.  x_lcfs
        variation at initialization time looks like unexplained noise (x_lcfs
        is not a GP feature), degrades early BO convergence, and was the
        reason initialization_sr_w_lcfs_jitter was removed.
        """
        Sigma_lcfs = self.wlm_options["Sigma_lcfs"]
        y_or_aly = self.wlm_options["y_or_aly"]
        sigma_is_rel = self.wlm_options["sigma_is_rel"]
        n_sigma = float(self.wlm_options["lcfs_n_sigma"])

        powerstate = self.optimization_object.powerstate
        if not (hasattr(powerstate, "bc_dict") and isinstance(powerstate.bc_dict, dict) and len(powerstate.bc_dict) > 0):
            print("\t* Phase 3 jitter: powerstate has no bc_dict -- skipping", typeMsg="w")
            return self.steps[-1].GP["combined_model"]

        families_by_mode = {"y": ["y"], "aly": ["aly"], "both": ["y", "aly"]}
        families = families_by_mode.get(y_or_aly, ["y"])

        def _sigma_val(sigma_spec, family):
            # Per-family sigma from a Sigma_lcfs[channel] spec. The gradient
            # family is tokenised "aly" internally but users commonly key the
            # dict "aLy" (matching the bc_dict "aL{channel}" convention), so
            # accept both spellings -- otherwise the gradient sigma silently
            # resolves to 0 and its draws become no-ops.
            if isinstance(sigma_spec, dict):
                keys = ("y",) if family == "y" else ("aly", "aLy", "aLY", "aL")
                for k in keys:
                    if k in sigma_spec:
                        return float(sigma_spec[k])
                return float(sigma_spec.get("both", 0.0))
            if isinstance(sigma_spec, (list, tuple)) and len(sigma_spec) == 2:
                return float(sigma_spec[0] if family == "y" else sigma_spec[1])
            return float(sigma_spec)

        survey_targets = []   # (bc_key, sigma_abs, nominal_val, roa_loc)
        for channel, sigma_spec in Sigma_lcfs.items():
            for family in families:
                bc_key = channel if family == "y" else f"aL{channel}"
                if bc_key not in powerstate.bc_dict:
                    continue
                nominal_val, roa_loc = powerstate.bc_dict[bc_key]
                sv = _sigma_val(sigma_spec, family)
                sigma_abs = sv * abs(float(nominal_val)) if sigma_is_rel else float(sv)
                survey_targets.append((bc_key, sigma_abs, float(nominal_val), roa_loc))

        if not survey_targets:
            print("\t* Phase 3 jitter: no bc_dict targets resolved -- skipping", typeMsg="w")
            return self.steps[-1].GP["combined_model"]

        bounds_arr = np.array(list(self.bounds.values()))   # (n_DV, 2)
        lb_np, ub_np = bounds_arr[:, 0], bounds_arr[:, 1]
        x_star_np = x_int_star.detach().cpu().numpy().flatten()
        bounds_width = ub_np - lb_np

        rng = np.random.default_rng(getattr(self, "seed", 0) or 0)

        print(f"\n\t--- Phase 3 pre-step: {n_jitter} real evaluation(s) across x_lcfs bracket + GP retraining ---")

        new_X, new_Y, new_Ystd = [], [], []
        for j in range(n_jitter):
            # Draw x_lcfs uniformly from the bracket for each target
            bc_overrides = {}
            for bc_key, sigma_abs, nominal_val, roa_loc in survey_targets:
                draw = nominal_val + rng.uniform(-n_sigma * sigma_abs, n_sigma * sigma_abs)
                bc_overrides[bc_key] = (draw, roa_loc)

            # Light x_int jitter (1% of bounds width) so GP inputs are distinct
            x_j = np.clip(
                x_star_np + rng.normal(size=x_star_np.shape) * 0.01 * bounds_width,
                lb_np, ub_np,
            )

            for bc_key, (val, loc) in bc_overrides.items():
                powerstate.bc_dict[bc_key] = [val, loc]
            try:
                _, y_j, ystd_j = self._evaluate_real(
                    x_j,
                    label=f"Phase3 jitter {j + 1}/{n_jitter}",
                    sync_csv=False,
                )
            finally:
                for bc_key, _, nominal_val, roa_loc in survey_targets:
                    powerstate.bc_dict[bc_key] = [nominal_val, roa_loc]

            new_X.append(x_j)
            new_Y.append(np.atleast_2d(y_j).flatten())
            new_Ystd.append(np.atleast_2d(ystd_j).flatten())

        # Append to training arrays and sync optimization_data
        self.train_X = np.append(self.train_X, np.array(new_X), axis=0)
        self.train_Y = np.append(self.train_Y, np.array(new_Y), axis=0)
        self.train_Ystd = np.append(self.train_Ystd, np.array(new_Ystd), axis=0)

        _, _, objective = self.optimization_object.scalarized_objective(
            torch.from_numpy(self.train_Y).to(self.dfT)
        )
        self.optimization_data.update_points(
            self.train_X, Y=self.train_Y, Ystd=self.train_Ystd,
            objective=objective.cpu().numpy(),
        )

        # Refit GP on the enlarged dataset (not appended to self.steps to avoid
        # disturbing the BO loop's step bookkeeping; the refitted GP is used
        # only by _assess_lcfs_feasibility and then discarded).
        train_Ystd_fit = (
            self.train_Ystd
            if self.optimization_options["evaluation_options"]["train_Ystd"] is None
            else self.optimization_options["evaluation_options"]["train_Ystd"]
        )
        refit_step = STEPtools.OPTstep(
            self.train_X,
            self.train_Y,
            train_Ystd_fit,
            bounds=self.bounds,
            stepSettings=self.stepSettings,
            currentIteration=self.currentIteration,
            strategy_options=self.strategy_options_use,
            BOmetrics=self.BOmetrics,
            surrogate_parameters=self.surrogate_parameters,
        )
        avoidPoints = np.unique([int(k) for k in np.append(self.avoidPoints_failed, self.avoidPoints_outside)])
        refit_step.fit_step(avoidPoints=avoidPoints)

        print("\t* Phase 3 jitter: GP refitted with bracket-survey data")
        return refit_step.GP["combined_model"]

    def _evaluate_real(self, x, label="", sync_csv=True):
        """
        Run the *real* transport model at an explicit point x -- not
        self.x_next, so this cannot disturb the outer run() loop's
        bookkeeping -- via EVALUATORtools.fun, the exact pathway
        MITIM_BO._evaluate uses to populate the GP training set (same
        Execution/ folder + numEval bookkeeping, so these calls are
        indistinguishable from ordinary real evaluations on disk).

        Used by Phase 3 to confirm surrogate-based conclusions: a claim as
        consequential as "re-optimizing at a nearby x_lcfs converges to a
        better operating point" should not be reported on surrogate
        evidence alone.

        sync_csv=False skips the update_data_point call inside EVALUATORtools
        (by passing optimization_data=None).  Use this when the caller manages
        CSV writes itself (e.g. _run_lcfs_jitter_and_retrain, which batch-syncs
        via update_points after the loop) or when the evaluation is diagnostic
        and should not enter the training CSV (e.g. _assess_lcfs_feasibility
        validation runs).  Also prevents a stale cold_start=False cache hit when
        bc_dict has been modified for the evaluation.

        Returns (phi, y, ystd); phi follows the scalarized_objective
        "value to maximize" convention used throughout this class.
        """
        x_np = np.atleast_2d(x.detach().cpu().numpy() if torch.is_tensor(x) else x)

        y, ystd, self.numEval = EVALUATORtools.fun(
            self.optimization_object,
            x_np,
            self.folderExecution,
            self.bounds,
            self.outputs,
            self.optimization_data if sync_csv else None,
            parallel=self.parallel_evaluations,
            cold_start=True,  # Always fresh: called at perturbed x_lcfs, not valid CSV cache
            numEval=self.numEval,
        )

        _, _, phi = self.scalarized_objective(torch.as_tensor(y).to(self.dfT))
        phi_val = phi.squeeze().item()

        if label:
            print(f"\t\t  [real-model check: {label}]  phi_real = {phi_val:+.4e}")

        return phi_val, y, ystd

    # ------------------------------------------------------------------
    # Phase 3 refinement: iterative real-model + inner-wLM at best x_lcfs
    # ------------------------------------------------------------------

    def _refine_lcfs_candidate(self, best_sample, nominal_bc, phi_nominal_real, combined_gp):
        """
        Refine the best LCFS BC candidate via iterative real evaluation + GP
        retraining + inner wLM re-optimization (phase 2 style, but anchored to
        the converged x_lcfs).

        If lcfs_refine_n_evals == 0: single confirmation check (minimal validation).
        If lcfs_refine_n_evals > 0: run that many real evaluations with warm-started
        inner wLM optimization in between, retraining the GP after each new point.

        Warm-starts from the best_sample's x_int and x_lcfs (phase 3 best BC).
        Returns a refined version of best_sample with accumulated real-model results.
        """
        n_refine_evals = max(0, int(self.wlm_options.get("lcfs_refine_n_evals", 0)))

        powerstate = self.optimization_object.powerstate
        ctx = self._lm_context()

        x_int_current = torch.tensor(
            best_sample["x_int_star"], dtype=self.dfT.dtype, device=self.dfT.device
        )

        print(
            f"\n\t--- Phase 3 refinement: best x_lcfs candidate "
            f"({n_refine_evals} real eval(s) + inner wLM) ---"
        )

        refine_history = []
        for refine_iter in range(n_refine_evals + 1):
            label_iter = f"Refine {refine_iter + 1}/{n_refine_evals + 1}"

            # Real evaluation at current x_int and best x_lcfs
            # Pass optimization_data directly so EVALUATORtools syncs the CSV
            for bc_key, val in best_sample["x_lcfs"].items():
                powerstate.bc_dict[bc_key] = [val, nominal_bc[bc_key][1]]
            try:
                x_np = np.atleast_2d(x_int_current.detach().cpu().numpy())
                y_real, ystd_real, self.numEval = EVALUATORtools.fun(
                    self.optimization_object,
                    x_np,
                    self.folderExecution,
                    self.bounds,
                    self.outputs,
                    self.optimization_data,  # Sync directly to CSV
                    parallel=self.parallel_evaluations,
                    cold_start=True,  # Always fresh: perturbed x_lcfs ≠ any prior CSV entry
                    numEval=self.numEval,
                )
                _, _, phi_real_t = self.scalarized_objective(torch.as_tensor(y_real).to(self.dfT))
                phi_real = phi_real_t.squeeze().item()

                print(f"\t\t  [Phase3 refine {label_iter}]  phi_real = {phi_real:+.4e}")
            finally:
                for bc_key in best_sample["x_lcfs"]:
                    powerstate.bc_dict[bc_key] = list(nominal_bc[bc_key])

            refine_history.append({
                "eval_iter": refine_iter + 1,
                "phi_real": phi_real,
                "x_int": x_int_current.detach().cpu().numpy().copy(),
                "y_real": y_real.copy() if hasattr(y_real, "copy") else np.array(y_real),
                "ystd_real": ystd_real.copy() if hasattr(ystd_real, "copy") else np.array(ystd_real),
            })

            # If this was the final evaluation, don't optimize further
            if refine_iter >= n_refine_evals:
                break

            # Append the new point to training data and retrain GP
            self.train_X = np.append(
                self.train_X, x_int_current.detach().cpu().numpy().reshape(1, -1), axis=0
            )
            y_real_flat = np.atleast_2d(y_real).flatten()
            ystd_real_flat = np.atleast_2d(ystd_real).flatten()
            self.train_Y = np.append(self.train_Y, y_real_flat.reshape(1, -1), axis=0)
            self.train_Ystd = np.append(self.train_Ystd, ystd_real_flat.reshape(1, -1), axis=0)

            # Refit GP
            train_Ystd_fit = (
                self.train_Ystd
                if self.optimization_options["evaluation_options"]["train_Ystd"] is None
                else self.optimization_options["evaluation_options"]["train_Ystd"]
            )
            refit_step = STEPtools.OPTstep(
                self.train_X,
                self.train_Y,
                train_Ystd_fit,
                bounds=self.bounds,
                stepSettings=self.stepSettings,
                currentIteration=self.currentIteration,
                strategy_options=self.strategy_options_use,
                BOmetrics=self.BOmetrics,
                surrogate_parameters=self.surrogate_parameters,
            )
            avoidPoints = np.unique(
                [int(k) for k in np.append(self.avoidPoints_failed, self.avoidPoints_outside)]
            )
            refit_step.fit_step(avoidPoints=avoidPoints)
            combined_gp = refit_step.GP["combined_model"]

            print(f"\t\t* Phase 3 refinement: GP refitted after eval #{refine_iter + 1}")

            # Run inner wLM to optimize x_int at the best x_lcfs
            max_iter_refine = max(
                1, int(self.wlm_options.get("lcfs_reopt_refine_max_iter", 3))
            )
            x_int_current, info = self._run_inner_lm(
                x_int_current, combined_gp, ctx,
                max_iter=max_iter_refine,
                label=f"Phase3-Refine[{refine_iter + 2}/{n_refine_evals + 1}]",
                verbose=True,
            )

        # Assemble final refined sample
        best_sample["refine_history"] = refine_history
        best_sample["x_int_final"] = x_int_current.detach().cpu().numpy()
        best_sample["phi_real_final"] = refine_history[-1]["phi_real"]
        best_sample["delta_phi_real_final"] = (
            refine_history[-1]["phi_real"] - phi_nominal_real
        )
        best_sample["n_refine_evals"] = n_refine_evals

        return best_sample

    # ------------------------------------------------------------------
    # Apply optimized x_lcfs from phase 3 refinement (for post-analysis)
    # ------------------------------------------------------------------

    def apply_best_x_lcfs_optimized(self):
        """
        Apply the optimized x_lcfs boundary conditions from phase 3 refinement
        to the powerstate. Call this before post-analysis / plotting to ensure
        the final results reflect the optimized (not modeled) boundary conditions.

        WARNING: This must be called BEFORE any powerstate.calculate() or
        calculateBoundaryConditions() that would overwrite bc_dict, otherwise
        the optimized values will be lost.

        Returns True if optimized values were applied, False if none available.
        """
        if not hasattr(self, "best_x_lcfs_optimized") or not self.best_x_lcfs_optimized:
            print("\t* No optimized x_lcfs available (phase 3 refinement may not have run)", typeMsg="i")
            return False

        powerstate = self.optimization_object.powerstate
        if not hasattr(powerstate, "bc_dict"):
            print("\t* powerstate has no bc_dict -- cannot apply optimized x_lcfs", typeMsg="w")
            return False

        print("\n\t--- Applying optimized x_lcfs from phase 3 refinement ---")
        for bc_key, val in self.best_x_lcfs_optimized.items():
            if bc_key in powerstate.bc_dict:
                nominal_val, roa_loc = powerstate.bc_dict[bc_key]
                powerstate.bc_dict[bc_key] = [val, roa_loc]
                print(f"\t\t{bc_key}: {nominal_val:+.4e} -> {val:+.4e}")
            else:
                print(f"\t\t{bc_key}: NOT found in bc_dict (skipping)", typeMsg="w")

        return True

    # ------------------------------------------------------------------
    # Phase 3: x_lcfs feasibility assessment (nested re-optimization)
    # ------------------------------------------------------------------

    def _assess_lcfs_feasibility(self, x_int_star, combined_gp):
        """
        After convergence at x_lcfs_nominal with x_int_star, ask: can the
        objective be reduced *further* by varying the LCFS boundary
        conditions within their modeled uncertainty Sigma_lcfs? The edge
        model supplies x_lcfs as uncertain parameters (deterministic priors,
        not design variables), so the converged objective is only as good as
        the assumed x_lcfs -- this quantifies how objective-consequential
        that boundary uncertainty is.

        Mechanism -- joint Monte Carlo + nested re-optimization:
        draw lcfs_n_samples *full* x_lcfs vectors from the joint prior
        (diagonal Sigma_lcfs: independent truncated normals per boundary
        parameter, N(nominal, sigma_abs^2) clipped to +/- lcfs_n_sigma*sigma_abs
        to stay inside the modeled prior and limit GP extrapolation through
        the bc -> profile -> feature path). For each draw, perturb every
        powerstate.bc_dict[bc_key] at once and re-run the *same* weighted-LM
        inner loop (_run_inner_lm, capped at lcfs_reopt_max_iter, warm-started
        from x_int_star) to find x_int*(x_lcfs_draw). All boundary parameters
        move together per draw (the uncertainty is on the boundary vector as a
        whole), and x_int is re-tuned against each draw -- this is the joint
        (x_int, x_lcfs) sweep a one-at-a-time or fixed-x_int* sensitivity cannot
        capture. Re-running the proven inner loop (rather than novel
        d(mu_F)/d(x_lcfs) derivative machinery) reuses 100% tested code;
        warm-starting + capping keeps each sub-solve cheap.

        `y_or_aly` selects which bc_dict famil(ies) are perturbed per channel:
        "y" -> plain value keys "te"/"ti"/"ne"/"ni", "aly" -> gradient keys
        "aLte"/"aLti"/"aLne"/"aLni", or "both" -> perturb each channel's "y"
        *and* "aly" keys (each with its own sigma, drawn jointly).
        Sigma_lcfs is keyed by the plain channel name regardless, since
        bc_dict carries both families simultaneously; in "both" mode
        Sigma_lcfs[channel] may be a 2-sequence (sigma_y, sigma_aly) or a
        {"y":.., "aLy":..} dict to give each family its own sigma (the two
        live on different scales -- e.g. keV vs a dimensionless
        gradient-scale-length ratio -- so a shared absolute number is not
        generally meaningful for both; a bare scalar is broadcast to both
        only when sigma_is_rel=True, since a relative fraction is scale-free).
        The gradient-family key is accepted as "aLy"/"aly" interchangeably.
        `sigma_is_rel` lets each resolved sigma be a fraction of that
        family's own nominal BC value (sigma_abs = sigma_rel * |nominal_val|)
        rather than an absolute std-dev the user would otherwise pre-compute.

        Outcome -- NOT a feasibility verdict. phi is "a value to maximize"
        (= -residual), so delta_phi = phi(x_int*(x_lcfs_draw)) - phi_nominal
        > 0 means a draw lowered the objective. The report gives the fraction
        of draws that improve, the best draw's reduction and the x_lcfs shift
        driving it, and -- crucially -- whether the best reduction clears the
        surrogate's own noise floor (sigma_rel*|phi_nominal|) so small,
        noise-level "improvements" are not over-claimed. No absolute
        acceptable-residual / convergence threshold is applied.

        Real-model confirmation: the surrogate triages lcfs_n_samples draws
        cheaply, but the top `lcfs_validate_n_best` most-improving draws are
        *always* re-evaluated with the real transport model (_evaluate_real,
        the same EVALUATORtools.fun pathway MITIM_BO._evaluate uses) before
        the verdict is finalised -- the surrogate is only ever a proxy.

        Switchable: returns None (no-op) unless
        wlm_options["lcfs_feasibility_enabled"] and a non-empty
        wlm_options["Sigma_lcfs"] are provided. Runs once, at the final
        converged point, via the run() override.
        """
        if not self.wlm_options["lcfs_feasibility_enabled"]:
            return None

        Sigma_lcfs = self.wlm_options["Sigma_lcfs"]
        if not Sigma_lcfs:
            print("\t* Phase 3: no Sigma_lcfs provided -- skipping x_lcfs feasibility check", typeMsg="i")
            return None

        powerstate = self.optimization_object.powerstate
        if not (hasattr(powerstate, "bc_dict") and isinstance(powerstate.bc_dict, dict) and len(powerstate.bc_dict) > 0):
            print("\t* Phase 3: powerstate has no bc_dict (not an edge state) -- skipping", typeMsg="i")
            return None

        y_or_aly = self.wlm_options["y_or_aly"]
        sigma_is_rel = self.wlm_options["sigma_is_rel"]
        n_sigma = self.wlm_options["lcfs_n_sigma"]
        max_reopt_iter = self.wlm_options["lcfs_reopt_max_iter"]
        stds = self.wlm_options["stds"]
        sigma_rel_tol = self.wlm_options["sigma_rel_tol"]

        # Which bc_dict famil(ies) to probe per channel, and how to pull
        # that family's sigma out of a Sigma_lcfs[channel] spec that may be
        # a bare scalar (broadcast) or a (sigma_y, sigma_aly) / {"y":..,
        # "aly":..} pair (only meaningful -- and only needed -- in "both"
        # mode, where the two families generally live on different scales).
        families_by_mode = {"y": ["y"], "aly": ["aly"], "both": ["y", "aly"]}
        families = families_by_mode.get(y_or_aly, ["y"])

        def _sigma_value_for_family(sigma_spec, family):
            # See _run_lcfs_jitter_and_retrain._sigma_val: the gradient family
            # is tokenised "aly" but is commonly keyed "aLy" in the user spec
            # (bc_dict uses "aL{channel}"), so accept both -- a key mismatch
            # here silently zeroes the gradient sigma and makes its draws no-ops.
            if isinstance(sigma_spec, dict):
                keys = ("y",) if family == "y" else ("aly", "aLy", "aLY", "aL")
                for k in keys:
                    if k in sigma_spec:
                        return float(sigma_spec[k])
                return float(sigma_spec.get("both", 0.0))
            if isinstance(sigma_spec, (list, tuple)) and len(sigma_spec) == 2:
                return float(sigma_spec[0] if family == "y" else sigma_spec[1])
            return float(sigma_spec)

        # Map plain channel name -> [(family, bc_key), ...] for this run
        # (bc_dict carries both "te"/"ti"/"ne" and "aLte"/"aLti"/"aLne"
        # simultaneously; normally only one family is varied, but "both"
        # probes each independently).
        bc_keys = {}
        for channel in Sigma_lcfs:
            entries = []
            for family in families:
                bc_key = channel if family == "y" else f"aL{channel}"
                if bc_key in powerstate.bc_dict:
                    entries.append((family, bc_key))
            if entries:
                bc_keys[channel] = entries

        if not bc_keys:
            print(
                f"\t* Phase 3: none of the Sigma_lcfs channels {list(Sigma_lcfs.keys())} "
                f"map to bc_dict keys (y_or_aly='{y_or_aly}') present in "
                f"{list(powerstate.bc_dict.keys())} -- skipping",
                typeMsg="w",
            )
            return None

        nominal_bc = {
            bc_key: list(powerstate.bc_dict[bc_key])
            for entries in bc_keys.values() for _family, bc_key in entries
        }

        # Index of the converged nominal solution: the reference every MC draw
        # is compared against. No feasibility threshold is read here -- Phase 3
        # asks "can the objective be reduced further by varying x_lcfs within
        # its uncertainty?", answered by the *change* in objective under
        # re-optimization, not by an absolute acceptable-residual bound.
        ind_best = self.BOmetrics["overall"]["indBest"]
        n_samples = max(1, int(self.wlm_options.get("lcfs_n_samples", 128)))

        ctx = self._lm_context()

        def _phi_at(x):
            x_eval = x.detach().clone().unsqueeze(0)
            with torch.no_grad():
                y_mean, y_upper, y_lower, _ = combined_gp.predict(x_eval)
                _, _, phi = self.scalarized_objective(y_mean)
            sigma_y = (y_upper - y_lower).squeeze(0).abs() / (2.0 * stds)
            sigma_rel = sigma_y.norm().item() / (y_mean.squeeze(0).norm().item() + 1e-12)
            return phi.squeeze().item(), sigma_rel

        # Resolve the per-key perturbation targets once: every (channel,
        # family) present in bc_dict gets its own absolute sigma. All targets
        # are drawn *jointly* per MC sample -- the modeled uncertainty is on
        # the boundary vector as a whole, and x_int is re-optimized against
        # each full draw (not one parameter at a time).
        targets = []  # (channel, family, bc_key, nominal_val, roa_loc, sigma_abs)
        for channel, entries in bc_keys.items():
            sigma_spec = Sigma_lcfs[channel]
            for family, bc_key in entries:
                sigma_val = _sigma_value_for_family(sigma_spec, family)
                nominal_val, roa_loc = nominal_bc[bc_key]
                sigma_abs = sigma_val * abs(nominal_val) if sigma_is_rel else sigma_val
                targets.append((channel, family, bc_key, nominal_val, roa_loc, sigma_abs))

        active = [t for t in targets if t[5] > 0.0]
        if not active:
            print(
                "\t* Phase 3: every resolved Sigma_lcfs sigma is zero (check the "
                "Sigma_lcfs spec / y_or_aly / key spelling, e.g. 'aLy' vs 'aly') -- "
                "no x_lcfs variation to sample, skipping",
                typeMsg="w",
            )
            return None

        rng = np.random.default_rng(getattr(self, "seed", 0) or 0)

        result = None
        try:
            # Re-optimize x_int with the current (post-jitter) GP at nominal x_lcfs.
            # x_int_star was found by the pre-jitter GP; after jitter retraining the
            # GP landscape may have shifted so x_int_star is no longer optimal.
            # Using x_int_star directly as the baseline would make every MC draw
            # (which re-optimizes x_int on the new GP) look artificially better.
            # This gives a fair baseline and the correct warm-start for MC draws.
            x_int_nominal, _info_nom = self._run_inner_lm(
                x_int_star, combined_gp, ctx,
                max_iter=max_reopt_iter,
                label="Phase3-Nominal",
                verbose=True,
            )
            phi_nominal, sigma_rel_nominal = _phi_at(x_int_nominal)

            print("\n\t--- Phase 3: x_lcfs sensitivity (joint MC + nested re-optimization) ---")
            print(
                f"\t\t{n_samples} joint draws over {len(active)} boundary parameter(s) "
                f"[{', '.join(t[2] for t in active)}]; truncated normal at +/-{n_sigma:.1f} sigma"
            )
            print(f"\t\tnominal (surrogate):  phi={phi_nominal:+.4e}  (sigma/|y|={sigma_rel_nominal:.3e})")

            # --- Joint Monte Carlo: each draw perturbs the *full* x_lcfs vector,
            # then re-optimizes x_int on the GP (warm-started from x_int_nominal,
            # the post-jitter optimal, capped). Surrogate-only -- a cheap sweep;
            # only the best draw is later confirmed against the real model.
            # verbose=False keeps the per-draw inner-loop traces out of the log.
            samples = []
            for s in range(n_samples):
                draws, zdraws = {}, {}   # bc_key -> drawn value / its z-score
                for _ch, _fam, bc_key, nominal_val, roa_loc, sigma_abs in active:
                    # Truncated standard normal -> draw within +/- n_sigma*sigma_abs
                    z = float(np.clip(rng.standard_normal(), -n_sigma, n_sigma))
                    val = nominal_val + z * sigma_abs
                    draws[bc_key], zdraws[bc_key] = val, z
                    powerstate.bc_dict[bc_key] = [val, roa_loc]
                try:
                    x_reopt, info = self._run_inner_lm(
                        x_int_nominal, combined_gp, ctx,
                        max_iter=max_reopt_iter,
                        label=f"Phase3[MC {s + 1}/{n_samples}]",
                        verbose=False,
                    )
                    phi_s, sigma_rel_s = _phi_at(x_reopt)
                finally:
                    for bc_key in draws:   # restore before the next draw
                        powerstate.bc_dict[bc_key] = list(nominal_bc[bc_key])

                samples.append({
                    "sample": s,
                    "x_lcfs": dict(draws),
                    "z": dict(zdraws),
                    "x_int_star": x_reopt.detach().cpu().numpy(),
                    "phi": phi_s,
                    "sigma_rel": sigma_rel_s,
                    "delta_phi": phi_s - phi_nominal,
                    "n_reopt_iter": info["n_iter"],
                    "reopt_stop_reason": info["stop_reason"],
                })

            delta = np.array([s["delta_phi"] for s in samples])
            phis = np.array([s["phi"] for s in samples])
            sigma_rel_max = max([sigma_rel_nominal] + [s["sigma_rel"] for s in samples])
            n_improving = int((delta > 0).sum())
            frac_improving = n_improving / len(samples)
            best_sample = max(samples, key=lambda s: s["delta_phi"])

            def _shift_str(sample):
                return ", ".join(
                    f"{sample['x_lcfs'][k] - nominal_bc[k][0]:+.3e} on {k}"
                    for k in sample["x_lcfs"]
                )

            # --- Real-model confirmation and refinement of the best draw -------
            # The surrogate triages n_samples cheaply; the real transport model
            # has the final word on whether a predicted reduction is real. The
            # best draw is then optionally refined with iterative real-model
            # evaluations + inner wLM re-optimization (lcfs_refine_n_evals).
            y_nominal_real = torch.as_tensor(np.atleast_2d(self.train_Y[ind_best])).to(self.dfT)
            _, _, phi_nominal_real_t = self.scalarized_objective(y_nominal_real)
            phi_nominal_real = phi_nominal_real_t.squeeze().item()

            best_sample_refined = self._refine_lcfs_candidate(
                best_sample, nominal_bc, phi_nominal_real, combined_gp
            )

            # Determine validation status from the refined result
            phi_real_final = best_sample_refined["phi_real_final"]
            delta_phi_real_final = best_sample_refined["delta_phi_real_final"]
            any_confirmed_better = delta_phi_real_final > 0
            best_validated = best_sample_refined

            # --- Outcome: can the objective be reduced by varying x_lcfs? -----
            # phi is "a value to maximize" (= -residual), so delta_phi > 0 means
            # a draw lowered the objective below the converged nominal. A claimed
            # reduction is only credible above the surrogate's own noise floor,
            # so the best draw must clear sigma_rel_nominal*|phi_nominal|.
            noise_floor = sigma_rel_nominal * abs(phi_nominal)
            if sigma_rel_max > sigma_rel_tol:
                outcome = "insufficient_coverage"
                verdict = (
                    f"Surrogate uncertainty too large over the sampled x_lcfs range "
                    f"(max sigma/|y|={sigma_rel_max:.3e} > {sigma_rel_tol}) -- take targeted "
                    "real evaluations at perturbed x_lcfs before trusting this assessment"
                )
            elif best_sample["delta_phi"] > noise_floor:
                outcome = "objective_reducible"
                verdict = (
                    f"Surrogate: the objective can be reduced further by moving x_lcfs within its "
                    f"uncertainty -- best of {n_samples} draws improves by Delta phi = "
                    f"{best_sample['delta_phi']:+.4e} (above the {noise_floor:.2e} surrogate noise floor; "
                    f"{n_improving}/{len(samples)} = {frac_improving:.0%} of draws improve). Driving x_lcfs "
                    f"shift: {_shift_str(best_sample)}. The modeled x_lcfs uncertainty is "
                    "objective-consequential -- worth tightening the upstream edge-model estimate"
                )
            else:
                outcome = "robust"
                verdict = (
                    f"No sampled x_lcfs (of {n_samples}) reduces the objective beyond the surrogate "
                    f"noise floor ({noise_floor:.2e}); best Delta phi = {best_sample['delta_phi']:+.4e}. "
                    "The converged solution is robust to the modeled x_lcfs uncertainty"
                )

            # Real-model check is the deciding word on a claimed reduction.
            if best_validated:
                if any_confirmed_better:
                    x_int_final = best_validated.get("x_int_final", best_validated["x_int_star"])
                    verdict += (
                        f"  ||  REAL MODEL CONFIRMS a reduction (Delta phi_real = "
                        f"{best_validated['delta_phi_real_final']:+.4e}; x_lcfs shift {_shift_str(best_validated)}; "
                        f"x_int* = {x_int_final}) -- act on this"
                    )
                elif outcome == "objective_reducible":
                    verdict += (
                        "  ||  REAL MODEL DOES NOT CONFIRM the surrogate-predicted reduction at the "
                        "best draw -- treat the surrogate-side finding as unconfirmed/likely surrogate noise"
                    )
                else:
                    verdict += "  ||  real-model check on the best draw found no reduction -- consistent with the surrogate-side verdict"

            # --- Report -------------------------------------------------------
            print(f"\t\tnominal (real model):  phi_real={phi_nominal_real:+.4e}")
            print(
                f"\t\tre-optimized phi over {n_samples} draws:  min={phis.min():+.4e}  "
                f"median={np.median(phis):+.4e}  max(best)={phis.max():+.4e}  "
                f"[Delta phi range {delta.min():+.3e} .. {delta.max():+.3e}]"
            )
            print(f"\t\tdraws that reduce the objective: {n_improving}/{len(samples)} ({frac_improving:.0%})")
            bs = best_sample_refined
            real_str = ""
            n_refine = bs.get("n_refine_evals", 0)
            if n_refine > 0:
                real_str = (
                    f"   [refined: {n_refine} real eval(s) + inner wLM; "
                    f"phi_final={bs['phi_real_final']:+.4e}  Delta_final={bs['delta_phi_real_final']:+.4e}]"
                )
            else:
                real_str = f"   [real check: phi={bs['phi_real_final']:+.4e}  Delta={bs['delta_phi_real_final']:+.4e}]"
            print(
                f"\t\tbest draw (#{bs['sample']}):  phi={bs['phi']:+.4e}  Delta={bs['delta_phi']:+.4e}  "
                f"({bs['n_reopt_iter']} reopt iters, {bs['reopt_stop_reason']}){real_str}"
            )
            print("\t\t  x_lcfs @ best draw:  " + "  ".join(
                f"{k}={bs['x_lcfs'][k]:+.4e}(z={bs['z'][k]:+.2f})" for k in bs["x_lcfs"]
            ))
            print(f"\t\tverdict: {verdict}")

            # Store optimized x_lcfs on the instance for persistence across post-analysis
            # WARNING: if calculateBoundaryConditions() is called on powerstate after this,
            # it will overwrite bc_dict with edge-model predictions, negating this optimization.
            # Post-analysis code MUST apply best_x_lcfs_optimized if available.
            self.best_x_lcfs_optimized = best_sample_refined.get("x_lcfs", {})
            self.best_x_int_optimized = best_sample_refined.get("x_int_final", best_sample_refined.get("x_int_star"))

            result = {
                "outcome": outcome,
                "verdict": verdict,
                "n_samples": n_samples,
                "phi_nominal": phi_nominal,
                "phi_nominal_real": phi_nominal_real,
                "sigma_rel_nominal": sigma_rel_nominal,
                "noise_floor": noise_floor,
                "n_improving": n_improving,
                "frac_improving": frac_improving,
                "delta_phi_min": float(delta.min()),
                "delta_phi_max": float(delta.max()),
                "samples": samples,
                "best_sample_surrogate": best_sample,
                "best_sample_refined": best_sample_refined,
                "real_model_confirms_reduction": any_confirmed_better,
                "n_refine_evals_performed": best_sample_refined.get("n_refine_evals", 0),
                "best_x_lcfs_optimized": self.best_x_lcfs_optimized,
                "best_x_int_optimized": self.best_x_int_optimized,
            }

        finally:
            # Always leave the shared powerstate in its nominal state, and
            # force one more rebuild so `powerstate.plasma` is consistent
            # with the restored `bc_dict` (predict() invalidates its cache on
            # every call regardless, but other code may read powerstate.plasma
            # directly between now and the next surrogate evaluation).
            for bc_key, (val, roa_loc) in nominal_bc.items():
                powerstate.bc_dict[bc_key] = [val, roa_loc]
            _ = _phi_at(x_int_star)

        return result

# ----------------------------------------------------------------------
# Stopping criteria
# ----------------------------------------------------------------------

def max_val(maximum_value_orig, maximum_value_is_rel, res_base):
    if maximum_value_is_rel:
        maximum_value = maximum_value_orig * res_base
        print(f'\t* Maximum value for convergence provided as relative value of {maximum_value_orig} from base {res_base:.3e} --> {maximum_value:.3e}')
    else:
        maximum_value = maximum_value_orig
        print(f'\t* Maximum value for convergence: {maximum_value} (starting case has {res_base:.3e})' )

    return maximum_value

def stopping_criteria_default(mitim_bo, parameters = {}):


    print('\n')
    print('--------------------------------------------------')
    print('Convergence criteria')
    print('--------------------------------------------------')

    # ------------------------------------------------------------------------------------
    # Determine the stopping criteria
    # ------------------------------------------------------------------------------------

    maximum_value_is_rel    = parameters["maximum_value_is_rel"]
    maximum_value_orig      = parameters["maximum_value"]
    minimum_inputs_variation   = parameters["minimum_inputs_variation"]

    res_base = -mitim_bo.BOmetrics["overall"]["Residual"][0].item()

    maximum_value = max_val(maximum_value_orig, maximum_value_is_rel, res_base)

    # ------------------------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------------------------

    if minimum_inputs_variation is not None:
        converged_by_dvs, yvals = stopping_criteria_by_dvs(mitim_bo, minimum_inputs_variation)
    else:
        converged_by_dvs = False
        yvals = None

    if maximum_value is not None:
        converged_by_value, yvals = stopping_criteria_by_value(mitim_bo, maximum_value)
    else:
        converged_by_value = False
        yvals = None

    converged = converged_by_value or converged_by_dvs
    
    return converged, yvals

def stopping_criteria_by_value(mitim_bo, maximum_value):

    # Grab scalarized objectives for each case
    print("\t- Checking maximum value so far...")
    _, _, maximization_value = mitim_bo.scalarized_objective(torch.from_numpy(mitim_bo.train_Y).to(mitim_bo.dfT))
    yvals = maximization_value.cpu().numpy()

    # Best case (maximization)
    best_value_so_far = np.nanmax(yvals)

    # Converged?
    print(f'\t\t* Best scalar function so far (to maximize): {best_value_so_far:.3e} (threshold: {maximum_value:.3e})')
    criterion_is_met = best_value_so_far > maximum_value

    return criterion_is_met, -yvals

def stopping_criteria_by_dvs(mitim_bo, minimum_inputs_variation):

    print("\t- Checking DV variations...")
    _, yG_max = TESTtools.DVdistanceMetric(mitim_bo.train_X)

    criterion_is_met = (
        mitim_bo.currentIteration
        >= minimum_inputs_variation[0]
        + minimum_inputs_variation[1]
    )
    for i in range(int(minimum_inputs_variation[1])):
        criterion_is_met = criterion_is_met and (
            yG_max[-1 - i] < minimum_inputs_variation[2]
        )

    if criterion_is_met:
        print(
            f"\t\t* DVs varied by less than {minimum_inputs_variation[2]}% compared to the rest of individuals for the past {int(minimum_inputs_variation[1])} iterations"
        )
    else:
        print(
            f"\t\t* DVs have varied by more than {minimum_inputs_variation[2]}% compared to the rest of individuals for the past {int(minimum_inputs_variation[1])} iterations"
        )

    return criterion_is_met, yG_max

def read_from_scratch(file):
    """
    This reads a pickle file for the entire class
    """

    optimization_object = opt_evaluator(None)
    mitim = MITIM_BO(optimization_object, onlyInitialize=True, askQuestions=False)
    mitim = mitim.read(file=file, iteration=-1, provideFullClass=True)

    return mitim

def avoidClassInitialization(folderWork):
    print("It was requested that I try read the class before I initialize and select parameters...",typeMsg="i")

    try:
        aux = IOtools.unpickle_mitim(folderWork / "Outputs" / "optimization_object.pkl")
        opt_fun = aux.optimization_object
        cold_start = False
        print("\t- cold_start was successful", typeMsg="i")
    except:
        opt_fun = None
        cold_start = True
        flagger = print("\t- cold_start was requested but it didnt work (c)", typeMsg="q")
        if not flagger:
            embed()

    return opt_fun, cold_start

def clean_state(folder):
    '''
    This function cleans the a read pickle file to avoid problems with reading cases run in a different machine
    '''        

    print(">><<>><< Cleaning state of the class...", typeMsg="i")

    aux = read_from_scratch(folder / "Outputs" / "optimization_object.pkl")

    if aux is not None:
        
        from mitim_modules.portals import PORTALStools, PORTALSmain

        if isinstance(aux.optimization_object, PORTALSmain.portals):
            aux.optimization_options['convergence_options']['stopping_criteria'] = PORTALStools.stopping_criteria_portals

        aux.folderOutputs = folder / "Outputs"
        aux.timings_file = aux.folderOutputs / "timing.jsonl"

        aux.save()

    print(">><<>><< Cleaning state of the class... Done", typeMsg="i")
