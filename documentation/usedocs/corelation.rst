Correlator
=======================================
The Correlator package contains the main image correlation routines.

First we need to import it::

    import muDIC as dic

Solver settings
---------------------------------
In order for us to run a DIC analysis, we have to prepare the inputs by generatin a settings object::

    settings = dic.DIC_settings(image_stack,mesh)


A image stack and a mesh has to be passed to the DIC_settings class.
The DIC_settings class contains all the settings the image correaltion routines need for performing the analysis.
Default values are used when the settings object is instanciated.

If we want to alter any settings, for instance set the frames at which to to a reference update, we can do::

    settings.update_ref_frames = [50,124,197]


or, if we want to set the increment size used as convergenve criterion by the solver::

    settings.convergence_inc = 1e-5

Running an analysis
---------------------------------

We are now ready for running a DIC-analysis. We now make a DIC-job object, and call the .run() method::

    dic_job = dic.DIC_job(settings)
    results = dic_job.run()

**Note that the results of the analysis are returned by the .run() method.**



