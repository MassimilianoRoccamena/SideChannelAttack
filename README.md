# Deep Learning Techniques for Side Channel Attack

Thesis work by Massimiliano Roccamena

In the following sections there is a description of the main components of the application and how to interact with them

## AIDENV

This is the platform component of the application and it is a basic small framework for developing artifical agents. It provides to an ***aidenv program*** an execution platform named ***aidenv environment***. Currently there are the following environments:

- ***dprocess***
  - execute computations on some data
- ***dlearn***
  - train and/or test a deep learning model
- ***mlearn***
  - train and/or test a classical machine learning model

You can define an *aidenv program* by writing an ***aidenv configuration*** *yaml* file:

- the field *base.origin* can be used to specify the core package of the *aidenv program* task module
- all other fields are specific configurations of each *aidenv environment*

The following environmental variable are used by *aidenv*:

- ***AIDENV_INPUT***
  - directory path of the data used by the program
- ***AIDENV_OUTPUT***
  - program log directory path
- ***AIDENV_PROGRAM***
  - path to an *aidenv configuration* program file

And if you want to use *Neptune* the system also uses the following ones:

- ***AIDENV_NEPTUNE_USER***
- ***AIDENV_NEPTUNE_TOKEN***
- ***AIDENV_NEPTUNE_PROJECT***

You can define all these variable from the configuration folder of *aidenv* which is *config/aidenv*
An *aidenv program* configuration files are stored in *config/{program_name}*

You can launch *aidenv* by specifying an *aidenv environment* identifier, an *aidenv program* name and the arguments being passed to it

```bash
sh aidenv.sh $environment_id $program_name $program_args
```

## SCA

It's the target component of the application which realizes the goals of the thesis. It include some *aidenv program*. In order to launch an SCA task, just submit the following commands

```bash
sh run/sca/{path_to_program}.sh
```

Inside the *preprocessing* package there is the *window* program, which basically allows to compute lookup tables for creating a dataset of fixed frequency power traces.

Inside the *profiling* package there are programs for fitting models to attack the traces. The fully deep related tasks are:

- *deep-static*
  - for fitting a model on static frequency traces
- *deep-dynamic*
  - for fitting a model on dynamic frequency traces

Inside the *profiling.classic* package there are programs for fitting methods which include the Template Attack. Inside *profiling.classic.aligned* there are programs for fitting the deep aligned template attack. The overall tasks for classic-related attacks are:

- *basic*
  - for fitting the PCA+QDA model on static traces
- *aligned.classification*
  - for fitting the frequency detector by classification on static frequency trace windows
- *aligned.segmentation.dynamic*
  - implement the Grad-CAM on the frequency detector for dynamic traces
- *aligned.segmentation.static*
  - implement the Grad-CAM on the frequency detector for static traces

Inside the *attacking* package there are task for attacking traces, after previous computation of corresponding profiled tasks. The fully deep related tasks are:

- *deep-static*
  - attack device without RDFS
- *deep-dynamic*
  - attack device with RDFS

Inside the *attacking.classic* package there are programs for attackingg methods which include the Template Attack. The overall tasks for classic-related attacks are:

- *basic*
  - attacking with PCA+QDA method the device without RDFS
- *aligned-static*
  - scale the whole traces without RDFS to a given frequency, then apply the corresponding PCA+QDA model
- *aligned-dynamic*
  - attacking with PCA+QDA method with previous Grad-CAM+interpolation frequency alignment

### Testing

(OUT OF DATE)