# Deep Learning Techniques for Side Channel Attack

Thesis work by Massimiliano Roccamena

## Instructions

In the following sections are described the main coponents of application and how to interact with them

### *aidenv*

This is the platform component of the application and it is used as a general framework for developing artifical agents. It provides to an ***aidenv program*** an execution platform called ***aidenv environment***. Currently there are the following environments:

- ***mining*** *(small usage)*
  - fetch and show info about something
- ***dlearn***
  - train and/or test a deep learning model
- ***mlearn*** *(not implemented)*
  - train and/or test a classical machine learning model

You can define an *aidenv program* by writing an ***aidenv configuration*** *yaml* file:

- the field *base.origin* can be used to specify the package of the *aidenv program*
- the field *base.prompt* can be used to specify the task inside the *aidenv program*
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

### *sca*

It's the target component of the application which realizes the goals of the thesis. It's the only *aidenv program* inside the application.

You can  launch the component by running

```bash
sh sca.sh $environment_id
```

The only environment currently used by *sca* is *dlearn*

### Testing

You can launch test for the main system by running

```bash
sh test.sh
```
