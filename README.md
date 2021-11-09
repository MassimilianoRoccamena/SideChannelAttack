# Deep Learning Techniques for Side Channel Attack

Thesis work by Massimiliano Roccamena

## Instructions

In the following sections are described the main coponents of application and how to interact with them

### AIDENV

It is the main component of the application and it is used as a general framework for developing artifical agents. It provides to an ***aidenv program*** an execution platform called ***aidenv environment***. Currently there are the following environments:

- ***mlearn*** *(not implemented)*
  - train and/or test a classical machine learning model
- ***dlearn***
  - train and/or test a deep learning model

You can define an *aidenv program* by providing an ***aidenv configuration*** *yaml* file:

- the field *base.origin* can be used to specify the package of the *aidenv program*
- the field *base.prompt* can be used to specify the task inside the *aidenv program*
- all other fields are specific configurations of each *aidenv environment*

You can launch an *aidenv environment* specifying its identifier and the arguments being passed to the *aidenv program*

```bash
sh run/environment.sh $environment_id $program_args
```

You must also specify the following environmental variable

- ***AIDENV_CONFIG***

And if you want to use *Neptune* you must specify also also

- ***NEPTUNE_USER***
- ***NEPTUNE TOKEN***
- ***NEPTUNE_PROJECT***

You can launch one *aidenv program* from default configuration folder (*config* folder) by specyfing its name (subfolder of *config*)

```bash
sh run/origram.sh $environment_id $program_name $program_args
```

And if you want to use *Neptune* you must also provide your account data inside *config/{program}/neptune/(user,token,project).conf*

### SCA

It's the target component of the application which realizes the goals of the thesis. It's the only *aidenv program* inside the application.

You can the component by running

```bash
sh sca.sh $environment_id
```

The only environment currently used by *sca* is *dlearn*

### Testing

You can launch test for the overall system by running

```bash
sh test.sh
```
