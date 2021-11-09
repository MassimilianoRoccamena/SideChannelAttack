# Deep Learning Techniques for Side Channel Attack

Thesis work by Massimiliano Roccamena.

## *AIDENV*

It's the main component of the application and is used as a general framework for developing artifical agents. It provides to an *aidenv user component* an interface for different execution environments:

- ***mlearn*** *(not implemented)*
  - train and/or test a classical machine learning model
- ***dlearn***
  - train and/or test a deep learning model

You can launch the app by running one execution environment, specifying its identifier and the arguments being passed to it (currently no additional arguments are being used)

```bash
sh app.sh $env_id $args
```

You can configure the execution flow of the component by modifying the corresponding *yaml* file inside *config* folder:

- the field *base.origin* can be used to specify the package of the *aidenv user component*
- the field *base.prompt* can be used to specify the task inside the *aidenv user component*
- all other fields are specific configurations of each execution environment

## *SCA*

It's the target component of the application which realizes the goals of the thesis. It's the only *aidenv user component* inside the application.
