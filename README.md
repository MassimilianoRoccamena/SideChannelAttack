# Deep Learning Techniques for Side Channel Attack

Thesis work by Massimiliano Roccamena

### Running

You can launch the app by running one of the executable scripts, specifying the identifier and the arguments being passed to it

```bash
sh run.sh $script_id $args
```

Currently no additional arguments are being used

Main executable identifiers are:

- ***deep-gym***
  - train and test a deep model

Other executables identifiers are:

- ***mining***
  - fetch and visualize informations of main
- ***testing***
  - run tests of main

### Neptune

You can setup your neptune username by running

```bash
sh utils/netpune/set-user.sh $user_name
```

You can setup your neptune token by running

```bash
sh utils/netpune/set-token.sh $api_token
```
