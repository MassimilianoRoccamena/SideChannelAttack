import os
import numpy as np

from omegaconf import OmegaConf

from main.base.utils.misc import withrepr

# Adds the ${eval:expression} resolver to omegaconf's parsing
# This enables runtime evaluation of python expressions in configs
# E.g., ${eval:"${params.val1} if ${params.cond} else ${params.val2}"}
OmegaConf.register_new_resolver("eval", lambda x: eval(x))

# Adds the ${runtime_eval:expression} resolver to omegaconf's parsing
# This extends the previous by encapsulating the expression into a function
# that evaluates the expression each time. This is useful when one wants to
# generate random values each time the option is evaluated
# E.g., ${runtime_eval: np.random.randint(0,100)}
OmegaConf.register_new_resolver("runtime_eval", lambda x: withrepr(x)(lambda: eval(x)))

def parse_args():
    # Retrieve the base config path
    #base_path = os.path.join(os.path.dirname(__file__), '../configs/base.yml')

    # Retrieve the default config
    #args = OmegaConf.load(base_path)

    # Read the cli args
    cli_args = OmegaConf.from_cli()

    assert cli_args.config
    args = OmegaConf.load(cli_args.config)
    # args = OmegaConf.merge(args, conf_args)

    # Merge cli args into config ones
    args = OmegaConf.merge(args, cli_args)

    return args