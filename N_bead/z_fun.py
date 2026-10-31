import sys


def get_solver(ver):
    if 'chain-v8' in ver:
        from solver_chain_v8 import StressModel as md
    elif 'chain-v9' in ver:
        from solver_chain_v9 import StressModel as md
    elif 'chain-v10' in ver:
        from solver_chain_v10 import StressModel as md
    elif 'chain-v11' in ver:
        from solver_chain_v11 import StressModel as md
    elif 'star-v11' in ver:
        from solver_star_v11 import StressModel as md
    else:
        sys.exit('solver version not found')

    save_prex = 'model_star_N7/' if 'star' in ver else 'model_chain_N7/'
    return md, save_prex
