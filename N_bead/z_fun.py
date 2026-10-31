
import sys
def get_solver(ver):
    if 'chain-v6' in ver:
        from solver_chain_v6 import StressModel as md
    elif 'chain-v7' in ver:
        from solver_chain_v7 import StressModel as md
    elif 'chain-v8' in ver:
        from solver_chain_v8 import StressModel as md
    elif 'chain-v9' in ver:
        from solver_chain_v9 import StressModel as md
    elif 'chain-v10' in ver:
        from solver_chain_v10 import StressModel as md
    elif 'chain-v11' in ver:
        from solver_chain_v11 import StressModel as md
    elif 'chain-v12' in ver:
        from solver_chain_v12 import StressModel as md
    elif 'chain-v13' in ver:
        from solver_chain_v13 import StressModel as md
    elif 'chain-v14' in ver:
        from solver_chain_v14 import StressModel as md
    elif 'chain-v15' in ver:
        from solver_chain_v15 import StressModel as md
    elif 'chain-v16' in ver:
        from solver_chain_v16 import StressModel as md
    elif 'chain-v17' in ver:
        from solver_chain_v17 import StressModel as md
    elif 'chain-v18' in ver:
        from solver_chain_v18 import StressModel as md


    elif 'chain-N15-v1' in ver:
        from solver_chain_N15_v1 import StressModel as md
    elif 'chain-N15-v2' in ver:
        from solver_chain_N15_v2 import StressModel as md
    elif 'chain-N15-v3' in ver:
        from solver_chain_N15_v3 import StressModel as md
    elif 'chain-N15-v4' in ver:
        from solver_chain_N15_v4 import StressModel as md
    elif 'chain-N15-v5' in ver:
        from solver_chain_N15_v5 import StressModel as md
    elif 'chain-N15-v6' in ver:
        from solver_chain_N15_v6 import StressModel as md

    elif 'loop-v1' in ver:
        from solver_loop_v1 import StressModel as md
    elif 'star-v6' in ver:
        from solver_star_v6 import StressModel as md
    elif 'star-v7' in ver:
        from solver_star_v7 import StressModel as md
    elif 'star-v8' in ver:
        from solver_star_v8 import StressModel as md
    elif 'star-v11' in ver:
        from solver_star_v11 import StressModel as md
    else:
        sys.exit('solver version not found')
    
    if 'chain-N15' in ver:
        save_prex='model_chain_N15/'
    elif 'chain' in ver:
        save_prex='model_chain_N7/'
    elif 'star' in ver:
        save_prex='model_star_N7/'
    elif 'loop' in ver:
        save_prex='model_loop_N7/'
    return md,save_prex