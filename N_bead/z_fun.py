import re
import sys


def get_solver(ver):
    # Baseline folder names include their parent model; select the last label.
    labels = re.findall(r'(chain|star)-(PDE_Law|PDE|VAR_Law|VAR)(?=_|$)', ver)
    if not labels:
        sys.exit('solver name not found')
    topology, name = labels[-1]
    if topology == 'chain' and name == 'PDE_Law':
        from solver_chain_PDE_Law import StressModel as md
    elif topology == 'chain' and name == 'PDE':
        from solver_chain_PDE import StressModel as md
    elif topology == 'chain' and name == 'VAR':
        from solver_chain_VAR import StressModel as md
    elif topology == 'chain' and name == 'VAR_Law':
        from solver_chain_VAR_Law import StressModel as md
    elif topology == 'star' and name == 'VAR_Law':
        from solver_star_VAR_Law import StressModel as md
    else:
        sys.exit('solver name not found')

    save_prex = 'model_star_N7/' if topology == 'star' else 'model_chain_N7/'
    return md, save_prex
