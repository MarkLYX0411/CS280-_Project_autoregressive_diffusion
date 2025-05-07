from FMT.config.AutoregressiveFMTConfig import AutoregressiveFMTConfig

def find_task_config(task_name: str):
    if task_name == 'fmt':
        return AutoregressiveFMTConfig
    else:
        raise NotImplementedError