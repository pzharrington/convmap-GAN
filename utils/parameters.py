from ruamel.yaml import YAML


def load_params(yaml_file, config_name):
    paramdict = {}
    with open(yaml_file) as yamlfile:
        for key, val in YAML().load(yamlfile)[config_name].items():
            if val =='None':
                val = None
            elif val == 'False':
                val = False
            elif val == 'True':
                val == True
            paramdict[key]=val
    return paramdict
