import yaml

with open('preset.yaml') as file:
    try:
        databaseConfig = yaml.safe_load(file)
        print(databaseConfig)
    except yaml.YAMLError as exc:
        print(exc)
