import json

def load_json(path):
    with open(path, 'r') as f:
        js = json.load(f)
    
    return js

def assert_type(name, data, expected_type):
    assert type(data) == expected_type, "'{}' type must be {}. You provided {}".format(name, expected_type, type(data))