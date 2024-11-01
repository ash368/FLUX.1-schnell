INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["A cat holding a sign that says hello world"]
    },
    "height": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [512]
    },
    "width": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [512]
    },
    "num_inference_steps": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [4]
    },
    "guidance_scale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [7.5]
    }
}
