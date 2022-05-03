import segmentation_models_pytorch as smp

def build_smp_model(config):
    '''
    config:
        decoder:
        args: {

        }
    '''

    return getattr(smp, config['decoder'])(**config['args'])
