import parlai.core.build_data as build_data
import os

def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'reddit')
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('Dummy!!! [building data: ' + dpath + ']')

        build_data.mark_done(dpath, version_string=version)
