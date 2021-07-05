##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201
from .search_model_darts    import TinyNetworkDarts, MobileNetDarts
from .search_model_gdas     import TinyNetworkGDAS, MobileNetGDAS
from .search_model_setn     import TinyNetworkSETN, MobileNetSETN
from .search_model_enas     import TinyNetworkENAS, MobileNetENAS
from .search_model_random   import TinyNetworkRANDOM, MobileNetRANDOM
from .genotypes             import Structure as CellStructure, architectures as CellArchitectures
# NASNet-based macro structure
from .search_model_gdas_nasnet import NASNetworkGDAS
from .search_model_darts_nasnet import NASNetworkDARTS


nas201_super_nets = {'DARTS-V1': TinyNetworkDarts,
                     "DARTS-V2": TinyNetworkDarts,
                     "GDAS": TinyNetworkGDAS,
                     "SETN": TinyNetworkSETN,
                     "ENAS": TinyNetworkENAS,
                     "RANDOM": TinyNetworkRANDOM}

mobilenet_super_nets = {'DARTS-V1': MobileNetDarts,
                     "DARTS-V2": MobileNetDarts,
                     "GDAS": MobileNetGDAS,
                     "SETN": MobileNetSETN,
                     "ENAS": MobileNetENAS,
                     "RANDOM": MobileNetRANDOM}

nasnet_super_nets = {"GDAS": NASNetworkGDAS,
                     "DARTS": NASNetworkDARTS}
