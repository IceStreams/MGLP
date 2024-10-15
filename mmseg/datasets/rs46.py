# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RS46Dataset(CustomDataset):
    """ISPRS dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('Dryland', 'Orchard', 'Tea Garden', 'Mulberry Garden',
               'Rubber garden', 'Nursery', 'Flower Bed', 'Other economic seedlings',
               'The Tree Forest','Bush','Joe irrigation mixed forest','Bamboo forest',
               'Shulin', 'Green woodlands', 'Artificial young forest', 'Sparsely irrigated grass',
               'Natural grassland', 'Artificial grassland', 'Multi-storey and above building area', 'Low-rise building area',
               'Abandoned house construction area', 'Multi-storey and above detached house construction', 'Low-rise detached house construction', 'Highway',
               'Rail', 'Hardened surface', 'Hydraulic facilities', 'City wall',
               'Greenhouses, greenhouses', 'Curing tank', 'Industrial facilities', 'Sand barrier',
               'Other structures', 'Open pit', 'Stacking', 'Construction site',
               'Other artificial excavation', 'Saline-alkali surface', 'Soil surface', 'Sandy surface',
               'Gravel surface', 'Rocky surface', 'Rivers and canals', 'Water surface',
               'Glaciers and perennial snow', 'Paddy Field')

    PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
               [128, 128, 64], [192, 0, 32]]

    def __init__(self, **kwargs):
        super(RS46Dataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
