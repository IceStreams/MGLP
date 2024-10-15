# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder
from .encoder_decoder_test1 import EncoderDecoder_Test1
from .encoder_decoder_test3 import EncoderDecoder_Test3
from .encoder_decoder_test4 import EncoderDecoder_Test4
from .encoder_decoder_test5 import EncoderDecoder_Test5
from .encoder_decoder_test6 import EncoderDecoder_Test6
from .encoder_decoder_test7 import EncoderDecoder_Test7
from .encoder_decoder_test8 import EncoderDecoder_Test8
from .encoder_decoder_test8_1 import EncoderDecoder_Test8_1
from .encoder_decoder_test9 import EncoderDecoder_Test9
from .encoder_decoder_test9_fine import EncoderDecoder_Test9_fine
from .encoder_decoder_test9_coarse import EncoderDecoder_Test9_coarse

from .encoder_decoder_test9_inference import EncoderDecoder_Test9_Inference
from .encoder_decoder_test8_inference import EncoderDecoder_Test8_Inference

from .encoder_decoder_test8_1 import EncoderDecoder_Test8_1
from .encoder_decoder_test8_1_inference import EncoderDecoder_Test8_1_Inference

from .encoder_decoder_test8_rs import EncoderDecoder_Test8_rs

from .encoder_decoder_test9_rs import EncoderDecoder_Test9_rs

from .encoder_decoder_test8_rs_inference import EncoderDecoder_Test8_Inference_rs

from .encoder_decoder_test9_without_coarse_pull import EncoderDecoder_Test9_without_coarse_pull

from .encoder_decoder_DHSS import EncoderDecoder_DHSS

from .encoder_decoder_test9_rs_inference import EncoderDecoder_Test9_rs_Inference

from .encoder_decoder_test9_without_coarse_pull_coarse2 import EncoderDecoder_Test9_without_coarse_pull_coarse2

from .encoder_decoder_test9_without_horizontal import EncoderDecoder_Test9_without_horizontal
from .encoder_decoder_test9_rs_without_horizontal import EncoderDecoder_Test9_rs_without_horizontal

from .encoder_decoder_test9_blu import EncoderDecoder_Test9_blu

from .cascade_encoder_decoder_test9 import CascadeEncoderDecoder_Test9

from .cascade_encoder_decoder_test9_inference import CascadeEncoderDecoder_Test9_inference
from .cascade_encoder_decoder_test9_rs import CascadeEncoderDecoder_Test9_rs

from .encoder_decoder_test9_danet import EncoderDecoder_Test9_danet
from .encoder_decoder_test9_danet_inference import EncoderDecoder_Test9_danet_inference

from .encoder_decoder_test9_coarse_fine import EncoderDecoder_Test9_coarse_fine

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'EncoderDecoder_Test1', 'EncoderDecoder_Test3',
           'EncoderDecoder_Test4', 'EncoderDecoder_Test5', 'EncoderDecoder_Test6', 'EncoderDecoder_Test7', 'EncoderDecoder_Test8',
           'EncoderDecoder_DHSS', 'EncoderDecoder_Test9', 'EncoderDecoder_Test9_fine', 'EncoderDecoder_Test9_coarse',
           'EncoderDecoder_Test9_without_coarse_pull', 'EncoderDecoder_Test9_Inference', 'EncoderDecoder_Test8_Inference',
           'EncoderDecoder_Test8_Inference', 'EncoderDecoder_Test8_1_Inference', 'EncoderDecoder_Test8_rs',
           'EncoderDecoder_Test8_Inference_rs', 'EncoderDecoder_Test9_rs', 'EncoderDecoder_Test9_rs_Inference', 'EncoderDecoder_Test9_without_coarse_pull_coarse2',
           'EncoderDecoder_Test9_without_horizontal', 'EncoderDecoder_Test9_rs_without_horizontal', 'EncoderDecoder_Test9_blu',
           'CascadeEncoderDecoder_Test9', 'CascadeEncoderDecoder_Test9_inference', 'CascadeEncoderDecoder_Test9_rs',
           'EncoderDecoder_Test9_danet', 'EncoderDecoder_Test9_danet_inference', 'EncoderDecoder_Test9_coarse_fine'
           ]
