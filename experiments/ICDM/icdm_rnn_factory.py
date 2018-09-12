import mxnet as mx
from nowcasting.config import cfg
from nowcasting.hko_evaluation import rainfall_to_pixel
from nowcasting.encoder_forecaster import EncoderForecasterBaseFactory
from nowcasting.operators import *
from nowcasting.ops import *


def get_loss_weight_symbol(data, mask, seq_len):
    if cfg.MODEL.USE_BALANCED_LOSS:
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = mx.sym.ones_like(data) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (data >= threshold)
        weights = weights * mask
    else:
        weights = mask
    if cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "same":
        return weights
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "linear":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        temporal_mult = 1 + \
                        mx.sym.arange(start=0, stop=seq_len) * (upper - 1.0) / (seq_len - 1.0)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        weights = mx.sym.broadcast_mul(weights, temporal_mult)
        return weights
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "exponential":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        base_factor = np.log(upper) / (seq_len - 1.0)
        temporal_mult = mx.sym.exp(mx.sym.arange(start=0, stop=seq_len) * base_factor)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        weights = mx.sym.broadcast_mul(weights, temporal_mult)
        return weights
    else:
        raise NotImplementedError


class ICDMFactory(EncoderForecasterBaseFactory):
    def __init__(self,
                 batch_size,
                 in_seq_len,
                 out_seq_len,
                 ctx_num=1,
                 name="icdm_nowcasting"):
        super(ICDMFactory, self).__init__(batch_size=batch_size,
                                                   in_seq_len=in_seq_len,
                                                   out_seq_len=out_seq_len,
                                                   ctx_num=ctx_num,
                                                   height=cfg.ICDM.IMG_SIZE,
                                                   width=cfg.ICDM.IMG_SIZE,
                                                   name=name)

    def loss_sym(self):
        """Construct loss symbol.

        Optional args:
            pred: Shape (out_seq_len, batch_size, C, H, W)
            target: Shape (out_seq_len, batch_size, C, H, W)
        """
        self.reset_all()
        pred = mx.sym.Variable('pred')  # Shape: (out_seq_len, batch_size, 1, H, W)
        target = mx.sym.Variable('target')  # Shape: (out_seq_len, batch_size, 1, H, W)
        avg_mse = mx.sym.mean(mx.sym.square(target - pred))
        avg_mse = mx.sym.MakeLoss(avg_mse,
                                  name="mse")
        loss = mx.sym.Group([avg_mse])
        return loss
