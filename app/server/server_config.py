from app.attacks.research_att import (
    min_max_attack_rand_noise,
    min_sum_attack_rand_noise,
    min_max_attack_ortho,
    min_sum_attack_ortho
)
from app.attacks.krum_att import krum_attack
from app.attacks.no_att import no_attack
from app.attacks.trim_att import trim_attack
from app.attacks.sign_flip import sign_flip_attack
from app.attacks.min_sum import min_sum_attack
from app.attacks.min_max import min_max_attack
from app.attacks.label_flip import label_flip_attack
from app.attacks.gaussian import gaussian_attack
from enum import Enum, auto

from app.aggregation_techniques.fedavg import fedavg
from app.aggregation_techniques.median import median_aggregation
from app.aggregation_techniques.trim import trim_mean
from app.aggregation_techniques.KeTS import KeTS
from app.aggregation_techniques.KeTS_MedTrim import KeTS_MedTrim
from app.aggregation_techniques.KeTSV2 import KeTSV2
from app.aggregation_techniques.krum import krum
from app.aggregation_techniques.testing import cluster_similarity_fedavg


class AggregationStrategy(Enum):
    FEDAVG = 'fedavg'
    KRUM = 'krum'
    MEDIAN = 'median'
    TRIM_MEAN = 'trim_mean'
    KeTS = 'KeTS'
    FLTRUST = 'FLTrust'
    KeTSV2 = 'KeTSV2'
    KeTS_MedTrim = 'KeTS_MedTrim'
    Testing = 'testing'


aggregation_methods = {
    AggregationStrategy.FEDAVG: fedavg,
    AggregationStrategy.KRUM: krum,
    AggregationStrategy.MEDIAN: median_aggregation,
    AggregationStrategy.TRIM_MEAN: trim_mean,
    AggregationStrategy.KeTS: KeTS,
    AggregationStrategy.FLTRUST: None,
    AggregationStrategy.KeTSV2: KeTSV2,
    AggregationStrategy.KeTS_MedTrim: KeTS_MedTrim,
    AggregationStrategy.Testing: cluster_similarity_fedavg
}


class AttackType(Enum):
    NO_ATTACK = 'no_attack'
    MIN_MAX = 'min_max'
    MIN_SUM = 'min_sum'
    KRUM = 'krum'
    TRIM = 'trim'
    GAUSSIAN = 'gaussian'
    LABEL_FLIP = 'label_flip'
    MIN_MAX_V2 = 'min_max_v2'
    MIN_SUM_V2 = 'min_sum_v2'
    SIGN_FLIP = 'sign_flip'
    MIN_MAX_V3 = 'min_max_v3'
    MIN_SUM_V3 = 'min_sum_v3'


attacks = {
    AttackType.MIN_MAX: min_max_attack,
    AttackType.MIN_SUM: min_sum_attack,
    AttackType.KRUM: krum_attack,
    AttackType.TRIM: trim_attack,
    AttackType.NO_ATTACK: no_attack,
    AttackType.GAUSSIAN: gaussian_attack,
    AttackType.LABEL_FLIP: label_flip_attack,
    AttackType.MIN_MAX_V2: min_max_attack_rand_noise,
    AttackType.MIN_SUM_V2: min_sum_attack_rand_noise,
    AttackType.SIGN_FLIP: sign_flip_attack,
    AttackType.MIN_MAX_V3: min_max_attack_ortho,
    AttackType.MIN_SUM_V3: min_sum_attack_ortho
}
