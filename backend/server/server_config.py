from enum import Enum, auto

from backend.aggregation_techniques.fedavg import fedavg
from backend.aggregation_techniques.median import median_aggregation
from backend.aggregation_techniques.trim import trim_mean
from backend.aggregation_techniques.KeTS import KeTS
from backend.aggregation_techniques.KeTS_MedTrim import KeTS_MedTrim
from backend.aggregation_techniques.KeTSV2 import KeTSV2
from backend.aggregation_techniques.krum import krum
from backend.aggregation_techniques.testing import  cluster_similarity_fedavg

class AggregationStrategy(Enum):
    FEDAVG = 'fedavg'
    KRUM = 'krum'
    MEDIAN =  'median'
    TRIM_MEAN =  'trim_mean'
    KeTS = 'KeTS'
    FLTRUST =  'FLTrust'
    KeTSV2 =  'KeTSV2'
    KeTS_MedTrim =  'KeTS_MedTrim'
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
    
    

from backend.attacks.gaussian import gaussian_attack
from backend.attacks.label_flip import label_flip_attack
from backend.attacks.min_max import min_max_attack
from backend.attacks.min_sum import min_sum_attack
from backend.attacks.sign_flip import sign_flip_attack
from backend.attacks.trim_att import trim_attack
from backend.attacks.no_att import no_attack
from backend.attacks.krum_att import krum_attack
from backend.attacks.research_att import (
    min_max_attack_rand_noise,
    min_sum_attack_rand_noise,
    min_max_attack_ortho,
    min_sum_attack_ortho
)

class AttackType(Enum):
    NO_ATTACK = 'no_attack'
    MIN_MAX = 'min_max'
    MIN_SUM =  'min_sum'
    KRUM = 'krum'
    TRIM =  'trim'
    GAUSSIAN = 'gaussian'
    LABEL_FLIP = 'label_flip'
    MIN_MAX_V2 = 'min_max_v2'
    MIN_SUM_V2 = 'min_sum_v2'
    SIGN_FLIP = 'sign_flip'
    MIN_MAX_V3 = 'min_max_v3'
    MIN_SUM_V3 =  'min_sum_v3'

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