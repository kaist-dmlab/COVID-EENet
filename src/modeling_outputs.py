#%%
from dataclasses import dataclass, fields
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union, List
from collections import OrderedDict

import tensorflow as tf

'''
Thanks to https://github.com/huggingface
Reference https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_outputs.py
'''
class ModelOutput(OrderedDict):
    def __post_init__(self):
        class_fields = fields(self)
        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not tf.is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True

            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

@dataclass
class COVIDNetOutputs(ModelOutput):
    """Class for keeping track of an item in inventory."""
    BS : tf.Tensor = None            # SM [batch, 34]
    CS : tf.Tensor = None    # SC [batch, 34 ,27]
    BuzCusStructureSimilarity : tf.Tensor = None                # SI [batch, 34]
    EmbeddingTarget : tf.Tensor = None              # Embedding [batch, 34, 20]
    EmbeddingInfected : tf.Tensor = None            # Embedding [batch, 34 ,20]

    OS : tf.Tensor = None                # SO [batch, 34,]
    OutbreakBusinessEmb : tf.Tensor = None          # [batch, 20]    

    P_Dist : tf.Tensor = None                   # [batch, 1]
    C_Dist: tf.Tensor = None                    # [batch, 1]
    GER : tf.Tensor = None                    # Geography [batch, 1]

    EPR : tf.Tensor = None
    composite_lst : tf.Tensor = None        # combination of three component, Economic/Geography/Epidemic
    composite_outputs : tf.Tensor = None    # mean(composite_lst , axis = -1)

@dataclass
class MainOutputs(ModelOutput):
    """Class for keeping track of an item in inventory."""
    ComponentOutputs : List[tf.Tensor] = None           # # of mass infection
    MARAttnWeight : tf.Tensor = None    # event attention
    MAR : tf.Tensor = None           # composite_outputs /dot event-attention
    WeekEmb : tf.Tensor = None                          # Weekday Embedding
    MARWeekEmb : tf.Tensor = None    # shortterminsensitivity + weekEmb
    MARWeekEmbFCN : tf.Tensor = None # FCN(shortterminsensitivityWeekEmb)
    CovidImpact : tf.Tensor = None                      
