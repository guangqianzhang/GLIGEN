import os
import torch as th



class GroundingNetInput:
    def __init__(self):
        self.set = False

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the
        input only for the ground tokenizer.
        """

        self.set = True

        sem=batch['sem']
        mask=batch['mask']
        depth=batch['depth']
        self.batch, self.sem_C, self.sem_H, self.sem_W = sem.shape
        self.batch, self.dep_C, self.dep_H, self.dep_W = depth.shape
        self.device = sem.device
        self.dtype = sem.dtype

        return {"sem":sem, 'depth': depth,"mask":mask}


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference,
        please define the null input for the grounding tokenizer
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        sem = th.zeros(self.batch, self.sem_C, self.sem_H, self.sem_W).type(dtype).to(device)
        depth = th.zeros(self.batch, self.dep_C, self.dep_H, self.dep_W).type(dtype).to(device)
        mask = th.zeros(batch).type(dtype).to(device)

        return {"sem":sem, "depth":depth,  "mask":mask}







