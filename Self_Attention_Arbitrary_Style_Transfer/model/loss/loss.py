import torch.nn as nn
import torch

class Loss():
  def __init__(self):
    self.mse_loss = nn.MSELoss()

  def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

  def calc_style_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      input_mean, input_std = calc_mean_std(input)
      target_mean, target_std = calc_mean_std(target)
      return self.mse_loss(input_mean, target_mean) + \
             self.mse_loss(input_std, target_std)

  def __call__(self,content,style,style_feats,content_feats, Ics, Ics_feats, Icc, Icc_feats, Iss, Iss_feats):
      #content_loss
      loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
      
      #style_loss
      loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
      for i in range(1, 5):
          loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
          
      #Identity losses lambda 1
             
      loss_lambda1 = self.calc_content_loss(Icc,content)+self.calc_content_loss(Iss,style)
      
      #Identity losses lambda 2
      Icc_feats=self.encode_with_intermediate(Icc)
      Iss_feats=self.encode_with_intermediate(Iss)
      loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
      for i in range(1, 5):
          loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])
          
      return loss_c, loss_s, loss_lambda1, loss_lambda2
