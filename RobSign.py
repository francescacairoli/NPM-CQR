import torch
import torch.nn as nn
import torch.nn.functional as F
from QR import QR

class RobSign(QR):

	def __init__(self, input_size = 2, hidden_size = 50, output_size = 2):
		super().__init__(input_size, hidden_size, output_size)

		self.fc_add = nn.Linear(output_size, 3)


	def forward(self, x):

		drop_prob = 0.01
		
		x = self.fc_in(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=drop_prob)(x)

		x = self.fc_1(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=drop_prob)(x)

		x = self.fc_2(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=drop_prob)(x)

		x = self.fc_3(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=drop_prob)(x)

		x = self.fc_out(x)

		x = self.fc_add(x)

		return x
