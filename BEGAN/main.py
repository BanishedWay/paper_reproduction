from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocess import combine_data


# 针对每个类别实现生成器
attack_cat_counts = combine_data["class"].value_counts()

attack_categories = sorted(combine_data["class"].unique())
# num_data_per_category = [attack_cat_counts.get(cat, 0) for cat in attack_categories]
# print(num_data_per_category)

y = combine_data["class"]
X = combine_data.drop("class", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

from Generator import Generator
from Discriminator import Discriminator

num = 0

gen_list = []

for i, cat in enumerate(attack_categories):
    print(f"Training for {cat}")
    X_train_cat = X_train[y_train == cat]
    y_train_cat = y_train[y_train == cat]
    y_train_cat = y_train_cat.replace(cat, 1)

    X_train_cat = torch.tensor(X_train_cat.values, dtype=torch.float32)
    y_train_cat = torch.tensor(y_train_cat.values, dtype=torch.float32)

    input_dim = 200
    output_dim = X_train_cat.shape[1]

    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(output_dim)

    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(X_train_cat, y_train_cat)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    batch_size = 64
    num_epochs = 150
    display_step = 25

    num = num + 1
    print(num)
    for epoch in range(num_epochs):

        for i, (real_features, labels) in enumerate(dataloader):
            real_label = Variable(torch.ones(real_features.size(0), 1))
            fake_label = Variable(torch.zeros(real_features.size(0), 1))

            z = Variable(torch.randn(real_features.size(0), input_dim))
            fake_data = generator(z)

            discriminator_optimizer.zero_grad()
            d_real = discriminator(real_features)
            d_fake = discriminator(fake_data)
            # print(real_label)
            loss_real = criterion(d_real, real_label)
            loss_fake = criterion(d_fake, fake_label)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            discriminator_optimizer.step()

            z = Variable(torch.randn(real_features.size(0), input_dim))
            fake_data = generator(
                z
            )  # TODO 这里不能重新生成，应该用上面的fake_data，但是生成会直接报错

            d_fake = discriminator(fake_data)
            loss_g = criterion(d_fake, real_label)
            generator_optimizer.zero_grad()
            loss_g.backward()
            generator_optimizer.step()

        if epoch % display_step == 0:
            print(f"Epoch: {epoch} Loss D: {loss_d.item()} Loss G: {loss_g.item()}")

    with torch.no_grad():
        z = torch.randn(1000, input_dim)
        fake_data = generator(z)
        fake_data = fake_data.numpy()

        fake_data[:, :-1] = cat
        gen_list.append(fake_data)

print(gen_list)
