import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, answer_classes, embed_dim=300, hidden_dim=256):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_fc = nn.Linear(512, hidden_dim)

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.classifier = nn.Linear(hidden_dim * 2, answer_classes)

    def forward(self, images, questions):
        img_feat = self.cnn(images).squeeze()
        img_feat = self.cnn_fc(img_feat)

        q_embed = self.embedding(questions)
        _, (h_n, _) = self.lstm(q_embed)
        q_feat = h_n[-1]

        fused = torch.cat([img_feat, q_feat], dim=1)

        output = self.classifier(fused)
        return output
