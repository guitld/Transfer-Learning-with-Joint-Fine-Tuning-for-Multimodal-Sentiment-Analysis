import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from transformers import DistilBertModel
from torchmetrics import Accuracy, F1Score, AUROC

class SentimentClassifier(pl.LightningModule):
    def __init__(self, lr, finetune=True, n_classes=2):
        super().__init__()

        if n_classes == 2:
            self.loss = nn.BCEWithLogitsLoss()
            self.output = n_classes - 1
            self.accuracy = Accuracy()
            self.f1 = F1Score()
            self.auc = AUROC()
        else:
            self.loss = nn.CrossEntropyLoss()
            self.accuracy = Accuracy(num_classes=n_classes)
            self.f1 = F1Score(num_classes=n_classes, average='macro')
            self.auc = AUROC(num_classes=n_classes)
            self.output = n_classes
    
        self.lr = lr
        self.n_classes = n_classes
        
        # DistilBert
        self.textual_extractor = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # ResNet
        backbone = torchvision.models.resnet50(pretrained=True)
        backbone.eval()
        self.vis_shape = backbone.fc.in_features
        layers = list(backbone.children())[:-1]


        # Fusion Layers
        self.visual_extractor = nn.Sequential(*layers)

        self.vis_fusion_layer = nn.Sequential(
                                            nn.Linear(in_features=self.vis_shape, out_features=32),
                                            nn.ReLU(inplace=True))
        self.txt_fusion_layer = nn.Sequential(
                                            nn.Linear(in_features=self.textual_extractor.config.hidden_size, out_features=32),
                                            nn.ReLU(inplace=True))
        self.fusion_layer = nn.Sequential(
                                            nn.Linear(in_features=32, out_features=32),
                                            nn.ReLU(inplace=True))

        if finetune is not True:
            for param in self.visual_extractor.parameters():
                param.requires_grad = False
            for param in self.textual_extractor.parameters():
                param.requires_grad = False

        # Classifier layers
        self.attention = torch.nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=False, dropout=0.4)
        self.clf_layer = nn.Linear(in_features=64, out_features=128)

        self.classifier = nn.Sequential(
                                        nn.Dropout(),
                                        self.clf_layer,
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(in_features=128, out_features=self.output))

    def forward(self, x):
        img, txt = x[0], x[1]
        txt = self.textual_extractor(txt[0], attention_mask=txt[1])
        img = self.visual_extractor(img).flatten(1)
        x = self.fusion(img, txt.last_hidden_state[:,0,:]).squeeze()
        x = self.classifier(x)
        return x

    def fusion(self, img, txt):
        img = self.vis_fusion_layer(img)
        txt = self.txt_fusion_layer(txt)
        img = img.unsqueeze(dim=0)
        txt = txt.unsqueeze(dim=0)
        return self.attention(torch.cat((img, txt), dim=2))

    def get_embeddings(self, x):
        img, txt = x[0], x[1]
        txt = self.textual_extractor(txt[0], attention_mask=txt[1])
        img = self.visual_extractor(img).flatten(1)
        x = self.fusion(img, txt.last_hidden_state[:,0,:]).squeeze()
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.forward(x)
        if self.n_classes == 2:
            labels = labels.unsqueeze(1)
            loss = self.loss(logits, labels.float())
        else:
            loss = loss = self.loss(logits, labels)
        acc = self.accuracy(logits, labels)
        f1 = self.f1(logits, labels)
        auc = self.auc(logits, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc)
        self.log("train_f1", f1)
        self.log("train_auc", auc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.forward(x)
        if self.n_classes == 2:
            labels = labels.unsqueeze(1)
            loss = self.loss(logits, labels.float())
        else:
            loss = loss = self.loss(logits, labels)
        acc = self.accuracy(logits, labels)
        f1 = self.f1(logits, labels)
        auc = self.auc(logits, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc)
        self.log("val_f1", f1)
        self.log("val_auc", auc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.forward(x)
        if self.n_classes == 2:
            labels = labels.unsqueeze(1)
            loss = self.loss(logits, labels.float())
        else:
            loss = loss = self.loss(logits, labels)
        acc = self.accuracy(logits, labels)
        f1 = self.f1(logits, labels)
        auc = self.auc(logits, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", acc)
        self.log("test_f1", f1)
        self.log("test_auc", auc)
        return loss
