class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() 

    def forward(self, logits, targets): 

        if len(targets.shape) == 3:  
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits) 
        smooth = 1e-5  

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        bce_loss = self.bce(logits, targets)

        return bce_loss + dice_loss

criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
