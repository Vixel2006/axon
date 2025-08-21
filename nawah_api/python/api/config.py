class Config:
    def __init__(
        self, epochs=10, batch_size=32, lr=0.001, device="cpu", log_interval=100
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.log_interval = log_interval
