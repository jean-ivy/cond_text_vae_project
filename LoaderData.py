import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class PascalLoadData(Dataset):
    def __init__(self, names, images, descriptions, vocab):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(size=(500, 500), pad_if_needed=True, fill=0),
            transforms.ToTensor()])

        self.names = names
        self.images = images
        self.descriptions = descriptions

        self.vocab = vocab
        self.vocab_keys = vocab.keys()
        self.max_len = 25

    def __getitem__(self, index):
        image = self.images[self.names[index]]
        image = self.transform(image)
        description = self.descriptions[self.names[index]]

        caption = [self.vocab[token] if token in self.vocab_keys else self.vocab['<UNK>'] for token in description]
        caption = [self.vocab['<BOS>']] + caption + [self.vocab['<EOS>']]
        caption = caption[:self.max_len] + (self.max_len - len(caption)) * [self.vocab['<PAD>']]
        caption = torch.LongTensor(caption)

        return image, caption

    def __len__(self):
        return len(self.images)