import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class LoadData(Dataset):
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
        self.max_len = 50

    def __getitem__(self, index):
        image = self.images[self.names[index]]
        image = self.transform(image)
        description = self.descriptions[self.names[index]]
        flat_description = [item for sublist in description for item in sublist]

        caption = [self.vocab[token] if token in self.vocab_keys else self.vocab['<UNK>'] for token in flat_description]
        caption = [self.vocab['<BOS>']] + caption + [self.vocab['<EOS>']]
        caption = caption[:self.max_len] + (self.max_len - len(caption)) * [self.vocab['<PAD>']]
        caption = torch.LongTensor(caption)

        return image, caption

    def __len__(self):
        return len(self.images)