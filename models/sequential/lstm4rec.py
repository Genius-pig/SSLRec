from torch import nn

from config.configurator import configs


class LSTM4Rec(nn.Module):
    def __init__(self, data_handler):
        super(LSTM4Rec, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.rnn = nn.LSTM(self.emb_size, self.emb_size, batch_first=True)
        self.ph_dense = self.dense_layer(self.emb_size, self.item_num + 1)
        self.items = nn.Embedding(self.item_num, self.emb_size, max_norm=1)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data
        item_embs = self.items(batch_seqs)
